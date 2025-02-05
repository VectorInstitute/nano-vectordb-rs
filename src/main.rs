use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, HashSet};
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;
use base64::{engine::general_purpose, Engine as _};

const F_ID: &str = "__id__";
const F_METRICS: &str = "__metrics__";
type Float = f32;

#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
    #[serde(rename = "__id__")]
    pub id: String,
    #[serde(skip)]
    pub vector: Vec<Float>,
    #[serde(flatten)]
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataBase {
    embedding_dim: usize,
    data: Vec<Data>,
    matrix: Vec<Vec<Float>>,
    additional_data: HashMap<String, serde_json::Value>,
}

impl DataBase {
    fn matrix_base64(&self) -> Result<String> {
        let flat: Vec<Float> = self.matrix.iter().flatten().copied().collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                flat.as_ptr() as *const u8,
                flat.len() * std::mem::size_of::<Float>(),
            )
        };
        Ok(general_purpose::STANDARD.encode(bytes))
    }
}

#[derive(Debug)]
pub struct NanoVectorDB {
    pub embedding_dim: usize,
    pub metric: String,
    storage_file: PathBuf,
    storage: DataBase,
}

impl NanoVectorDB {
    pub fn new(embedding_dim: usize, storage_file: &str) -> Result<Self> {
        let storage_file = PathBuf::from(storage_file);
        let storage = if storage_file.exists() {
            let contents = fs::read_to_string(&storage_file)?;
            let mut db: DataBase = serde_json::from_str(&contents)?;
            db.matrix = decode_matrix(&db.matrix_base64()?, embedding_dim)?;
            db
        } else {
            DataBase {
                embedding_dim,
                data: Vec::new(),
                matrix: Vec::new(),
                additional_data: HashMap::new(),
            }
        };

        Ok(Self {
            embedding_dim,
            metric: "cosine".to_string(),
            storage_file,
            storage,
        })
    }

    pub fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    pub fn get_metric(&self) -> &str {
        &self.metric
    }

    #[allow(dead_code)]
    fn matrix_base64(&self) -> Result<String> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.storage.matrix.as_ptr() as *const u8,
                self.storage.matrix.len() * std::mem::size_of::<Float>(),
            )
        };
        Ok(general_purpose::STANDARD.encode(bytes))
    }

    pub fn upsert(&mut self, mut datas: Vec<Data>) -> Result<(Vec<String>, Vec<String>)> {
        let mut updates = Vec::new();
        let mut inserts = Vec::new();
        let existing_ids: HashSet<_> = self.storage.data.iter().map(|d| &d.id).collect();

        for data in datas.iter_mut() {
            if existing_ids.contains(&data.id) {
                if let Some(pos) = self.storage.data.iter().position(|d| d.id == data.id) {
                    self.storage.matrix[pos] = data.vector.clone();
                    updates.push(data.id.clone());
                }
            }
        }

        let new_datas: Vec<Data> = datas
            .into_iter()
            .filter(|d| !existing_ids.contains(&d.id))
            .collect();

        for data in new_datas {
            self.storage.data.push(data);
            inserts.push(self.storage.data.last().unwrap().id.clone());
        }

        self.storage.matrix = self.storage.data
            .iter()
            .map(|d| normalize(&d.vector))
            .collect();

        Ok((updates, inserts))
    }

    pub fn query(
        &self,
        query: &[Float],
        top_k: usize,
        better_than: Option<Float>,
        filter: Option<&dyn Fn(&Data) -> bool>,
    ) -> Vec<HashMap<String, serde_json::Value>> {
        let query_norm = normalize(query);
        let mut scores = Vec::with_capacity(self.storage.data.len());

        for (idx, data) in self.storage.data.iter().enumerate() {
            if let Some(filter) = filter {
                if !filter(data) {
                    continue;
                }
            }
            
            let score = dot_product(&query_norm, &self.storage.matrix[idx]);
            scores.push((score, idx));
        }

        scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        scores.iter()
            .take(top_k)
            .filter(|(score, _)| better_than.map(|t| *score >= t).unwrap_or(true))
            .map(|(score, idx)| {
                let mut result = self.storage.data[*idx].fields.clone();
                result.insert(F_METRICS.to_string(), serde_json::json!(score));
                result.insert(F_ID.to_string(), serde_json::json!(self.storage.data[*idx].id));
                result
            })
            .collect()
    }

    pub fn save(&self) -> Result<()> {
        let serialized = serde_json::to_string(&self.storage)?;
        fs::write(&self.storage_file, serialized)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct MultiTenantNanoVDB {
    embedding_dim: usize,
    max_capacity: usize,
    storage_dir: PathBuf,
    tenants: HashMap<String, NanoVectorDB>,
    lru: VecDeque<String>,
}

impl MultiTenantNanoVDB {
    pub fn new(embedding_dim: usize, max_capacity: usize, storage_dir: &str) -> Self {
        let storage_dir = PathBuf::from(storage_dir);
        // Create directory if it doesn't exist
        if !storage_dir.exists() {
            fs::create_dir_all(&storage_dir).expect("Failed to create storage directory");
        }
        
        Self {
            embedding_dim,
            max_capacity,
            storage_dir,
            tenants: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub fn create_tenant(&mut self) -> String {
        let tenant_id = Uuid::new_v4().to_string();
        let storage_path = self.storage_dir.join(format!("{tenant_id}.json"));
        
        // Ensure directory exists before creating tenant
        if !self.storage_dir.exists() {
            fs::create_dir_all(&self.storage_dir).expect("Failed to create tenant directory");
        }

        let db = NanoVectorDB::new(self.embedding_dim, storage_path.to_str().unwrap())
            .expect("Failed to create tenant DB");
        
        self.lru.push_back(tenant_id.clone());
        self.tenants.insert(tenant_id.clone(), db);
        
        if self.tenants.len() > self.max_capacity {
            if let Some(oldest) = self.lru.pop_front() {
                self.tenants.remove(&oldest);
            }
        }
        
        tenant_id
    }

    pub fn get_tenant(&mut self, tenant_id: &str) -> Option<&mut NanoVectorDB> {
        if let Some(index) = self.lru.iter().position(|id| id == tenant_id) {
            self.lru.remove(index);
        }
        self.lru.push_back(tenant_id.to_string());
        self.tenants.get_mut(tenant_id)
    }
}

// Helper functions
fn normalize(vector: &[Float]) -> Vec<Float> {
    let norm = vector.iter()
        .map(|x| x.powi(2))
        .sum::<Float>()
        .sqrt();
    vector.iter().map(|x| x / norm).collect()
}

fn dot_product(a: &[Float], b: &[Float]) -> Float {
    a.iter().zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

fn decode_matrix(base64_str: &str, dim: usize) -> Result<Vec<Vec<Float>>> {
    let bytes = general_purpose::STANDARD.decode(base64_str)?;
    let floats: Vec<Float> = bytes.chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            Float::from_le_bytes(bytes)
        })
        .collect();
    
    if floats.len() % dim != 0 {
        anyhow::bail!("Invalid matrix dimensions");
    }
    
    Ok(floats.chunks(dim)
        .map(|chunk| chunk.to_vec())
        .collect())
}

fn main() -> Result<()> {
    use std::time::Instant;
    use std::fs;

    let embedding_dim = 1024;
    let num_vectors = 100_000;
    let query_vector = vec![0.2; embedding_dim];

    // Clean up previous runs
    let _ = fs::remove_file("data.json");
    let _ = fs::remove_dir_all("tenants");

    // Initialize database
    let mut db = NanoVectorDB::new(embedding_dim, "data.json")?;

    // Benchmark insertion
    let insert_start = Instant::now();
    let mut data_vec = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        data_vec.push(Data {
            id: format!("vec_{i}"),  // Fixed format string
            vector: vec![0.1; embedding_dim],
            fields: HashMap::new(),
        });
    }
    let (_updates, inserts) = db.upsert(data_vec)?;
    let insert_duration = insert_start.elapsed();
    
    println!("Embedding Dim: {embedding_dim}");
    println!("Inserted {} vectors in {:.2}ms", 
        inserts.len(),
        insert_duration.as_secs_f64() * 1000.0
    );

    // Benchmark query
    let query_start = Instant::now();
    let results = db.query(&query_vector, 10, None, None);
    let query_duration = query_start.elapsed();
    
    println!("Queried {} vectors in {:.2}ms",
        num_vectors,
        query_duration.as_secs_f64() * 1000.0
    );
    
    // Safe result handling
    if let Some(top_result) = results.get(0) {
        println!("Top result score: {:.4}", 
            top_result.get(F_METRICS).unwrap().as_f64().unwrap()
        );
    } else {
        println!("No results found");
    }

    // Save and report size
    db.save()?;
    let size = fs::metadata("data.json")?.len() as f64 / 1_000_000.0;
    println!("Storage size: {:.1}MB", size);

    // Multi-tenant example with cleanup
    let mut multi_db = MultiTenantNanoVDB::new(embedding_dim, 10, "tenants");
    let tenant_id = multi_db.create_tenant();
    let _tenant = multi_db.get_tenant(&tenant_id).unwrap();

    // Clean up after benchmark
    if PathBuf::from("data.json").exists() {
        fs::remove_file("data.json")?;
    }
    
    let tenants_dir = PathBuf::from("tenants");
    if tenants_dir.exists() && tenants_dir.is_dir() {
        fs::remove_dir_all("tenants")?;
    }

    Ok(())
}


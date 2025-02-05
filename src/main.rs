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
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataBase {
    embedding_dim: usize,
    data: Vec<Data>,
    #[serde(with = "base64_bytes")]
    matrix: Vec<Float>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    additional_data: HashMap<String, serde_json::Value>,
}

mod base64_bytes {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(vec: &[Float], serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vec.as_ptr() as *const u8,
                vec.len() * std::mem::size_of::<Float>(),
            )
        };
        let b64 = general_purpose::STANDARD.encode(bytes);
        serializer.serialize_str(&b64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Float>, D::Error> {
        let s = String::deserialize(deserializer)?;
        let bytes = general_purpose::STANDARD.decode(s).map_err(serde::de::Error::custom)?;
        Ok(bytes.chunks_exact(4)
            .map(|chunk| Float::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
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
            let db: DataBase = serde_json::from_str(&contents)?; // Removed mut
            
            // Verify matrix dimensions
            let expected_len = db.data.len() * db.embedding_dim;
            if db.matrix.len() != expected_len {
                anyhow::bail!("Matrix size mismatch: expected {}, got {}", expected_len, db.matrix.len());
            }
            
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

    pub fn upsert(&mut self, mut datas: Vec<Data>) -> Result<(Vec<String>, Vec<String>)> {
        let mut updates = Vec::new();
        let mut inserts = Vec::new();
        let existing_ids: HashSet<_> = self.storage.data.iter().map(|d| &d.id).collect();

        // Process updates
        for data in datas.iter_mut() {
            if existing_ids.contains(&data.id) {
                if let Some(pos) = self.storage.data.iter().position(|d| d.id == data.id) {
                    let norm_vec = normalize(&data.vector);
                    let start = pos * self.embedding_dim;
                    let end = start + self.embedding_dim;
                    self.storage.matrix[start..end].copy_from_slice(&norm_vec);
                    updates.push(data.id.clone());
                }
            }
        }

        // Process inserts
        let new_datas: Vec<Data> = datas
            .into_iter()
            .filter(|d| !existing_ids.contains(&d.id))
            .collect();

        for data in new_datas {
            let norm_vec = normalize(&data.vector);
            let vec_clone = norm_vec.clone(); // Clone the normalized vector
            self.storage.matrix.extend(vec_clone);
            self.storage.data.push(Data {
                id: data.id.clone(),
                vector: norm_vec,
                fields: data.fields,
            });
            inserts.push(data.id);
        }

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
            
            let start = idx * self.embedding_dim;
            let vector = &self.storage.matrix[start..start + self.embedding_dim];
            let score = dot_product(&query_norm, vector);
            scores.push((score, idx));
        }

        scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        scores.iter()
            .take(top_k)
            .filter(|(score, _)| better_than.map(|t| *score >= t).unwrap_or(true))
            .map(|(score, idx)| {
                let data = &self.storage.data[*idx];
                let mut result = data.fields.clone();
                result.insert(F_METRICS.to_string(), serde_json::json!(score));
                result.insert(F_ID.to_string(), serde_json::json!(data.id));
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

// Updated main function with optimizations
fn main() -> Result<()> {
    use std::time::Instant;
    use std::fs;

    let embedding_dim = 1024;
    let num_vectors = 100_000;
    let query_vector = vec![0.2; embedding_dim];

    // Cleanup previous runs
    let _ = fs::remove_file("data.json");
    let _ = fs::remove_dir_all("tenants");

    let mut db = NanoVectorDB::new(embedding_dim, "data.json")?;

    // Insert benchmark
    let insert_start = Instant::now();
    let data_vec: Vec<Data> = (0..num_vectors)
        .map(|i| Data {
            id: format!("vec_{i}"),
            vector: vec![0.1; embedding_dim],
            fields: HashMap::new(),
        })
        .collect();
    
    let (_, inserts) = db.upsert(data_vec)?;
    let insert_duration = insert_start.elapsed();

    println!("Embedding Dim: {embedding_dim}");
    println!("Inserted {} vectors in {:.2}ms", 
        inserts.len(),
        insert_duration.as_secs_f64() * 1000.0
    );

    // Query benchmark
    let query_start = Instant::now();
    let results = db.query(&query_vector, 10, None, None);
    let query_duration = query_start.elapsed();
    
    println!("Queried {} vectors in {:.2}ms",
        num_vectors,
        query_duration.as_secs_f64() * 1000.0
    );
    
    if let Some(top_result) = results.get(0) {
        println!("Top result score: {:.4}", 
            top_result.get(F_METRICS).unwrap().as_f64().unwrap()
        );
    }

    // Storage size
    db.save()?;
    let size = fs::metadata("data.json")?.len() as f64 / 1_000_000.0;
    println!("Storage size: {:.1}MB", size);

    // Cleanup
    fs::remove_file("data.json")?;
    Ok(())
}


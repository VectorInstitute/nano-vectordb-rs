//! A lightweight vector database implementation
#![warn(missing_docs)]

use anyhow::Result;
use base64::{engine::general_purpose, Engine as _};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

/// Constants used for special field names
pub mod constants {
    /// Identifier field name
    pub const F_ID: &str = "__id__";
    /// Similarity metrics field name
    pub const F_METRICS: &str = "__metrics__";
}

type Float = f32;

/// A single vector entry with metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
    /// Unique identifier for the vector
    #[serde(rename = "__id__")]
    pub id: String,
    /// The vector data (non-normalized)
    #[serde(skip)]
    pub vector: Vec<Float>,
    /// Additional metadata fields stored with the vector
    #[serde(flatten, skip_serializing_if = "HashMap::is_empty")]
    pub fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataBase {
    embedding_dim: usize,
    data: Vec<Data>,
    #[serde(with = "base64_bytes")]
    matrix: Vec<Float>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
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
        let bytes = general_purpose::STANDARD
            .decode(s)
            .map_err(serde::de::Error::custom)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| Float::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }
}

/// Main vector database struct
#[derive(Debug)]
pub struct NanoVectorDB {
    /// Dimensionality of stored vectors
    pub embedding_dim: usize,
    /// Distance metric used for similarity searches
    pub metric: String,
    storage_file: PathBuf,
    storage: DataBase,
}

#[derive(PartialEq)]
struct ScoredIndex {
    score: Float,
    index: usize,
}

impl Eq for ScoredIndex {}

impl PartialOrd for ScoredIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoredIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl NanoVectorDB {
    /// Creates a new NanoVectorDB instance
    pub fn new(embedding_dim: usize, storage_file: &str) -> Result<Self> {
        let storage_file = PathBuf::from(storage_file);
        let storage = if storage_file.exists() && storage_file.metadata()?.len() > 0 {
            let contents = fs::read_to_string(&storage_file)?;
            let db: DataBase = serde_json::from_str(&contents)?;

            let expected_len = db.data.len() * db.embedding_dim;
            if db.matrix.len() != expected_len {
                anyhow::bail!(
                    "Matrix size mismatch: expected {}, got {}",
                    expected_len,
                    db.matrix.len()
                );
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

    /// Upserts vectors into the database
    pub fn upsert(&mut self, mut datas: Vec<Data>) -> Result<(Vec<String>, Vec<String>)> {
        let mut updates = Vec::new();
        let mut inserts = Vec::new();
        let existing_ids: HashSet<_> = self.storage.data.iter().map(|d| &d.id).collect();

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

        let new_datas: Vec<Data> = datas
            .into_iter()
            .filter(|d| !existing_ids.contains(&d.id))
            .collect();

        for data in new_datas {
            let norm_vec = normalize(&data.vector);
            let vec_clone = norm_vec.clone();
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

    /// Queries the database for similar vectors
    pub fn query(
        &self,
        query: &[Float],
        top_k: usize,
        better_than: Option<Float>,
        filter: Option<Box<dyn Fn(&Data) -> bool + Send + Sync>>,
    ) -> Vec<HashMap<String, serde_json::Value>> {
        let query_norm = normalize(query);
        let embedding_dim = self.embedding_dim;
        let matrix = &self.storage.matrix;
        let threshold = better_than.unwrap_or(Float::MIN);

        // Precompute query chunks for SIMD-friendly operations
        let query_chunks: Vec<[Float; 4]> = query_norm
            .chunks_exact(4)
            .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
            .collect();
        let query_remainder = &query_norm[query_chunks.len() * 4..];

        // Parallel processing with Rayon
        let heap = matrix
            .par_chunks(embedding_dim)
            .enumerate()
            .filter(|(idx, _)| {
                filter
                    .as_ref()
                    .map(|f| f(&self.storage.data[*idx]))
                    .unwrap_or(true)
            })
            .fold(
                || BinaryHeap::with_capacity(top_k + 1),
                |mut heap, (idx, vector)| {
                    let score = dot_product(vector, &query_chunks, query_remainder);

                    if score >= threshold {
                        heap.push(ScoredIndex { score, index: idx });
                        if heap.len() > top_k {
                            heap.pop();
                        }
                    }
                    heap
                },
            )
            .reduce(
                || BinaryHeap::with_capacity(top_k + 1),
                |mut heap1, heap2| {
                    for si in heap2 {
                        heap1.push(si);
                        if heap1.len() > top_k {
                            heap1.pop();
                        }
                    }
                    heap1
                },
            );

        // Convert to sorted results
        let mut sorted = heap.into_sorted_vec();
        sorted.reverse();

        sorted
            .into_iter()
            .map(|si| {
                let data = &self.storage.data[si.index];
                let mut result = data.fields.clone();
                result.insert(
                    constants::F_METRICS.to_string(),
                    serde_json::json!(si.score),
                );
                result.insert(constants::F_ID.to_string(), serde_json::json!(data.id));
                result
            })
            .collect()
    }

    /// Saves the database to disk
    pub fn save(&self) -> Result<()> {
        let serialized = serde_json::to_string(&self.storage)?;
        fs::write(&self.storage_file, serialized)?;
        Ok(())
    }

    /// Get the number of vectors in the database (added for test access)
    pub fn len(&self) -> usize {
        self.storage.data.len()
    }

    /// Check if database is empty (common Rust idiom)
    pub fn is_empty(&self) -> bool {
        self.storage.data.is_empty()
    }
}

#[inline]
fn dot_product(vec: &[Float], query_chunks: &[[Float; 4]], query_remainder: &[Float]) -> Float {
    let mut sum = 0.0;
    let mut vec_chunks = vec.chunks_exact(4);

    // Process chunks of 4 elements
    for (i, chunk) in vec_chunks.by_ref().enumerate() {
        let q = query_chunks[i];
        sum += chunk[0] * q[0] + chunk[1] * q[1] + chunk[2] * q[2] + chunk[3] * q[3];
    }

    // Process remainder elements
    sum + vec_chunks
        .remainder()
        .iter()
        .zip(query_remainder)
        .map(|(a, b)| a * b)
        .sum::<Float>()
}

fn normalize(vector: &[Float]) -> Vec<Float> {
    let norm = vector.iter().map(|x| x.powi(2)).sum::<Float>().sqrt();
    vector.iter().map(|x| x / norm).collect()
}

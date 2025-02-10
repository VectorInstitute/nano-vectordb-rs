[![Crates.io](https://img.shields.io/crates/v/nano-vectordb-rs?style=flat-square)](https://crates.io/crates/nano-vectordb-rs)
[![Docs.rs](https://img.shields.io/badge/docs.rs-latest-blue?style=flat-square)](https://docs.rs/nano-vectordb-rs)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://opensource.org/licenses/MIT)
[![Codecov](https://img.shields.io/codecov/c/github/amrit110/nano-vectordb-rs?style=flat-square)](https://codecov.io/github/amrit110/nano-vectordb-rs)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg?logo=rust&style=flat-square)](https://www.rust-lang.org)
<div align="center">
  <h1>nano-vectordb-rs</h1>
  <p><strong>A simple, easy-to-hack vector database in rust</strong></p>
</div>


## Installation

```bash
cargo install nano-vectordb-rs
```

## Quickstart example

```rust
use anyhow::Result;
use nano_vectordb_rs::{constants, Data, NanoVectorDB};
use serde_json::json;
use tempfile::NamedTempFile;

fn main() -> Result<()> {
    // Create temporary storage file
    let temp_file = NamedTempFile::new()?;
    let db_path = temp_file.path().to_str().unwrap();

    // Initialize database with 3-dimensional vectors
    let mut db = NanoVectorDB::new(3, db_path)?;

    // Create sample data with metadata
    let samples = vec![
        Data {
            id: "vec1".into(),
            vector: vec![1.02, 2.0, 3.0],
            fields: [("color".into(), json!("red"))].into(),
        },
        Data {
            id: "vec2".into(),
            vector: vec![-4.0, 5.0, 6.0],
            fields: [("color".into(), json!("blue"))].into(),
        },
        Data {
            id: "vec3".into(),
            vector: vec![7.0, 8.0, -9.0],
            fields: [("color".into(), json!("green"))].into(),
        },
    ];

    // Upsert data and show results
    let (updated, inserted) = db.upsert(samples)?;
    println!("Updated IDs: {:?}", updated);
    println!("Inserted IDs: {:?}\n", inserted);

    // Persist to disk
    db.save()?;

    // Query similar vectors
    let query_vec = vec![0.1, 0.2, 0.3]; // Should be closest to vec1
    let results = db.query(&query_vec, 1, None, None);

    println!("Top 1 result:");
    for result in results {
        println!(
            "- ID: {} | Color: {} | Score: {:.4}",
            result[constants::F_ID],
            result["color"],
            result[constants::F_METRICS]
        );
    }

    // Delete a vector
    db.delete(&["vec3".into()]);
    db.save()?;

    println!("\nAfter deletion:");
    println!("Total vectors: {}", db.len());

    Ok(())
}
```

## Motivation 💡

**Why choose nano-vectordb-rs?** A Rust port of the popular [nano-vectordb](https://github.com/gusye1234/nano-vectordb).

✨ **Key Features**:
- ⚡ Fast cosine similarity searches using Rayon parallelism
- 🧩 Simple API surface
- 📈 Embedded persistence with compact serialization
- 🎯 No abstractions, easy to hack

🏆 **Perfect For**:
- Rust ML pipelines needing lightweight vector storage
- Prototyping semantic search systems
- Educational use (clean, hackable implementation)


## Benchmark 🚀

> Embedding Dim: 1024. Device: MacBook M4

- Saving an index with `100,000` vectors will generate a `~540M` json file.
- Inserting `100,000` vectors in  `~175 ms`.
- Querying from `100,000` vectors in `~13 ms`.

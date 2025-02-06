//! Benchmarking script for the vector database
use nano_vectordb_rs::{constants::F_METRICS, NanoVectorDB};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let embedding_dim = 1024;
    let num_vectors = 100_000;
    let query_vector = vec![0.2; embedding_dim];

    // Initialize database
    let mut db = NanoVectorDB::new(embedding_dim, "benchmark_data.json")?;

    // Benchmark insertion
    let insert_start = Instant::now();
    let data_vec: Vec<_> = (0..num_vectors)
        .map(|i| nano_vectordb_rs::Data {
            id: format!("vec_{i}"),
            vector: vec![0.1; embedding_dim],
            fields: std::collections::HashMap::new(),
        })
        .collect();

    let (_, inserts) = db.upsert(data_vec)?;
    db.save()?; // Explicit save
    let insert_duration = insert_start.elapsed();

    println!("Embedding Dim: {embedding_dim}");
    println!(
        "Inserted {} vectors in {:.2}ms",
        inserts.len(),
        insert_duration.as_secs_f64() * 1000.0
    );

    // Benchmark query
    let query_start = Instant::now();
    let results = db.query(&query_vector, 10, None, None);
    let query_duration = query_start.elapsed();

    println!(
        "Queried {} vectors in {:.2}ms",
        num_vectors,
        query_duration.as_secs_f64() * 1000.0
    );

    if let Some(top_result) = results.get(0) {
        println!(
            "Top result score: {:.4}",
            top_result.get(F_METRICS).unwrap().as_f64().unwrap()
        );
    }

    let size = std::fs::metadata("benchmark_data.json")?.len() as f64 / 1_000_000.0;
    println!("Storage size: {:.1}MB", size);

    // Cleanup with existence check
    if std::path::Path::new("benchmark_data.json").exists() {
        std::fs::remove_file("benchmark_data.json")?;
    }
    Ok(())
}

// A basic example demonstrating the usage of NanoVectorDB for similarity search.
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
    let results = db.query(&query_vec, 2, None, None);

    println!("Top 2 results:");
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

use nanovecdb::NanoVectorDB;
use ndarray::Array1;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    const DIM: usize = 1024;
    const NUM_VECTORS: usize = 100_000;
    
    // Initialize database
    let mut db = NanoVectorDB::new(DIM, "test_db.bin")?;
    
    // Generate test vectors
    let mut vectors = Vec::with_capacity(NUM_VECTORS);
    for i in 0..NUM_VECTORS {
        let vec = Array1::from_vec(vec![i as f32; DIM]);
        vectors.push((format!("vec_{}", i), vec));
    }
    
    // Benchmark upsert
    let start = Instant::now();
    let (updated, inserted) = db.upsert(vectors);
    println!("Upsert {} vectors: {:.2?}", NUM_VECTORS, start.elapsed());
    
    // Benchmark query
    let query = Array1::from_vec(vec![0.5; DIM]);
    let start = Instant::now();
    let results = db.query(query, 10, None, None);
    println!("Query: {:.2?}", start.elapsed());
    
    // Verify results
    println!("Top 5 results:");
    for (i, (data, score)) in results.iter().take(5).enumerate() {
        println!("{}. {}: {:.4}", i+1, data.id, score);
    }
    
    // Benchmark save
    let start = Instant::now();
    db.save()?;
    println!("Save: {:.2?}", start.elapsed());
    
    // Cleanup
    std::fs::remove_file("test_db.bin")?;
    Ok(())
}

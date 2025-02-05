use criterion::{criterion_group, criterion_main, Criterion};
use nanovecdb::NanoVectorDB;
use ndarray::Array1;

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorDB Operations");
    group.sample_size(10);
    
    group.bench_function("Upsert 10k vectors", |b| {
        b.iter(|| {
            let mut db = NanoVectorDB::new(1024, "bench_db.bin").unwrap();
            let vectors = (0..10_000)
                .map(|i| (format!("vec_{}", i), Array1::from_vec(vec![i as f32; 1024])))
                .collect();
            db.upsert(vectors);
            db.save().unwrap();
            std::fs::remove_file("bench_db.bin").unwrap();
        })
    });

    group.bench_function("Query 1M vectors", |b| {
        let mut db = NanoVectorDB::new(1024, "large_db.bin").unwrap();
        let vectors = (0..1_000_000)
            .map(|i| (format!("vec_{}", i), Array1::from_vec(vec![i as f32; 1024])))
            .collect();
        db.upsert(vectors);
        
        b.iter(|| {
            db.query(Array1::from_vec(vec![0.5; 1024]), 10, None, None);
        });
        
        std::fs::remove_file("large_db.bin").unwrap();
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

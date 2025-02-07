//! Benchmarking script with complete metrics
use nano_vectordb_rs::NanoVectorDB;
use rand::Rng;
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        embedding_dim: 1024,
        num_vectors: 1_000_000,
        num_runs: 5,
    };

    let mut metrics = BenchmarkMetrics::new(config.num_runs);

    for run in 0..config.num_runs {
        let filename = format!("benchmark_data_{}.json", run);
        let run_metrics = benchmark_run(&config, &filename)?;
        metrics.add(run_metrics);
        cleanup_file(&filename)?;
    }

    metrics.print();
    Ok(())
}

struct BenchmarkConfig {
    embedding_dim: usize,
    num_vectors: usize,
    num_runs: usize,
}

struct BenchmarkMetrics {
    insert_times: Vec<f64>,
    save_times: Vec<f64>,
    query_times: Vec<f64>,
    file_sizes: Vec<f64>,
}

impl BenchmarkMetrics {
    fn new(capacity: usize) -> Self {
        Self {
            insert_times: Vec::with_capacity(capacity),
            save_times: Vec::with_capacity(capacity),
            query_times: Vec::with_capacity(capacity),
            file_sizes: Vec::with_capacity(capacity),
        }
    }

    fn add(&mut self, metrics: RunMetrics) {
        self.insert_times.push(metrics.insert_time);
        self.save_times.push(metrics.save_time);
        self.query_times.push(metrics.query_time);
        self.file_sizes.push(metrics.file_size);
    }

    fn print(&self) {
        let (ins_mean, ins_std) = calculate_stats(&self.insert_times);
        let (save_mean, save_std) = calculate_stats(&self.save_times);
        let (q_mean, q_std) = calculate_stats(&self.query_times);
        let (size_mean, size_std) = calculate_stats(&self.file_sizes);

        println!("\nBenchmark Results ({} runs):", self.insert_times.len());
        println!("==================================================");
        println!("Operation          | Mean ± Std Dev");
        println!("----------------------------------");
        println!("Insert Time: {:7.2}ms ± {:.2}", ins_mean, ins_std);
        println!("Save Time:   {:7.2}ms ± {:.2}", save_mean, save_std);
        println!("Query Time:  {:7.3}ms ± {:.3}", q_mean, q_std);
        println!("File Size:   {:7.2}MB ± {:.2}", size_mean, size_std);
    }
}

struct RunMetrics {
    insert_time: f64,
    save_time: f64,
    query_time: f64,
    file_size: f64,
}

fn benchmark_run(config: &BenchmarkConfig, filename: &str) -> anyhow::Result<RunMetrics> {
    let mut db = NanoVectorDB::new(config.embedding_dim, filename)?;

    // Generate random vectors for each run
    let mut rng = rand::rng();
    let data_vec: Vec<_> = (0..config.num_vectors)
        .map(|i| {
            let mut vector = vec![0.0; config.embedding_dim];
            rng.fill(&mut vector[..]);

            nano_vectordb_rs::Data {
                id: format!("vec_{}", i),
                vector,
                fields: std::collections::HashMap::new(),
            }
        })
        .collect();

    // Time insertion (upsert only)
    let insert_start = Instant::now();
    let (_, _) = db.upsert(data_vec)?;
    let insert_time = duration_to_ms(insert_start.elapsed());

    // Time saving separately
    let save_start = Instant::now();
    db.save()?;
    let save_time = duration_to_ms(save_start.elapsed());

    // Generate random query vector
    let mut query_vector = vec![0.0; config.embedding_dim];
    rng.fill(&mut query_vector[..]);

    // Time query
    let query_start = Instant::now();
    let _ = db.query(&query_vector, 10, None, None);
    let query_time = duration_to_ms(query_start.elapsed());

    // Get file size
    let file_size = std::fs::metadata(filename)
        .map(|md| md.len() as f64 / 1_000_000.0)
        .unwrap_or(0.0);

    Ok(RunMetrics {
        insert_time,
        save_time,
        query_time,
        file_size,
    })
}

fn cleanup_file(filename: &str) -> anyhow::Result<()> {
    if std::path::Path::new(filename).exists() {
        std::fs::remove_file(filename)?;
    }
    Ok(())
}

fn calculate_stats(data: &[f64]) -> (f64, f64) {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    (mean, variance.sqrt())
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

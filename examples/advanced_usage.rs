// An advanced example demonstrating how to load a dataset and upsert it into NanoVectorDB for similarity search.
use anyhow::Result;
use colored::Colorize;
use hf_hub::api::sync::ApiBuilder;
use nano_vectordb_rs::{constants, Data, NanoVectorDB};
use parquet::file::reader::SerializedFileReader;
use parquet::record::{ListAccessor, RowAccessor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use tempfile::NamedTempFile;

#[derive(Debug, Serialize, Deserialize)]
struct WikiEntity {
    id: String,
    text: String,
    embedding: Vec<f32>,
}

fn main() -> Result<()> {
    // Load dataset
    let api = ApiBuilder::new().build()?;
    let repo = api.dataset("Cohere/wikipedia-22-12-simple-embeddings".to_string());
    let parquet_file = repo.get("data/train-00000-of-00004-1a1932c9ca1c7152.parquet")?;

    // Read Parquet file
    let file = File::open(parquet_file)?;
    let reader = SerializedFileReader::new(file)?;
    let mut row_iter = reader.into_iter().flatten();

    // Take first 1000 samples
    let mut samples = Vec::new();
    for _ in 0..1000 {
        if let Some(row) = row_iter.next() {
            samples.push(row);
        }
    }
    println!(
        "{}",
        format!("Loaded {} samples from Wikipedia dataset", samples.len())
            .green()
            .bold()
    );

    // Initialize vector database
    let temp_file = NamedTempFile::new()?;
    let mut db = NanoVectorDB::new(768, temp_file.path().to_str().unwrap())?;

    // Correct column indices based on dataset schema
    const ID_IDX: usize = 0; // int32
    const TITLE_IDX: usize = 1; // string
    const TEXT_IDX: usize = 2; // string
    const EMBEDDING_IDX: usize = 8; // sequence

    // Prepare data entries
    let data_entries: Vec<Data> = samples
        .iter()
        .map(|record| {
            let mut fields = HashMap::new();

            // Handle ID as int32 and convert to string
            let id = record
                .get_int(ID_IDX)
                .map(|v| v.to_string())
                .unwrap_or_else(|_| "invalid_id".to_string());

            // Handle string fields with proper conversion
            fields.insert(
                "title".to_string(),
                serde_json::json!(record
                    .get_string(TITLE_IDX)
                    .map(|s| s.as_str())
                    .unwrap_or("")),
            );

            fields.insert(
                "text".to_string(),
                serde_json::json!(record
                    .get_string(TEXT_IDX)
                    .map(|s| s.as_str())
                    .unwrap_or("")),
            );

            // Handle embedding sequence
            let list = record.get_list(EMBEDDING_IDX).unwrap();
            let embedding: Vec<f32> = (0..list.len())
                .map(|i| {
                    list.get_float(i)
                        .or_else(|_| list.get_double(i).map(|v| v as f32))
                        .unwrap_or(0.0)
                })
                .collect();

            Data {
                id,
                vector: embedding,
                fields,
            }
        })
        .collect();

    // Upsert data
    let (updates, inserts) = db.upsert(data_entries)?;
    println!(
        "{} {} / {}",
        "Operation complete:".bold().cyan(),
        format!("Updated: {}", updates.len()).yellow(),
        format!("Inserted: {}", inserts.len()).green()
    );
    db.save()?;

    let query_sample = &samples[0];

    println!("\n{}", "Query Sample:".bold().underline());
    println!(
        "{} {}",
        "Title:".bold().cyan(),
        query_sample
            .get_string(TITLE_IDX)
            .map(|s| s.as_str())
            .unwrap_or("[No Title]")
            .italic()
    );
    println!(
        "{} {}",
        "Text:".bold().cyan(),
        query_sample
            .get_string(TEXT_IDX)
            .map(|s| s.as_str())
            .unwrap_or("[No Text]")
            .dimmed()
    );

    // Get query vector
    let list = query_sample.get_list(EMBEDDING_IDX).unwrap();
    let query_vector: Vec<f32> = (0..list.len())
        .map(|i| {
            list.get_float(i)
                .or_else(|_| list.get_double(i).map(|v| v as f32))
                .unwrap_or(0.0)
        })
        .collect();

    // Perform search
    let results = db.query(&query_vector, 5, Some(0.5), None);

    println!("\n{}", "Top 5 Similar Entries:".bold().underline());
    for result in results {
        println!(
            "{} {} {}\n  {}...",
            "-".bold(),
            format!(
                "ID: {}",
                result[constants::F_ID].as_str().unwrap_or("[Invalid ID]")
            )
            .blue(),
            format!("(Score: {:.4})", result[constants::F_METRICS]).yellow(),
            result["text"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(100)
                .collect::<String>()
                .dimmed()
        );
    }

    Ok(())
}

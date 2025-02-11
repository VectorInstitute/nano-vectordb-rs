// A basic example demonstrating the usage of NanoVectorDB for similarity search.
use anyhow::Result;
use colored::Colorize;
use comfy_table::{presets::UTF8_FULL, Attribute, Cell, Color, ContentArrangement, Table};
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

    let mut upsert_table = Table::new();
    upsert_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Operation")
                .fg(Color::Green)
                .add_attribute(Attribute::Bold),
            Cell::new("IDs").fg(Color::Yellow),
        ])
        .add_row(vec![
            Cell::new("Updated").fg(Color::Cyan),
            Cell::new(format!("{:?}", updated)).fg(Color::White),
        ])
        .add_row(vec![
            Cell::new("Inserted").fg(Color::Cyan),
            Cell::new(format!("{:?}", inserted)).fg(Color::White),
        ]);

    println!("\n{}", " UPSERT RESULTS ".bold().on_green().black());
    println!("{upsert_table}");

    // Persist to disk
    db.save()?;

    // Query similar vectors
    let query_vec = vec![0.1, 0.2, 0.3]; // Should be closest to vec1
    let results = db.query(&query_vec, 2, None, None);

    let mut results_table = Table::new();
    results_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("ID")
                .fg(Color::Magenta)
                .add_attribute(Attribute::Bold),
            Cell::new("Color")
                .fg(Color::Blue)
                .add_attribute(Attribute::Bold),
            Cell::new("Score")
                .fg(Color::Yellow)
                .add_attribute(Attribute::Bold),
        ]);

    for result in results {
        let score = result[constants::F_METRICS].as_f64().unwrap_or(0.0);
        results_table.add_row(vec![
            Cell::new(result[constants::F_ID].as_str().unwrap_or("")).fg(Color::White),
            Cell::new(result["color"].as_str().unwrap_or("")).fg(
                match result["color"].as_str().unwrap_or("") {
                    "red" => Color::Red,
                    "green" => Color::Green,
                    "blue" => Color::Blue,
                    _ => Color::White,
                },
            ),
            Cell::new(format!("{:.4}", score)).fg(if score > 0.5 {
                Color::Green
            } else {
                Color::Yellow
            }),
        ]);
    }

    println!("\n{}", " TOP 2 RESULTS ".bold().on_blue().black());
    println!("{results_table}");

    // Delete a vector
    db.delete(&["vec3".into()]);
    db.save()?;

    println!(
        "\n{} {}",
        "Vectors after deletion:".bold().cyan(),
        format!("{}", db.len()).bold().yellow()
    );

    Ok(())
}

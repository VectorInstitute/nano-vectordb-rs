use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct};
use hf_hub::api::sync::Api;
use nano_vectordb_rs::NanoVectorDB;
use std::collections::HashMap;
use tempfile::NamedTempFile;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

#[test]
fn test_large_dataset() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();

    // Insert 1000 vectors
    let vectors = (0..1000)
        .map(|i| nano_vectordb_rs::Data {
            id: format!("vec_{i}"),
            vector: vec![0.1; 128],
            fields: std::collections::HashMap::new(),
        })
        .collect();

    let (updates, inserts) = db.upsert(vectors).unwrap();
    assert_eq!(inserts.len(), 1000);
    assert_eq!(updates.len(), 0);

    // Verify query
    let results = db.query(&vec![0.1; 128], 5, None, None);
    assert_eq!(results.len(), 5);
    assert!(results[0].get("__metrics__").unwrap().as_f64().unwrap() > 0.99);
}

#[test]
fn test_update_operations() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();

    // Initial insert
    let data = nano_vectordb_rs::Data {
        id: "vec_0".to_string(),
        vector: vec![0.1; 128],
        fields: std::collections::HashMap::new(),
    };
    let (_, inserts) = db.upsert(vec![data]).unwrap();
    assert_eq!(inserts.len(), 1);

    // Update
    let updated_data = nano_vectordb_rs::Data {
        id: "vec_0".to_string(),
        vector: vec![0.2; 128],
        fields: std::collections::HashMap::new(),
    };
    let (updates, _) = db.upsert(vec![updated_data]).unwrap();
    assert_eq!(updates.len(), 1);
}

#[test]
fn test_real_embeddings_with_candle() -> Result<()> {
    // Initialize model repo
    let api = Api::new()?;
    let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());

    // Load tokenizer with padding configuration
    let mut tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {:?}", e))?;

    let padding = PaddingParams {
        strategy: PaddingStrategy::Fixed(128),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: tokenizer.get_vocab(true)["[PAD]"],
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    };
    let truncation = TruncationParams {
        max_length: 128,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        direction: TruncationDirection::Left,
    };

    // Handle tokenizer configuration Result
    let _ = tokenizer
        .with_padding(Some(padding))
        .with_truncation(Some(truncation));

    // Configure BERT model
    let config = Config {
        vocab_size: 30522,
        hidden_size: 384,
        num_hidden_layers: 6,
        num_attention_heads: 12,
        intermediate_size: 1536,
        hidden_act: HiddenAct::Gelu,
        ..Default::default()
    };

    let device = Device::Cpu;

    // Load model weights
    let filenames = vec![repo.get("model.safetensors")?];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device) }
        .context("Failed to load safetensors")?;

    let bert = BertModel::load(vb, &config).context("Failed to load BERT model")?;

    // Setup vector database
    let temp_file = NamedTempFile::new()?;
    let mut db = NanoVectorDB::new(384, temp_file.path().to_str().unwrap())?;

    // Test sentences with different relationships
    let sentences = vec![
        ("s1", "The quick brown fox jumps over the lazy dog"),
        ("s2", "A fast brown fox leaps over a sleepy hound"),
        ("s3", "A swift auburn fox vaults above a tired canine"),
        ("s4", "A hungry wolf prowls through the snowy forest"),
        ("s5", "House cats nap peacefully in the sunlight"),
        (
            "s6",
            "Rust's ownership model prevents data races at compile time",
        ),
        ("s7", "Python uses garbage collection for memory management"),
        ("s8", "The Eiffel Tower is located in central Paris"),
    ];

    // Generate embeddings with proper attention masking
    let embeddings = sentences
        .iter()
        .map(|(_, text)| -> Result<Vec<f32>> {
            let encoding = tokenizer
                .encode(*text, true)
                .map_err(|e| anyhow::anyhow!("Encoding error: {:?}", e))?;

            // Convert to U32 for index operations
            let token_ids = Tensor::new(encoding.get_ids(), &device)?.unsqueeze(0)?;
            let attention_mask = Tensor::new(encoding.get_attention_mask(), &device)?
                .to_dtype(DType::U32)? // Changed to U32 for index operations
                .unsqueeze(0)?;

            let embeddings = bert.forward(&token_ids, &attention_mask, None)?;

            // Proper mean pooling with attention mask
            let mask = attention_mask
                .unsqueeze(2)?
                .to_dtype(DType::F32)?
                .expand((1, 128, 384))?;
            let summed = (embeddings * &mask)?.sum_keepdim(1)?;
            let mask_sum = mask.sum_keepdim(1)?.to_dtype(DType::F32)? + 1e-12;
            let pooled = (summed / mask_sum)?
                .squeeze(1)?
                .squeeze(0)?
                .to_vec1::<f32>()?;

            Ok(pooled)
        })
        .collect::<Result<Vec<_>>>()?;

    let query_embedding = embeddings[0].clone();

    // Store raw embeddings
    let test_data = sentences
        .iter()
        .zip(embeddings)
        .map(|((id, text), embedding)| {
            let mut fields = HashMap::new();
            fields.insert("text".to_string(), serde_json::json!(text));

            nano_vectordb_rs::Data {
                id: id.to_string(),
                vector: embedding,
                fields,
            }
        })
        .collect();

    let (updates, inserts) = db.upsert(test_data)?;
    assert!(updates.is_empty());
    assert_eq!(inserts.len(), 8);

    // Query with top 5 results
    let results = db.query(&query_embedding, 5, Some(0.7), None);

    // Verify semantic relationships
    let result_texts: Vec<&str> = results
        .iter()
        .map(|r| r.get("text").unwrap().as_str().unwrap())
        .collect();

    // Expected order: s1, s3, s2 (fox-related), then others
    assert!(
        result_texts[..3].iter().all(|t| t.contains("fox")),
        "Top 3 should be fox-related: {:?}",
        &result_texts[..3]
    );

    // Verify scores are descending
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r.get("__metrics__").unwrap().as_f64().unwrap())
        .collect();

    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "Scores not descending: {:?}",
        scores
    );

    // Verify negative control not in top results
    assert!(
        !result_texts.contains(&"The Eiffel Tower is located in central Paris"),
        "Unrelated result in top"
    );

    Ok(())
}

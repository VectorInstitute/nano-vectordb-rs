use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use nano_vectordb_rs::NanoVectorDB;
use serde_json;
use tempfile::NamedTempFile;
use tokenizers::Tokenizer;

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
fn test_embedding_model_integration() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    // Initialize DB with correct embedding dimension
    let mut db = NanoVectorDB::new(384, path).unwrap();

    // Load model components using Hugging Face Hub API
    let model_id = "BAAI/bge-small-en-v1.5";
    let api = Api::new().unwrap();
    let repo = api.model(model_id.to_string());

    // Load config
    let config_path = repo.get("config.json").unwrap();
    let config_content = std::fs::read_to_string(config_path).unwrap();
    let config: Config = serde_json::from_str(&config_content).unwrap();

    // Load weights
    let weights_path = repo.get("model.safetensors").unwrap();
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).unwrap()
    };

    // Load tokenizer
    let tokenizer_path = repo.get("tokenizer.json").unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // Create model
    let model = BertModel::load(vb, &config).unwrap();

    let sentences = vec![
        "The quick brown fox jumps over the lazy dog",
        "Penguins are flightless birds living in Antarctica",
        "Rust is a systems programming language focusing on safety",
        "Machine learning models require careful evaluation",
    ];

    let mut datas = Vec::new();
    for (i, text) in sentences.iter().enumerate() {
        let encoding = tokenizer.encode(*text, true).unwrap();

        // Use native u32 token IDs from tokenizer
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let token_ids_len = token_ids.len();

        // Create attention mask
        let attention_mask: Vec<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as f32)
            .collect();
        let attention_mask_len = attention_mask.len();

        // Create tensors with proper dtypes
        let tokens_tensor = Tensor::from_vec(token_ids, (token_ids_len,), &device)
            .unwrap()
            .to_dtype(DType::U32) // Must match tokenizer's native type
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let mask_tensor = Tensor::from_vec(attention_mask, (attention_mask_len,), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // Forward pass with corrected dtypes
        let output = model.forward(&tokens_tensor, &mask_tensor, None).unwrap();

        // Mean pooling
        let embeddings = output.mean(1).unwrap().squeeze(0).unwrap();
        let embeddings: Vec<f32> = embeddings.to_dtype(DType::F32).unwrap().to_vec1().unwrap();

        datas.push(nano_vectordb_rs::Data {
            id: format!("vec_{i}"),
            vector: embeddings,
            fields: [(
                "text".to_string(),
                serde_json::Value::String(text.to_string()),
            )]
            .into_iter()
            .collect(),
        });
    }

    // Database operations
    let (updates, inserts) = db.upsert(datas).unwrap();
    assert_eq!(inserts.len(), 4);
    assert_eq!(updates.len(), 0);

    // Query processing
    let query_text = "A fast fox leaps over a sleeping canine";
    let query_encoding = tokenizer.encode(query_text, true).unwrap();

    // Native u32 token IDs
    let query_ids: Vec<u32> = query_encoding.get_ids().to_vec();
    let query_ids_len = query_ids.len();

    let query_mask: Vec<f32> = query_encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as f32)
        .collect();
    let query_mask_len = query_mask.len();

    let query_tensor = Tensor::from_vec(query_ids, (query_ids_len,), &device)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let query_mask_tensor = Tensor::from_vec(query_mask, (query_mask_len,), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let output = model
        .forward(&query_tensor, &query_mask_tensor, None)
        .unwrap();
    let query_embedding: Vec<f32> = output
        .mean(1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();

    // Execute query
    let results = db.query(&query_embedding, 2, None, None);

    // Validate results
    assert_eq!(results.len(), 2);
    let top_result = &results[0];
    assert!(top_result["__metrics__"].as_f64().unwrap() > 0.7);
    assert_eq!(
        top_result["text"],
        "The quick brown fox jumps over the lazy dog"
    );

    db.save().unwrap();
}

use nano_vectordb_rs::{constants, dot_product, normalize, Data, NanoVectorDB};
use std::collections::HashMap;
use tempfile::NamedTempFile;

#[test]
fn test_basic_operations() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();

    let data = Data {
        id: "test".to_string(),
        vector: vec![0.1; 128],
        fields: HashMap::new(),
    };
    let (updates, inserts) = db.upsert(vec![data]).unwrap();
    db.save().unwrap();

    assert_eq!(inserts.len(), 1);
    assert_eq!(updates.len(), 0);

    let results = db.query(&vec![0.1; 128], 1, None, None);
    assert!(!results.is_empty());
    assert!(
        results[0]
            .get(constants::F_METRICS)
            .unwrap()
            .as_f64()
            .unwrap()
            > 0.99
    );
}

#[test]
fn test_persistence() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();
    db.upsert(vec![Data {
        id: "test".to_string(),
        vector: vec![0.1; 128],
        fields: HashMap::new(),
    }])
    .unwrap();
    db.save().unwrap();

    drop(db);
    let db2 = NanoVectorDB::new(128, path).unwrap();
    assert_eq!(db2.len(), 1);
}

#[test]
fn test_additional_data_handling() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();

    // Test storing and retrieving additional data
    let mut test_data = HashMap::new();
    test_data.insert("version".to_string(), serde_json::json!("1.0.0"));
    test_data.insert("config".to_string(), serde_json::json!({"max_size": 1000}));

    db.store_additional_data(test_data);
    let additional_data = db.get_additional_data();

    assert_eq!(additional_data.get("version").unwrap(), "1.0.0");
    assert_eq!(additional_data["config"]["max_size"], 1000);

    // Test persistence of additional data
    db.save().unwrap();
    let db2 = NanoVectorDB::new(128, path).unwrap();
    let loaded_data = db2.get_additional_data();

    assert_eq!(loaded_data.get("version").unwrap(), "1.0.0");
    assert!(!loaded_data.is_empty());
}

#[test]
fn test_get_method() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();

    let data1 = Data {
        id: "test1".to_string(),
        vector: vec![0.1; 128],
        fields: [("color".to_string(), "red".into())].into(),
    };

    let data2 = Data {
        id: "test2".to_string(),
        vector: vec![0.2; 128],
        fields: [("color".to_string(), "blue".into())].into(),
    };

    db.upsert(vec![data1, data2]).unwrap();

    // Test getting existing and non-existing IDs
    let results = db.get(&["test1".to_string(), "missing".to_string()]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "test1");
    assert_eq!(results[0].fields["color"], "red");
}

#[test]
fn test_delete_method() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    let mut db = NanoVectorDB::new(128, path).unwrap();

    let data1 = Data {
        id: "test1".to_string(),
        vector: vec![0.1; 128],
        fields: HashMap::new(),
    };

    let data2 = Data {
        id: "test2".to_string(),
        vector: vec![0.2; 128],
        fields: HashMap::new(),
    };

    db.upsert(vec![data1, data2]).unwrap();
    assert_eq!(db.len(), 2);

    // Delete one entry
    db.delete(&["test1".to_string()]);
    assert_eq!(db.len(), 1);

    // Verify matrix size was updated correctly
    assert_eq!(db.vector_bytes_len(), 128);

    // Verify remaining entry
    let results = db.query(&vec![0.2; 128], 1, None, None);
    assert!(!results.is_empty());
    assert_eq!(results[0][constants::F_ID], "test2");
}

#[test]
fn test_dot_product() {
    type Float = f32; // Ensure this matches your actual type

    // Test exact 4-element chunks
    let vec4 = vec![1.0, 2.0, 3.0, 4.0];
    let query4_chunks = &[[1.0, 1.0, 1.0, 1.0]];
    let query4_remainder = &[];
    assert_eq!(dot_product(&vec4, query4_chunks, query4_remainder), 10.0);

    // Test with remainder
    let vec5 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let query5_chunks = &[[1.0, 1.0, 1.0, 1.0]];
    let query5_remainder = &[1.0];
    assert_eq!(dot_product(&vec5, query5_chunks, query5_remainder), 15.0);

    // Test multiple chunks
    let vec8 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let query8_chunks = &[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]];
    let query8_remainder = &[];
    assert_eq!(dot_product(&vec8, query8_chunks, query8_remainder), 62.0);

    // Test empty vectors
    let vec_empty: &[Float] = &[];
    let query_empty_chunks: &[[Float; 4]] = &[];
    let query_empty_remainder: &[Float] = &[];
    assert_eq!(
        dot_product(vec_empty, query_empty_chunks, query_empty_remainder),
        0.0
    );

    // Test negative values
    let vec_neg = vec![2.0, -3.0];
    let query_neg_chunks: &[[Float; 4]] = &[];
    let query_neg_remainder = &[4.0, 5.0];
    assert_eq!(
        dot_product(&vec_neg, query_neg_chunks, query_neg_remainder),
        -7.0
    );

    // Test zero values
    let vec_zero = vec![0.0; 4];
    let query_zero_chunks = &[[0.0; 4]];
    let query_zero_remainder = &[];
    assert_eq!(
        dot_product(&vec_zero, query_zero_chunks, query_zero_remainder),
        0.0
    );

    // Test mismatched lengths (should panic)
    let vec_mismatch = vec![1.0, 2.0];
    let query_mismatch_chunks = &[[1.0, 1.0, 1.0, 1.0]];
    let query_mismatch_remainder = &[];
    let result = std::panic::catch_unwind(|| {
        dot_product(
            &vec_mismatch,
            query_mismatch_chunks,
            query_mismatch_remainder,
        )
    });
    assert!(result.is_err());
}

#[test]
fn test_normalization() {
    type Float = f32;
    let epsilon = 1e-5;

    // Basic normalization
    let v = vec![3.0, 4.0];
    let normalized = normalize(&v);
    let norm = normalized
        .iter()
        .fold(0.0 as Float, |acc, &x| x.mul_add(x, acc))
        .sqrt();
    assert!((norm - 1.0).abs() <= epsilon, "Norm: {}", norm);

    // High-dimensional vector
    let v = vec![1.0; 128];
    let normalized = normalize(&v);
    let expected = 1.0 / (128.0 as Float).sqrt();
    assert!(
        (normalized[0] - expected).abs() <= epsilon,
        "Expected: {}, Actual: {}",
        expected,
        normalized[0]
    );

    // Precision test
    let v = vec![1.0, 2.0, 3.0];
    let normalized = normalize(&v);
    let norm = normalized
        .iter()
        .fold(0.0 as Float, |acc, &x| x.mul_add(x, acc))
        .sqrt();
    assert!((norm - 1.0).abs() <= epsilon, "Norm: {}", norm);
}

#[test]
#[should_panic(expected = "Cannot normalize zero-length vector")]
fn test_zero_vector_normalization() {
    let zero_vec = vec![0.0; 128];
    normalize(&zero_vec);
}

#[test]
fn test_empty_state_checks() {
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    // Fresh database should be empty
    let mut db = NanoVectorDB::new(128, path).unwrap();
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);

    // Add some data and verify not empty
    db.upsert(vec![Data {
        id: "test".to_string(),
        vector: vec![0.1; 128],
        fields: HashMap::new(),
    }])
    .unwrap();
    assert!(!db.is_empty());
    assert_eq!(db.len(), 1);

    // Delete all data and verify empty again
    db.delete(&["test".to_string()]);
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);

    // Verify persistence of empty state
    db.save().unwrap();
    let db2 = NanoVectorDB::new(128, path).unwrap();
    assert!(db2.is_empty());
}

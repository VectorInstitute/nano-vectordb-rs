use nano_vectordb_rs::NanoVectorDB;
use tempfile::NamedTempFile;

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

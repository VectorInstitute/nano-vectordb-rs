use nano_vectordb_rs::{NanoVectorDB, Data, constants};
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
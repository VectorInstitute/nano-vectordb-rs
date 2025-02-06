use nano_vectordb_rs::{constants, Data, NanoVectorDB};
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

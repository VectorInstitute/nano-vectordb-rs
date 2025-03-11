## Implementation details


### Core data structures

#### `Data` Struct

```rust
pub struct Data {
    pub id: String,          // Unique identifier
    pub vector: Vec<Float>,  // Normalized vector data
    pub fields: HashMap<String, serde_json::Value> // Metadata
}
```

* Stores both the vector and arbitrary metadata
* Uses f32 for vector elements (type alias Float)
* Vectors are normalized during storage

#### `DataBase` Struct (Internal)

```rust
struct DataBase {
    embedding_dim: usize,     // Vector dimensionality
    data: Vec<Data>,          // All entries
    matrix: Vec<Float>,       // Flattened vectors for SIMD
    additional_data: HashMap<String, serde_json::Value> // DB metadata
}
```

Key optimization: Stores vectors in a flattened matrix for:

* Memory efficiency
* SIMD-friendly data layout
* Batch operations

#### `NanoVectorDB` Main Class

```rust
pub struct NanoVectorDB {
    pub embedding_dim: usize,  // Vector dimensionality
    pub metric: String,        // Distance metric ("cosine")
    storage_file: PathBuf,     // Persistence location
    storage: DataBase,         // Core data storage
}
```

##### Key methods

1. Initialization

```rust
pub fn new(embedding_dim: usize, storage_file: &str) -> Result<Self>
```

* Creates new DB instance or loads from file
* Validates matrix consistency
* Enforces dimensionality constraints

2. Upsert mechanism

```rust
pub fn upsert(&mut self, mut datas: Vec<Data>) -> Result<(Vec<String>, Vec<String>)>
```

* Update or Insert semantics:
    * Updates existing vectors by ID
    * Appends new vectors
* Normalizes all vectors before storage
* Returns tuple of (updated_ids, inserted_ids)

3. Vector Search (query)

```rust
pub fn query(&self, query: &[Float], top_k: usize, ...) -> Vec<HashMap...>
```

* Query normalization
* Parallel similarity calculation using Rayon
* Threshold filtering (better_than)
* Custom filtering support via `DataFilter`
* Top-k results using max-heap
* Result formatting with metadata

4. Persistence

```rust
pub fn save(&self) -> Result<()>
```

5. Helper Functions

***Normalization***

```rust
pub fn normalize(vector: &[Float]) -> Vec<Float>
```

* Ensures unit vectors for cosine similarity

***Dot Product***

```rust
pub fn dot_product(...) -> Float
```

* Optimized with 4-element chunks
* SIMD-friendly memory layout
* Handles remainder elements

### Usage

#### Basic usage example

A [simple example](https://github.com/VectorInstitute/nano-vectordb-rs/blob/main/examples/basic_usage.rs) 
demonstrating the basic usage of the library. The example demonstrates:

1. Creating a 3-dimensional vector database
2. Upserting sample data
3. Querying for similar vectors
4. Deleting a vector

#### Advanced usage example

A more [advanced example](https://github.com/VectorInstitute/nano-vectordb-rs/blob/main/examples/advanced_usage.rs), 
using a real-world dataset (Wikipedia embeddings) and querying for similar entries. 
The example uses the `hf-hub` crate to fetch the dataset and the `parquet` crate to read 
parquet files. The example demonstrates:

* Loading a dataset from Hugging Face Hub
* Reading Parquet files
* Upserting embedding data into the database
* Querying for similar entries
* Displaying results

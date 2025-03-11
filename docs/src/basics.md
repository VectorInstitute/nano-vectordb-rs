## Basics of vector databases

### What is a vector database?

Think of it as a specialized library to store vectors:

* Books = Vectors (arrays of numbers representing images, text, etc.)
* Card Catalog = Metadata (additional info like timestamps or categories)
* Librarian = Query engine (finds similar vectors quickly)

### What are the core components of a vector database?

#### Vector Storage

Like a chef turning ingredients into recipes, embedding models convert raw data (text, images) 
into numerical vectors - think of it as creating a unique "flavor profile" for each ingredient. 
Just as a master chef can describe tomatoes as "sweet, acidic, umami," the translator 
turns "dog" into `[0.7, -0.3, 0.5...]` capturing its essence in machine language.

To store these vectors, we need a data structure. Hence a vector storage component is essential.

#### Similarity Search and Optimization

When retrieving ingredients based on the their flavour profiles, if we are looking for
specific ingredients that are similar to what the chef is looking for, we have to perform
similarity search. The most naive approach is to compare a query ingredient (vector) with all
the ingredients in the database, which is essentially brute-force search.

When there are only a few ingredients to search over, brute-force is great. However, for
applications where we are searching over millions or even billions of vectors, brute-force
becomes too slow. We need special algorithms such as Approximate Nearest Neighbor (ANN), 
which are a class of algorithms designed to efficiently find data points closest to a query in 
high-dimensional spaces, balancing speed and accuracy.


<div class="warning">

There are several ANN algorithms which are mostly graph based (such as HNSW),
but there are also hashing techniques, tree based structures, etc. Besides ANN algorithms,
production grade vector databases also use quantization technniques to reduce memory
footprint, allowing scalability.

</div>

#### Metadata Store

For every ingredient, we might want to store additional information that are tied to it such as
calorific value, or when the ingredient was bought (timestamp). These are exact details that
we might be interested to fetch when we have found similar ingredients.

#### Vector Index

Similarity search and optimization is usually a dual step process. Firstly, algorithms like 
the [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320) are used to 
map vectors into search-optimized structures when inserting them into the database. 
In the querying step, these maps or indexes are then used to retrieve similar vectors very quickly.


Besides these core components, there are several other techniques and tools used to build 
scalable and reliable vector databases such as sharding, replication, multi-tenancy, etc.








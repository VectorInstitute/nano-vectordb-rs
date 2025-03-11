## Design choices

Before explaining the implementation details, some design choices need to be explained.


### Flattened matrix storage

Vectors are stored in a contiguous ```Vec<Float>``` with row-major order, enabling:

* Better cache locality during similarity calculations
* Efficient SIMD-friendly memory access patterns
* Compact serialization/deserialization via base64 encoding


### Rayon-based parallelism

Query processing uses data parallelism across CPU cores for:

* Near-linear scaling with core count
* Automatic work stealing for load balancing

### Hybrid JSON/base64 format

The serialization strategy, i.e. the approach to convert the da8ta
into a storable or transmittable format
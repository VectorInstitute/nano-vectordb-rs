# Building a Tiny Vector Database in Rust: A Technical Walkthrough 🦀

Vector databases have existed for a while but recent progress in using Retrieval Augmented Generation (RAG)
systems at scale has brought this technology to the attention of several engineering teams looking to 
adopt AI.

Vector databases have existed for years, but recent advancements in scaling Retrieval Augmented Generation (RAG) 
systems have drawn significant attention from engineering teams seeking to implement AI solutions.
These implementations employ distinct design philosophies. Some prioritize low-latency operations, 
while others emphasize scalability and distributed architecture support.
This raises practical questions: How might researchers experiment with basic vector database 
implementations to test novel retrieval enhancements or semantic search improvements? What about
 developers seeking to understand the underlying mechanisms? Highly optimized production-grade systems 
 present a challenge here – they often incorporate multiple abstraction layers that obscure implementation 
 details from users.

``nano-vectordb-rs`` provides a simple, minimalistic implementation using Rust - a systems programming 
language that ensures [memory safety](https://github.blog/developer-skills/programming-languages-and-frameworks/why-rust-is-the-most-admired-language-among-developers/) through its ownership and borrowing system. 
This approach empowers developers with precise control over memory allocation/deallocation without garbage 
collection overhead, while zero-cost abstractions guarantee high-level code compiles to optimized machine 
instructions with no runtime penalty. These features make Rust particularly well-suited for building vector 
databases, where performance is critical. Additionally, Rust's built-in concurrency safety enables safe 
parallel execution, which we'll leverage to accelerate our algorithms through straightforward parallelism.

This implementation strips away the complexity to reveal core vector database mechanics 
through ~350 lines of focused code. Designed to educate, we'll explore:

* Basics of vector databases
* Design choices
* Implementation details

# Building a Tiny Vector Database in Rust: A Technical Walkthrough 🦀

Vector databases have existed for a while but recent progress in using Retrieval Augmented Generation (RAG)
systems at scale has brought this technology to the attention of several engineering teams looking to 
adopt AI.

Vector database implementations are built with different design choices in mind. For example, some of them
prioritize latency, while others pay more attention to scale and supporting distributed architectures. 
But what if you are a researcher who wants to hack a naive vector database implementation and try new ideas
to improve retrieval or semantic search? Or just someone who wants to learn the inner workings of vector
databases? If you go looking at highly engineered implementations, you can see that they introduce several 
abstraction layers, most of which are opaque to the user.

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

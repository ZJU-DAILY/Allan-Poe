# All-in-one Graph-based Indexing for Hybrid Search on GPUs

**Allan-Poe** is a unified, GPU-accelerated graph-based index designed for efficient hybrid search. It overcomes the limitations of "separate-then-fuse" retrieval paradigms by integrating dense vectors, sparse vectors, and full-text search into a single, cohesive structure.

## Key Features

* **Unified Semantic Metric Space (USMS):** Fuses heterogeneous retrieval paths (dense, sparse, keyword) into a single high-dimensional representation without requiring index reconstruction for different weights.
* **GPU-Accelerated Construction:** Utilizes GPUs for massive parallelism in hybrid distance calculations and graph optimization, achieving significant speedups over CPU-based methods.
* **Dynamic Fusion Framework:** Supports arbitrary weights for different retrieval paths at query time, allowing flexible adjustment between semantic, lexical, and logical matching.
* **Advanced Pruning Strategies:**
    * **RNG-IP Joint Pruning:** Balances graph connectivity and search efficiency by combining Relative Neighborhood Graph (RNG) principles with Inner Product (IP) pruning.
    * **Keyword-aware Neighbor Recycling:** Preserves keyword search functionality by selectively retaining edges that would otherwise be pruned.
* **Logical Edge Augmentation:** Integrates Knowledge Graph (KG) relations to handle complex multi-hop queries by connecting logically related entities.

## Project Structure

The project is organized as a hybrid C++/CUDA and Python project using Pybind11, which is easy to use by using Python APIs:

```text
.
├── CMakeLists.txt                 # CMake build configuration
├── include/
│   ├── hybrid_search.cuh          # Core API declarations
│   └── utils.cuh                  # Shared CUDA helper functions
├── src/
│   ├── bindings.cpp               # Pybind11 wrappers for Python API
│   ├── index.cu                   # Index construction
│   └── query.cu                   # Query processing
├── example_data/
│   ├── 2wiki_bm25_corpus.bin      # Statistical sparse vectors of a small 2WikiMultiHopQA corpus in CSR format (encoded by BM25)
│   ├── 2wiki_bm25_queries.bin     # Statistical sparse vectors of 2WikiMultiHopQA queries in CSR format (encoded by BM25)
│   ├── 2wiki_dense_corpus.fvecs   # Dense vectors of a small 2WikiMultiHopQA corpus (encoded by BGE-M3)
│   ├── 2wiki_dense_queries.fvecs  # Dense vectors of 2WikiMultiHopQA queries (encoded by BGE-M3)
│   ├── 2wiki_sparse_corpus.bin    # Learned sparse vectors of a small 2WikiMultiHopQA corpus in CSR format (encoded by SPLADE)
│   ├── 2wiki_sparse_queries.bin   # Learned sparse vectors of a small 2WikiMultiHopQA queries in CSR format (encoded by SPLADE)
│   ├── 2wiki_kg.bin               # Knowledge graph of a small 2WikiMultiHopQA corpus in CSR format
│   ├── 2wiki_entity2doc.bin       # Mapping each entity to corpus (documents) in CSR format
│   ├── 2wiki_query_kg.bin         # IDs of the queried entities in CSR format
│   └── 2wiki_qrels.txt            # Groundtruth of the queries
└── example.py                     # Python script for building and searching on the small 2WikiMultiHopQA dataset
```

## Prerequisites

* **Hardware:** NVIDIA GPU
* **Software:**
    * CUDA Toolkit (tested on 12.2).
    * C++ Compiler (tested on 11.2).
    * CMake >= 3.10.
    * Python 3.x & Pybind11.

## Installation

Allan-Poe is built as a Python extension module using CMake.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ZJU-DAILY/Allan-Poe.git
    cd Allan-Poe
    ```

2.  **Compile the module:**
    ```bash
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release .. 
    make -j
    ```

## Usage

Allan-Poe exposes two primary APIs via Python: `build_index` and `search_index`.

### 1. Data Preparation
Ensure your data is in the correct binary formats:
* **Dense Vectors:** `.fvecs` format.
* **Statistical/Learned Sparse Vectors:** Binary files beginning with the number of non-zero elements (unsigned int) and following offsets, indices, and values (CSR format).
* **Knowledge Graph/Map of Entity to Documents/Queried Entities:** Binary files beginning with the number of entities and non-zero elements (unsigned int) and following offsets, indices of entities/documents.

### 2. Building the Index
The `build_index` function constructs the hybrid graph, performing NN-Descent, RNG-IP pruning, and keyword edge recycling.

```python
import hybrid_search

print("Building Allan-Poe Index...")
hybrid_search.build_index(
    dense_data_path="example_data/2wiki_dense_corpus.fvecs",
    sparse_data_path="example_data/2wiki_sparse_corpus.bin",
    bm25_data_path="example_data/2wiki_bm25_corpus.bin",
    output_graph_path="index.poe"
)
print("Index construction complete.")
```

### 3. Performing Hybrid Search
The `search_index` function executes the query using the constructed graph. You can dynamically adjust `dense_weight`, `sparse_weight`, and `bm25_weight` to tune the retrieval results without rebuilding the index.

```python
import hybrid_search

print("Searching Index...")
hybrid_search.search_index(
    dense_data_path="example_data/2wiki_dense_corpus.fvecs",
    dense_query_path="example_data/2wiki_dense_queries.fvecs",
    sparse_data_path="example_data/2wiki_sparse_corpus.bin",
    sparse_query_path="example_data/2wiki_sparse_queries.bin",
    bm25_data_path="example_data/2wiki_bm25_corpus.bin",
    bm25_query_path="example_data/2wiki_bm25_queries.bin",
    keyword_id_path="",
    knowledge_path="example_data/2wiki_kg.bin",
    entity2doc_path="example_data/2wiki_entity2doc.bin",
    query_entity_path="example_data/2wiki_query_kg.bin",
    graph_path="index.poe",
    ground_truth_path="example_data/2wiki_qrels.txt",
    top_k=10,
    cands=128,          # Size of candidates pool
    sparse_weight=1.0,  # Weight for learned sparse vectors
    bm25_weight=1.0,    # Weight for statistical sparse vectors
    dense_weight=1.0,  # Weight for dense vectors
    kg_weight=40.0      # Weight for the knowledge graph
)
```

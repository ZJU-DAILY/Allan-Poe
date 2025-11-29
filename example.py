import hybrid_search

# 1. Build Index
hybrid_search.build_index(
    dense_data_path="example_data/2wiki_dense_corpus.fvecs",
    sparse_data_path="example_data/2wiki_sparse_corpus.bin",
    bm25_data_path="example_data/2wiki_bm25_corpus.bin",
    output_graph_path="index.poe"
)

# 2. Search Index
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
    cands=128,
    sparse_weight=1.0,
    bm25_weight=1.0,
    dense_weight=1.0,
    kg_weight=40.0
)

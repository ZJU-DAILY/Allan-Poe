#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hybrid_search.cuh"

namespace py = pybind11;

PYBIND11_MODULE(hybrid_search, m) {
    m.doc() = "Hybrid Search with Allan-Poe";

    m.def("build_index", &build_index_impl, 
        "Build the hybrid index",
        py::arg("dense_data_path"),
        py::arg("sparse_data_path"),
        py::arg("bm25_data_path"),
        py::arg("output_graph_path")
    );

    m.def("search_index", &search_index_impl, 
        "Search the hybrid index",
        py::arg("dense_data_path"),
        py::arg("dense_query_path"),
        py::arg("sparse_data_path"),
        py::arg("sparse_query_path"),
        py::arg("bm25_data_path"),
        py::arg("bm25_query_path"),
        py::arg("keyword_id_path"),
        py::arg("knowledge_path"),
        py::arg("entity2doc_path"),
        py::arg("query_entity_path"),
        py::arg("graph_path"),
        py::arg("ground_truth_path"),
        py::arg("top_k"),
        py::arg("cands"),
        py::arg("sparse_weight"),
        py::arg("bm25_weight"),
        py::arg("dense_weight"),
        py::arg("kg_weight")
    );

}
from .search_utils import load_golden_dataset, load_doctors
from .hybrid_search import HybridSearch
from .semantic_search import SemanticSearch

def evaluation_command(limit: int):
    drs_docs = load_doctors()
    test_cases = load_golden_dataset() 

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(drs_docs)
    hybryd_search = HybridSearch(drs_docs)   

    for i, case in enumerate(test_cases):
        query = case["query"]
        relevant_docs_ids = [doc["id"] for doc in case["relevant_docs"]]
        relevant_docs_names = [doc["name"] for doc in case["relevant_docs"]]
        relevant_retrieved = 0
        results = hybryd_search.rrf_search(query, k=60, limit=limit)
        retrieved = []
        for result in results:
            if result[1]["doc"]["id"] in relevant_docs_ids:
                relevant_retrieved += 1
            retrieved.append(result[1]["doc"]["name"])
        precision = relevant_retrieved / len(results) if len(results) > 0 else 0
        test_cases[i]["precision"] = precision
        test_cases[i]["retrieved"] = retrieved
        test_cases[i]["relevant"] = relevant_docs_names

    return test_cases
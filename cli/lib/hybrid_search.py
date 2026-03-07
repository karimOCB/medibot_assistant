import os

from .keyword_search import InvertedIndex
from .semantic_search import SemanticSearch
from .search_utils import HYBRID_A, DEFAULT_SEARCH_LIMIT, load_doctors, RRF_K
from .query_enhancement import enhance_query

class HybridSearch:
    def __init__(self, drs_docs: list[dict]) -> None:
        self.drs_docs = drs_docs
        self.semantic_search = SemanticSearch()
        self.semantic_search.load_or_create_embeddings(drs_docs)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 50)
        semantic_results = self.semantic_search.search(query, limit * 50)
        bm25_normalized_scores = normalize_command([result["score"] for result in bm25_results])
        semantic_normalized_scores = normalize_command([result["score"] for result in semantic_results])
        results = {}
        for i, result in enumerate(bm25_results):
            results[result["doc"]["id"]] = {
                "doc": result["doc"],
                "bm25_normalized": bm25_normalized_scores[i],
                "semantic_normalized": semantic_normalized_scores[i],
                "hybrid_score": self.hybrid_score(bm25_normalized_scores[i], semantic_normalized_scores[i], alpha)
            }
        results_sorted = sorted(results.items(), key=lambda item: item[1]["hybrid_score"], reverse=True)
        return [result[1] for result in results_sorted[:limit]]

    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha=HYBRID_A) -> float:
            return alpha * bm25_score + (1 - alpha) * semantic_score    

    def rrf_search(self, query, k: int, limit=10) -> list[tuple]:
        bm25_results = self._bm25_search(query, limit * 50)
        semantic_results = self.semantic_search.search(query, limit * 50)
        semantic_map = {ss["id"]: i for i, ss in enumerate(semantic_results, start=1)}
        results = {}
        for i, result in enumerate(bm25_results, start=1):
            doc_id = result["doc"]["id"] 
            results[doc_id] = {
                "doc": result["doc"],
                "bm25_rank": i,
                "semantic_rank": semantic_map[doc_id],
                "rrf_score": (1 / (k + i)) + (1 / (k + semantic_map[doc_id]))
            }
        return sorted(results.items(), key=lambda item: item[1]["rrf_score"], reverse=True)[:limit]


def weighted_search_command(query: str, alpha: float, limit: int) -> list[dict]:
    drs_docs = load_doctors()
    hybrid_search = HybridSearch(drs_docs)
    return hybrid_search.weighted_search(query, alpha, limit)


def rrf_search_command(query: str, limit: int, k: int = RRF_K, enhance: str = None) -> list[dict]:
    drs_docs = load_doctors()
    hybrid_search = HybridSearch(drs_docs)
    
    original_q = query
    enhanced_q = None
    if enhance:
        enhanced_q = enhance_query(query, method=enhance)
        query = enhanced_q

    results = [result[1] for result in hybrid_search.rrf_search(query, k, limit)]
    
    return {
        "original_query": original_q,
        "enhanced_query": enhanced_q,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "results": results,
    }


def normalize_command(scores: list[float]) -> list[float]:
    if not scores or scores == []:
        return
    min_s = min(scores)
    max_s = max(scores)
    
    if min_s == max_s:
        return [1.0] * len(scores)
    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_s) / (max_s - min_s))
    return normalized_scores
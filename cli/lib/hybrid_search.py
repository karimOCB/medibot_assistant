import os

from .keyword_search import InvertedIndex
from .semantic_search import SemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = SemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[str]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize_command(scores: list[float]) -> None:
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
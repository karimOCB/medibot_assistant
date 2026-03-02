import numpy as np
import os
from sentence_transformers import SentenceTransformer
from lib.search_utils import cache_path, load_doctors, DEFAULT_SEARCH_LIMIT

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.drs_docs = None
        self.drs_docs_map = {}
        self.embeddings_path = os.path.join(cache_path, "drs_embeddings.npy")
    
    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Give a text of at least one word.")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, drs_docs: list[dict]) -> np.ndarray:
        self.drs_docs = drs_docs
        drs_representations = []
        for dr in drs_docs:
            self.drs_docs_map[dr["id"]] = dr
            drs_representations.append(f"{dr["name"]}. {dr["age"]}. {dr["specialty"]}. {dr["bio"]}. {dr["availability"]}")
        self.embeddings = self.model.encode(drs_representations, show_progress_bar=True)
        
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, drs_docs: list[dict]) -> np.ndarray:
        self.drs_docs = drs_docs
        for dr in drs_docs:
            self.drs_docs_map[dr["id"]] = dr
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(drs_docs):
                return self.embeddings
        
        return self.build_embeddings(drs_docs)

    def search(self, query: str, limit:int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        q_embedding = self.generate_embedding(query)
        scores = []
        for i, dr_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(q_embedding, dr_embedding)
            document = self.drs_docs[i]
            scores.append((similarity_score, document))
        sorted_scores = sorted(scores, key= lambda item: item[0], reverse=True)[:limit]
        return [{"score": ss[0], "name": ss[1]['name'], "id": ss[1]['id'], "dr_info": f"Age: {ss[1]['age']}. Specialty: {ss[1]['specialty']}. Bio: {ss[1]['bio']} Availability: {ss[1]['availability']}"} for ss in sorted_scores]
        

def search_command(query: str, limit: int) -> list[dict]:
    semantic_search = SemanticSearch()
    drs_docs = load_doctors()
    semantic_search.load_or_create_embeddings(drs_docs)
    return semantic_search.search(query, limit)


def embed_text(text: str) -> np.ndarray:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    return embedding


def verify_model() -> None:
    semantic_search = SemanticSearch()
    model = semantic_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    drs_docs = load_doctors()
    embeddings = semantic_search.load_or_create_embeddings(drs_docs)
    print(f"Number of docs:   {len(drs_docs)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str) -> np.ndarray:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
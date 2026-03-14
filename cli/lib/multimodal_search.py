import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import cosine_similarity, load_doctors, DEFAULT_SEARCH_LIMIT

class MultimodalSearch():
    def __init__(self, drs_docs: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.drs_docs = drs_docs
        self.text = [f"{dr["name"]}. {dr["age"]}. {dr["specialty"]}. {dr["bio"]}. {dr["availability"]}" for dr in drs_docs]
        self.text_embeddings = self.model.encode(self.text, show_progress_bar=True)

    def search_with_image(self, img_path: str) -> list[dict]:
        img_embedding = self.embed_image(img_path)
        results = self.drs_docs
        for i, embedding in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(img_embedding, embedding)
            results[i]["similarity_score"] = similarity_score
        return sorted(results, key=lambda result: result["similarity_score"], reverse=True)[:DEFAULT_SEARCH_LIMIT]

    def embed_image(self, img_path: str) -> np.ndarray: 
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path)
        embedding = self.model.encode([img])
        return embedding[0]


def verify_image_embedding(img_path: str) -> np.ndarray:
    multimodal = MultimodalSearch()
    embedding = multimodal.embed_image(img_path)
    return embedding


def image_search_commad(img_path: str) -> list[dict]:
    drs_docs = load_doctors()
    multimodal = MultimodalSearch(drs_docs)
    return multimodal.search_with_image(img_path)
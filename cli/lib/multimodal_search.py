import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
    
    def embed_image(self, img_path: str) -> np.ndarray: 
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path)
        embedding = self.model.encode([img])
        return embedding[0]

def verify_image_embedding(img_path: str):
    multimodal = MultimodalSearch()
    embedding = multimodal.embed_image(img_path)
    return embedding
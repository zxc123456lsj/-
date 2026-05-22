import numpy as np
from typing import Optional

from app.config import settings


class EmbeddingService:
    _bge_model = None
    _clip_model = None
    _clip_processor = None

    @classmethod
    def _load_bge(cls):
        if cls._bge_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._bge_model = SentenceTransformer(settings.bge_model_name)
            except ImportError:
                raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

    @classmethod
    def _load_clip(cls):
        if cls._clip_model is None:
            try:
                import torch
                import clip
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._clip_model, cls._clip_processor = clip.load(settings.clip_model_name, device=device)
            except ImportError:
                raise RuntimeError("clip not installed. Run: pip install openai-clip")

    @classmethod
    def embed_text_bge(cls, text: str) -> list[float]:
        cls._load_bge()
        embedding = cls._bge_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    @classmethod
    def embed_texts_bge(cls, texts: list[str]) -> list[list[float]]:
        cls._load_bge()
        embeddings = cls._bge_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @classmethod
    def embed_text_clip(cls, text: str) -> list[float]:
        cls._load_clip()
        import torch
        device = next(cls._clip_model.parameters()).device
        tokens = cls._clip_processor.tokenizer([text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = cls._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()

    @classmethod
    def embed_image_clip(cls, image_path: str) -> list[float]:
        cls._load_clip()
        from PIL import Image
        import torch
        device = next(cls._clip_model.parameters()).device
        image = cls._clip_processor(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = cls._clip_model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()

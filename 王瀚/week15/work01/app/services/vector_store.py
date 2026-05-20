import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VectorStore:
    _instance = None
    _text_collection = None
    _image_collection = None

    @classmethod
    def _get_in_memory_store(cls):
        if cls._instance is None:
            cls._instance = InMemoryVectorStore()
            logger.info("Using in-memory vector store (no Milvus connection)")
        return cls._instance

    @classmethod
    def get_text_collection(cls):
        try:
            from pymilvus import connections, Collection, utility
            connections.connect(alias="default", host="localhost", port="19530")
            if utility.has_collection("text_chunks"):
                c = Collection(name="text_chunks")
                c.load()
                return c
        except Exception as e:
            logger.warning("Milvus not available: %s. Using in-memory store.", e)
        return cls._get_in_memory_store()

    @classmethod
    def get_image_collection(cls):
        try:
            from pymilvus import connections, Collection, utility
            connections.connect(alias="default", host="localhost", port="19530")
            if utility.has_collection("image_embeddings"):
                c = Collection(name="image_embeddings")
                c.load()
                return c
        except Exception as e:
            logger.warning("Milvus not available: %s. Using in-memory store.", e)
        return cls._get_in_memory_store()

    @classmethod
    def insert_text_embeddings(cls, entities: list[dict]):
        store = cls._get_in_memory_store()
        store.insert_text(entities)

    @classmethod
    def insert_image_embeddings(cls, entities: list[dict]):
        store = cls._get_in_memory_store()
        store.insert_image(entities)

    @classmethod
    def search_text(cls, query_embedding: list[float], kb_id: str, top_k: int = 5):
        store = cls._get_in_memory_store()
        return store.search_text(query_embedding, kb_id, top_k)

    @classmethod
    def search_image(cls, query_embedding: list[float], kb_id: str, top_k: int = 3):
        store = cls._get_in_memory_store()
        return store.search_image(query_embedding, kb_id, top_k)

    @classmethod
    def delete_by_doc(cls, doc_id: str):
        store = cls._get_in_memory_store()
        store.delete_by_doc(doc_id)


class InMemoryVectorStore:
    """Fallback in-memory vector store when Milvus is unavailable."""

    def __init__(self):
        self._text_data: list[dict] = []
        self._text_vectors: list[list[float]] = []
        self._image_data: list[dict] = []
        self._image_vectors: list[list[float]] = []

    def insert_text(self, entities: list[dict]):
        for ent in entities:
            vec = ent.pop("embedding", [0.0])
            self._text_data.append(ent)
            self._text_vectors.append(vec)

    def insert_image(self, entities: list[dict]):
        for ent in entities:
            vec = ent.pop("embedding", [0.0])
            self._image_data.append(ent)
            self._image_vectors.append(vec)

    def search_text(self, query_embedding: list[float], kb_id: str, top_k: int = 5):
        scores = self._cosine_similarities(query_embedding, self._text_vectors)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if self._text_data[idx].get("kb_id") == kb_id or not kb_id:
                results.append({
                    **self._text_data[idx],
                    "score": float(scores[idx]),
                })
        return results

    def search_image(self, query_embedding: list[float], kb_id: str, top_k: int = 3):
        scores = self._cosine_similarities(query_embedding, self._image_vectors)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if self._image_data[idx].get("kb_id") == kb_id or not kb_id:
                results.append({
                    **self._image_data[idx],
                    "score": float(scores[idx]),
                })
        return results

    def delete_by_doc(self, doc_id: str):
        self._text_data = [d for d in self._text_data if d.get("doc_id") != doc_id]
        self._image_data = [d for d in self._image_data if d.get("doc_id") != doc_id]

    @staticmethod
    def _cosine_similarities(q: list[float], vectors: list[list[float]]):
        import math
        q_norm = math.sqrt(sum(x * x for x in q)) or 1.0
        results = []
        for v in vectors:
            dot = sum(a * b for a, b in zip(q, v))
            v_norm = math.sqrt(sum(x * x for x in v)) or 1.0
            results.append(dot / (q_norm * v_norm))
        return results

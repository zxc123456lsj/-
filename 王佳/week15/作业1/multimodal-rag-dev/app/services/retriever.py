import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.services.milvus_utils import connect_milvus, disconnect_milvus, search_image, search_text

load_dotenv()

BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-small-zh-v1.5")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "jinaai/jina-clip-v2")

bge_model: SentenceTransformer | None = None
clip_model: SentenceTransformer | None = None


def _load_models() -> tuple[SentenceTransformer, SentenceTransformer]:
    global bge_model, clip_model
    if bge_model is None:
        bge_model = SentenceTransformer(BGE_MODEL_PATH)
    if clip_model is None:
        clip_model = SentenceTransformer(CLIP_MODEL_PATH, trust_remote_code=True)
    return bge_model, clip_model


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    bge, clip = _load_models()

    query_text_vector = bge.encode(query, normalize_embeddings=True).tolist()
    query_clip_vector = clip.encode(query, normalize_embeddings=True).tolist()

    connect_milvus()
    try:
        text_results = search_text(query_text_vector, top_k=top_k)
        image_results = search_image(query_clip_vector, top_k=3)

        merged: dict[str, dict] = {}
        for r in text_results:
            key = f"{r['file_name']}_{r['chunk_index']}"
            merged[key] = {**r, "type": "text", "score": r["score"]}

        for r in image_results:
            key = f"{r['file_name']}_img_{r['chunk_index']}"
            if key not in merged or r["score"] > merged[key]["score"]:
                merged[key] = {**r, "type": "image", "score": r["score"]}

        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]
    finally:
        disconnect_milvus()

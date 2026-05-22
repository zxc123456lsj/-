import os

from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_data")

TEXT_DIM = 512  # BGE-small-zh-v1.5
CLIP_DIM = 1024  # Jina-CLIP-v2


def connect_milvus(alias: str = "default") -> None:
    connections.connect(alias=alias, uri=MILVUS_URI)


def disconnect_milvus(alias: str = "default") -> None:
    connections.disconnect(alias)


def create_collection() -> Collection:
    if utility.has_collection(MILVUS_COLLECTION):
        return Collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=TEXT_DIM),
        FieldSchema(name="clip_text_vector", dtype=DataType.FLOAT_VECTOR, dim=CLIP_DIM),
        FieldSchema(name="clip_image_vector", dtype=DataType.FLOAT_VECTOR, dim=CLIP_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="db_id", dtype=DataType.INT64),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="多模态RAG数据集合")
    collection = Collection(MILVUS_COLLECTION, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="text_vector", index_params=index_params)
    collection.create_index(field_name="clip_text_vector", index_params=index_params)
    collection.create_index(field_name="clip_image_vector", index_params=index_params)
    collection.load()
    return collection


def get_collection() -> Collection:
    if not utility.has_collection(MILVUS_COLLECTION):
        return create_collection()
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    return collection


def insert_data(data: list[dict]) -> None:
    collection = get_collection()
    entities = [
        [item.get("text_vector", []) for item in data],
        [item.get("clip_text_vector", []) for item in data],
        [item.get("clip_image_vector", []) for item in data],
        [item.get("text", "") for item in data],
        [item.get("db_id", 0) for item in data],
        [item.get("file_name", "") for item in data],
        [item.get("file_path", "") for item in data],
        [item.get("chunk_index", 0) for item in data],
    ]
    collection.insert(entities)
    collection.flush()


def search_text(query_vector: list[float], top_k: int = 5) -> list[dict]:
    collection = get_collection()
    results = collection.search(
        data=[query_vector],
        anns_field="text_vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["text", "file_name", "file_path", "chunk_index", "db_id"],
    )
    return [
        {"text": hit.entity.get("text", ""), "file_name": hit.entity.get("file_name", ""), "file_path": hit.entity.get("file_path", ""), "chunk_index": hit.entity.get("chunk_index", 0), "db_id": hit.entity.get("db_id", 0), "score": hit.score}
        for hit in results[0]
    ]


def search_image(query_vector: list[float], top_k: int = 3) -> list[dict]:
    collection = get_collection()
    results = collection.search(
        data=[query_vector],
        anns_field="clip_image_vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["text", "file_name", "file_path", "chunk_index", "db_id"],
    )
    return [
        {"text": hit.entity.get("text", ""), "file_name": hit.entity.get("file_name", ""), "file_path": hit.entity.get("file_path", ""), "chunk_index": hit.entity.get("chunk_index", 0), "db_id": hit.entity.get("db_id", 0), "score": hit.score}
        for hit in results[0]
    ]

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MultiModal RAG"
    debug: bool = True

    storage_root: str = str(Path(__file__).parent.parent / "storage")
    pdf_dir: str = ""
    image_dir: str = ""
    parsed_dir: str = ""

    database_url: str = "sqlite:///./storage/mrag.db"

    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_text_collection: str = "text_chunks"
    milvus_image_collection: str = "image_embeddings"

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_parse_topic: str = "document_parse"
    kafka_consumer_group: str = "parse_worker_group"

    bge_model_name: str = "BAAI/bge-large-zh-v1.5"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    qwen_vl_model: str = "qwen-vl-plus"

    embed_dim_bge: int = 1024
    embed_dim_clip: int = 512

    retrieve_top_k_text: int = 5
    retrieve_top_k_image: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pdf_dir = os.path.join(self.storage_root, "pdfs")
        self.image_dir = os.path.join(self.storage_root, "images")
        self.parsed_dir = os.path.join(self.storage_root, "parsed")
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.parsed_dir, exist_ok=True)


settings = Settings()

"""
Milvus 向量数据库客户端
"""
from pymilvus import MilvusClient, DataType
import config

# 全局客户端实例
_client = None


def get_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _client
    if _client is None:
        _client = MilvusClient(
            uri=config.MILVUS_URI,
            token=config.MILVUS_TOKEN
        )
    return _client


def create_collection_if_not_exists():
    """
    创建 Collection（如果不存在）
    定义 Schema 包括：
    - text_vector: BGE 编码的文本向量 (512维)
    - clip_text_vector: CLIP 编码的文本向量 (1024维)
    - clip_image_vector: CLIP 编码的图像向量 (1024维)
    """
    client = get_client()

    if client.has_collection(config.MILVUS_COLLECTION):
        print(f"Collection {config.MILVUS_COLLECTION} already exists")
        return

    # 定义 Schema
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="clip_text_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="clip_image_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="db_id", datatype=DataType.INT64)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=255)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1000)

    # 创建 Collection
    client.create_collection(
        collection_name=config.MILVUS_COLLECTION,
        schema=schema
    )

    # 创建索引
    index_params = MilvusClient.prepare_index_params()

    # 为 text_vector 创建索引
    index_params.add_index(
        field_name="text_vector",
        index_type="IVF_FLAT",
        metric_type="IP",  # 内积，用于归一化向量
        params={"nlist": 128}
    )

    # 为 clip_text_vector 创建索引
    index_params.add_index(
        field_name="clip_text_vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    # 为 clip_image_vector 创建索引
    index_params.add_index(
        field_name="clip_image_vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    client.create_index(
        collection_name=config.MILVUS_COLLECTION,
        index_params=index_params
    )

    print(f"Collection {config.MILVUS_COLLECTION} created successfully")


def insert_vectors(data: list) -> dict:
    """
    插入向量数据

    Args:
        data: 数据列表，每个元素为字典

    Returns:
        插入结果
    """
    client = get_client()
    return client.insert(
        collection_name=config.MILVUS_COLLECTION,
        data=data
    )


def search_vectors(query_vector: list, top_k: int = 5, filter: str = None) -> list:
    """
    搜索相似向量

    Args:
        query_vector: 查询向量 (BGE 512维)
        top_k: 返回结果数量
        filter: 过滤条件

    Returns:
        搜索结果列表
    """
    client = get_client()

    search_params = {
        "collection_name": config.MILVUS_COLLECTION,
        "data": [query_vector],
        "limit": top_k,
        "anns_field": "text_vector",
        "output_fields": ["text", "db_id", "file_name", "file_path"],
        "search_params": {"metric_type": "IP", "params": {"nprobe": 10}}
    }

    if filter:
        search_params["filter"] = filter

    results = client.search(**search_params)

    # 格式化结果
    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append({
                "id": hit.get("id"),
                "score": hit.get("distance"),
                "text": hit.get("entity", {}).get("text"),
                "db_id": hit.get("entity", {}).get("db_id"),
                "file_name": hit.get("entity", {}).get("file_name"),
                "file_path": hit.get("entity", {}).get("file_path"),
            })

    return formatted_results


def delete_by_file_id(file_id: int):
    """
    根据文件ID删除所有相关向量

    Args:
        file_id: 文件ID
    """
    client = get_client()
    client.delete(
        collection_name=config.MILVUS_COLLECTION,
        filter=f"db_id == {file_id}"
    )


# 初始化时创建 Collection
create_collection_if_not_exists()

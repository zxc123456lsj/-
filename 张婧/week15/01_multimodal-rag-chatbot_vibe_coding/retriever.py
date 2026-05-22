import numpy as np

def hybrid_search(milvus_client, query_vector, top_k=5, alpha=0.6):
    """
    同时检索 text_vector 和 clip_image_vector，融合分数
    alpha: 文本检索权重, (1-alpha) 图像检索权重
    """
    # 文本向量检索
    text_res = milvus_client.search(
        collection_name="rag_data_new",
        data=[query_vector],
        limit=top_k,
        anns_field="text_vector",
        output_fields=["text", "file_name", "file_path", "clip_image_vector"]
    )[0]

    # 图像向量检索（用同一个 query_vector 搜索 clip_image_vector 字段）
    img_res = milvus_client.search(
        collection_name="rag_data_new",
        data=[query_vector],
        limit=top_k,
        anns_field="clip_image_vector",
        output_fields=["text", "file_name", "file_path"]
    )[0]

    # 合并去重（按 id）
    merged = {}
    for res in text_res:
        entity = res["entity"]
        entity_id = entity.get("id", hash(entity["text"]))
        merged[entity_id] = {
            "text": entity["text"],
            "file_name": entity["file_name"],
            "file_path": entity["file_path"],
            "score": res["distance"] * alpha
        }
    for res in img_res:
        entity = res["entity"]
        entity_id = entity.get("id", hash(entity["text"]))
        if entity_id in merged:
            merged[entity_id]["score"] += res["distance"] * (1 - alpha)
        else:
            merged[entity_id] = {
                "text": entity["text"],
                "file_name": entity["file_name"],
                "file_path": entity["file_path"],
                "score": res["distance"] * (1 - alpha)
            }

    sorted_chunks = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return sorted_chunks
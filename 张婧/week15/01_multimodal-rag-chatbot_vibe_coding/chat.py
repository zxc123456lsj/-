import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from app.services.retriever import hybrid_search
from app.services.qwen_client import qwen_vl_chat

router = APIRouter()

# 加载 BGE 模型（用于问题向量化）
bge_model = SentenceTransformer('/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5')

milvus_client = MilvusClient(
    uri="https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    token="9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b"
)

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@router.post("/chat")
async def chat(req: ChatRequest):
    # 1. 问题向量化
    q_emb = bge_model.encode(req.question, normalize_embeddings=True).tolist()

    # 2. 混合检索（文本向量 + 图像向量）
    chunks = hybrid_search(milvus_client, q_emb, top_k=req.top_k)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found")

    # 3. 构造多模态上下文（文本 + 图片URL）
    context_text = []
    image_urls = []
    for chunk in chunks:
        context_text.append(chunk["text"])
        # 提取 markdown 图片链接并转换为可访问URL
        import re
        matches = re.findall(r'!\[.*?\]\((.*?)\)', chunk["text"])
        for local_path in matches:
            # local_path 形如 images/xxx.png，转换为 /processed/{file_dir}/images/xxx.png
            file_dir = os.path.basename(chunk["file_path"]).split(".")[0]
            rel_path = local_path.replace("images/", f"./processed/{file_dir}/images/")
            abs_url = f"http://localhost:8000{rel_path[1:]}"  # 去掉开头的 '.'
            image_urls.append(abs_url)

    # 4. 调用 Qwen-VL 多模态模型
    answer = qwen_vl_chat(req.question, context_text, image_urls)

    return {
        "answer": answer,
        "references": [
            {"file_name": c["file_name"], "text_snippet": c["text"][:200]}
            for c in chunks
        ]
    }
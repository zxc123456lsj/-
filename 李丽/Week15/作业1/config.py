"""
多模态 RAG 聊天机器人 - 配置文件
"""
import os

# =============================================================================
# 大模型配置 (DashScope / OpenAI 兼容)
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# =============================================================================
# Milvus 向量数据库配置
# =============================================================================
MILVUS_URI = os.getenv("MILVUS_URI", "https://your-milvus-server.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "your-token")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_data_new")

# =============================================================================
# Kafka 配置
# =============================================================================
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_DOCUMENT = os.getenv("KAFKA_TOPIC_DOCUMENT", "rag-data")

# =============================================================================
# 模型路径配置
# =============================================================================
# 使用 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-small-zh-v1.5")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "jinaai/jina-clip-v2")
CLIP_DIM = 1024  # jina-clip-v2 输出维度

# =============================================================================
# 存储路径配置
# =============================================================================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./processed")
DB_PATH = os.getenv("DB_PATH", "./db.db")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =============================================================================
# MinerU 配置
# =============================================================================
MINERU_ENDPOINT = os.getenv("MINERU_ENDPOINT", "http://127.0.0.1:30000")

# =============================================================================
# RAG 配置
# =============================================================================
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # 检索 top-k 个 chunk
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "256"))  # 文本分块大小

# RAG Prompt 模板
RAG_PROMPT_TEMPLATE = """基于以下资料回答用户问题。

用户问题：{question}

相关资料：
{context}

回答要求：
- 回答要客观、有逻辑，必须基于提供的资料
- 如果资料中包含图片链接，单独一行输出，保留原始链接，将图放在合适位置
- 如果资料无法回答问题，请明确说明
- 在回答末尾列出参考来源
"""

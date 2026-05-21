# 多模态RAG问答系统

支持PDF文档解析、多模态向量检索、智能问答的完整系统。

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| 后端框架 | FastAPI + Uvicorn |
| 消息队列 | Kafka + Zookeeper |
| 向量数据库 | Milvus |
| 元数据存储 | SQLite/SQLAlchemy |
| 文档解析 | MinerU (vlm-http-client模式) |
| 文本编码 | BGE-small-zh-v1.5 (512维) |
| 图像编码 | Jina-CLIP-v2 (1024维) |
| 推理模型 | Qwen-VL-Chat |
| 前端界面 | Streamlit |

## 快速开始

### 1. 环境准备

- Docker Desktop
- Python 3.10+
- GPU (可选，加速MinerU和Qwen-VL)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，修改配置项
```

### 4. 启动基础设施服务

```bash
docker-compose up -d zookeeper kafka etcd minio milvus
```

### 5. 启动MinerU（可选，需要GPU）

```bash
docker-compose up -d mineru
```

### 6. 启动后端API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 启动离线解析Worker

```bash
python -m app.worker.parser
```

### 8. 启动前端界面

```bash
streamlit run web_app.py
```

访问 http://localhost:8501 使用系统。

## 项目结构

```
multimodal-rag-dev/
├── app/
│   ├── api/            # REST API路由
│   │   ├── upload.py   # 文件上传接口
│   │   └── chat.py     # 问答接口
│   ├── models/         # 数据库模型
│   │   └── database.py
│   ├── services/       # 业务逻辑
│   │   ├── chat.py     # 问答服务
│   │   ├── retriever.py # 检索引擎
│   │   └── milvus_utils.py # Milvus操作
│   ├── worker/         # 后台Worker
│   │   └── parser.py   # 文档解析Worker
│   ├── utils/          # 工具模块
│   │   ├── config.py
│   │   ├── hash_utils.py
│   │   ├── kafka_utils.py
│   │   └── logger.py
│   └── main.py         # FastAPI入口
├── tests/              # 测试用例
│   ├── test_upload.py
│   ├── test_retrieval.py
│   └── test_chat.py
├── uploads/            # PDF上传目录
├── processed/          # 解析结果
├── static/             # 静态资源
├── docker-compose.yml  # Docker服务编排
├── requirements.txt    # Python依赖
├── .env.example        # 环境变量模板
└── README.md           # 使用文档
```

## API接口

### 文件上传

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### 查询文件状态

```bash
curl http://localhost:8000/api/files/1/status
```

### 智能问答

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "文档的主要内容是什么？", "top_k": 5}'
```

### 健康检查

```bash
curl http://localhost:8000/api/health
```

## 运行测试

```bash
pytest tests/ -v
```

## 性能指标

- 文档解析: 1-2分钟/PDF (GPU)
- 向量检索: < 100ms (Top-5)
- 问答响应: 3-8秒 (含推理)
- 并发支持: 10用户上传 + 5人同时问答

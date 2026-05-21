# 多模态 RAG 聊天机器人

基于 BGE + CLIP + Qwen 的多模态检索增强生成系统，支持图文混排 PDF 文档的理解和问答。

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│   Kafka     │────▶│ Document Worker │
│   Web UI    │     │   Queue     │     │  (MinerU+CLIP)  │
└──────┬──────┘     └─────────────┘     └────────┬────────┘
       │                                          │
       │                                          ▼
       │                                   ┌─────────────┐
       │                                   │   Milvus    │
       │                                   │   Vectors   │
       │                                   └──────┬──────┘
       │                                          │
       └──────────────────────────────────────────┘
                         │
                         ▼
                   ┌─────────────┐
                   │  Qwen-Plus  │
                   │  Generate   │
                   └─────────────┘
```

## 功能特性

- 📄 **文档上传**：支持 PDF 格式文档上传
- 🔍 **文档解析**：使用 MinerU 将 PDF 解析为 Markdown + 图片
- 🧠 **多模态编码**：BGE 编码文本，CLIP 编码文本+图片
- 💾 **向量存储**：Milvus 向量数据库存储
- 💬 **智能问答**：基于 RAG 的问答，支持图文混合回答

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp config.py config_local.py

# 编辑配置，设置你的 API Key 和向量数据库地址
```

### 3. 启动服务

**启动 Streamlit Web 界面：**
```bash
streamlit run main.py
```

**启动文档处理 Worker（另一个终端）：**
```bash
python workers/document_processor.py
```

### 4. 使用

1. 打开浏览器访问 `http://localhost:8501`
2. 在"文件管理"页面上传 PDF 文档
3. 等待文档解析完成
4. 在"图文问答"页面进行提问

## 项目结构

```
.
├── README.md                   # 项目说明
├── requirements.txt            # 依赖列表
├── config.py                   # 配置文件
├── main.py                     # Streamlit 主入口
├── models/
│   └── file_model.py           # SQLite 数据模型
├── services/
│   ├── upload_service.py       # 上传服务
│   ├── chat_service.py         # 问答服务
│   └── embedding_service.py    # 向量编码服务
├── utils/
│   └── milvus_client.py        # Milvus 客户端
├── workers/
│   └── document_processor.py   # 文档处理 Worker
└── pages/
    ├── upload_page.py          # 上传页面
    └── chat_page.py            # 问答页面
```

## 核心接口

| 接口 | 说明 |
|------|------|
| `POST /upload/document` | 上传 PDF 文档 |
| `GET /files` | 获取文件列表 |
| `DELETE /files/{id}` | 删除文件 |
| `POST /chat` | 问答接口 |

## 技术栈

- **Web UI**: Streamlit
- **消息队列**: Kafka
- **向量数据库**: Milvus
- **关系数据库**: SQLite
- **文档解析**: MinerU
- **文本 Embedding**: BGE-small-zh-v1.5
- **多模态 Embedding**: jina-clip-v2
- **大模型**: Qwen-Plus (DashScope)

## 注意事项

1. **MinerU 服务**：需要单独部署 MinerU 解析服务，默认地址 `http://127.0.0.1:30000`
2. **Kafka**：需要启动 Kafka 服务，默认地址 `localhost:9092`
3. **Milvus**：需要配置 Milvus 向量数据库连接信息
4. **API Key**：需要在 `config.py` 中配置 DashScope API Key

## 测试

```bash
# 运行单元测试
pytest tests/unit -v

# 运行集成测试
pytest tests/integration -v --integration

# 运行所有测试
pytest tests/ -v
```

## License

MIT

# 多模态RAG问答系统 - 需求文档

## 1. 系统概述
构建一个多模态 RAG（检索增强生成）问答系统，能够从包含图文混排的 PDF 知识库中检索相关信息，进行跨模态推理，并生成附带来源的准确答案。

## 2. 核心功能

### 2.1 知识库管理
- **知识库 CRUD**：创建、查询、删除知识库
- **文档上传**：向指定知识库上传 PDF 文档
- **文档解析**：异步解析 PDF（提取 markdown 文本 + 图片文件）
- **文档管理**：查看/删除知识库中的文档

### 2.2 多模态检索
- **文本检索**：使用 BGE 模型对文本 chunk 进行向量化，检索相关文本
- **图像检索**：使用 CLIP 模型对图片进行向量化，检索相关图片
- **混合检索**：同时检索文本和图片，按相关性排序返回 Top-K 结果

### 2.3 多模态问答
- 接收用户问题 + 知识库 ID
- 自动检索相关文本和图片
- 将检索结果作为上下文，使用 Qwen-VL 进行图文推理
- 生成带来源引用的答案（哪个 PDF、哪一页、哪个图表）

### 2.4 异步文档处理
- 上传文档后，通过 Kafka 消息队列异步处理
- Worker 服务消费消息，调用 mineru/DeepSeek-OCR 解析文档
- 解析完成后自动进行 chunk 分割、embedding、存入向量库

## 3. 系统架构

```
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  FastAPI     │    │   Kafka         │    │   Worker         │
│  (Upload/    │───▶│  document_parse │───▶│  parse_document  │
│   Chat API)  │    └─────────────────┘    │  (Offline)       │
│             │                            └────────┬─────────┘
└──────┬──────┘                                     │
       │                                            ▼
       │                                    ┌─────────────────┐
       │                                    │  MinerU/OCR     │
       │                                    │  PDF → MD+图    │
       │                                    └────────┬─────────┘
       │                                             │
       ▼                                             ▼
┌─────────────────┐                      ┌──────────────────────┐
│   Milvus        │◀─────────────────────│  Chunk/Image Embed   │
│   向量数据库     │                      │  (BGE + CLIP)        │
└─────────────────┘                      └──────────────────────┘
       ▲
       │
┌──────┴──────┐    ┌──────────────────┐
│  Qwen-VL    │    │  SQLite/MySQL    │
│  多模态推理  │    │  元信息存储      │
└─────────────┘    └──────────────────┘
```

## 4. API 接口定义

### 4.1 知识库管理
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/knowledge-bases` | 创建知识库 |
| GET | `/api/v1/knowledge-bases` | 知识库列表 |
| GET | `/api/v1/knowledge-bases/{kb_id}` | 知识库详情 |
| DELETE | `/api/v1/knowledge-bases/{kb_id}` | 删除知识库 |

### 4.2 文档管理
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/knowledge-bases/{kb_id}/documents` | 上传文档 |
| GET | `/api/v1/knowledge-bases/{kb_id}/documents` | 文档列表 |
| DELETE | `/api/v1/knowledge-bases/{kb_id}/documents/{doc_id}` | 删除文档 |
| GET | `/api/v1/documents/{doc_id}/parse-status` | 查询解析状态 |

### 4.3 检索与问答
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/retrieve` | 多模态检索（文本+图片） |
| POST | `/api/v1/chat` | 多模态问答 |

## 5. 数据结构

### 5.1 知识库 KnowledgeBase
- `id`: str (UUID)
- `name`: str
- `description`: str (可选)
- `created_at`: datetime
- `updated_at`: datetime

### 5.2 文档 Document
- `id`: str (UUID)
- `kb_id`: str (外键)
- `filename`: str
- `file_path`: str
- `file_size`: int
- `page_count`: int (解析后更新)
- `parse_status`: enum (pending/parsing/completed/failed)
- `created_at`: datetime

### 5.3 Chunk (文本片段)
- `id`: str (UUID)
- `doc_id`: str (外键)
- `kb_id`: str
- `content`: text
- `page_num`: int
- `embedding`: vector (BGE 768维)
- `metadata`: json

### 5.4 Image (图片)
- `id`: str (UUID)
- `doc_id`: str (外键)
- `kb_id`: str
- `file_path`: str
- `page_num`: int
- `caption`: str (可选，来自 OCR)
- `embedding`: vector (CLIP 512维)
- `metadata`: json

## 6. 评价方法
每个回答综合评分（满分1.0）：
- **页面匹配度** (0.25分)：答案引用的页面与实际来源页面是否一致
- **文件名匹配度** (0.25分)：答案引用的文件名与实际来源是否一致
- **答案内容相似度** (0.5分)：Jaccard 相似系数（字符交集/并集）

## 7. 技术栈
- **框架**: FastAPI + Uvicorn
- **文档解析**: MinerU / DeepSeek-OCR
- **文本编码**: BGE (BAAI/bge-large-zh-v1.5)
- **图像编码**: CLIP (openai/clip-vit-base-patch32)
- **向量检索**: Milvus (pymilvus)
- **消息队列**: Kafka (confluent-kafka)
- **元数据存储**: SQLite (开发) / MySQL (生产)
- **多模态模型**: Qwen-VL (qwen-vl-plus 或 qwen-vl-max)

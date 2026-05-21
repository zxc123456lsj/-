# API接口规范

## 基础信息
- Base URL: `http://localhost:8000/api`
- Content-Type: `application/json`

---

## 1. 文件管理接口

### 1.1 上传文档
**POST** `/upload`

**Request**:
Content-Type: multipart/form-data file: <PDF文件> knowledge_base_id: 1 (可选)

**Response** (200 OK):
```json
{ "status": "success", "file_id": 123, "filename": "example.pdf", "message": "文件已上传，正在后台解析" }
```

---

### 1.2 查询文件状态
**GET** `/files/{file_id}/status`

**Response** (200 OK):
```json
{ "file_id": 123, "filename": "example.pdf", "status": "processing", "progress": 60, "created_at": "2026-05-21T10:00:00" }
```

**Status枚举**: pending, processing, completed, failed

---

### 1.3 列出所有文件
**GET** `/files`

**Query Parameters**: page, limit, status

**Response** (200 OK):
```json
{ "total": 50, "page": 1, "limit": 20, "files": [ { "id": 123, "filename": "example.pdf", "status": "completed", "created_at": "2026-05-21T10:00:00" } ] }
```

---

## 2. 问答接口

### 2.1 智能问答
**POST** `/chat`

**Request**:
```json
{ "query": "这张图表显示了什么？", "knowledge_base_ids": [1, 2], "top_k": 5 }
```

**Response** (200 OK):
```json
{ "answer": "根据文档第3页的图表显示...", "sources": [ { "text": "相关文本片段...", "file_name": "example.pdf", "page": 3, "image_url": "/static/images/xxx.jpg", "relevance_score": 0.92 } ], "model": "Qwen-VL-Chat", "response_time_ms": 3500 }
```

---

## 3. 健康检查

**GET** `/health`

**Response** (200 OK):
```json
{ "status": "healthy", "services": { "kafka": "connected", "milvus": "connected", "mineru": "available" } }
```

---

## 错误码规范

| 状态码 | 含义 |
|--------|------|
| 200 | 成功 |
| 400 | 请求错误 |
| 404 | 未找到 |
| 413 | 文件过大 |
| 500 | 服务器错误 |

**错误响应格式**:
```json
{ "status": "error", "code": 404, "message": "文件不存在" }
```

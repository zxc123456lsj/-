# 系统架构详细设计

## 整体架构图
```mermaid 
graph TB 
A[Streamlit前端] -->|上传PDF| B[FastAPI后端] B -->|保存文件| C[./uploads/] B -->|发送消息| D[Kafka: rag-data]
D -->|消费| E[Offline Worker]
E -->|调用| F[MinerU解析]
F -->|输出| G[Markdown + Images]

G -->|编码| H[BGE/CLIP模型]
H -->|向量| I[Milvus向量库]

J[用户提问] --> A
A -->|查询| B
B -->|检索| I
I -->|返回片段| K[Qwen-VL推理]
K -->|生成答案| A
```

## 数据流详解

### 1. 文档上传流程
```mermaid 
graph TD
用户选择PDF 
-->Streamlit接收文件 
-->FastAPI验证(MD5去重) 
-->保存到 ./uploads/{filename} 
-->SQLite创建记录(status=pending) 
-->Kafka Producer发送消息 
-->返回file_id给用户
```

### 2. 离线解析流程
```mermaid
graph TD
Worker监听Kafka 
-->消费rag-data topic 
-->调用MinerU解析PDF 
-->输出: content.md + images/ 
-->读取Markdown分割chunks(256字符) 
-->BGE编码文本向量(512维) 
-->CLIP编码图像向量(1024维) 
-->批量插入Milvus 
-->更新SQLite状态(status=completed)
```

### 3. 智能问答流程
```mermaid
graph TD
用户输入问题 
↓ BGE编码问题向量 
↓ Milvus检索Top-5文本片段 
↓ CLIP编码问题向量 
↓ Milvus检索Top-3图像片段 
↓ 合并排序取Top-5 
↓ 构建Prompt(问题+参考片段) 
↓ Qwen-VL生成答案 
↓ 格式化输出(答案+来源)
```

## 数据库设计

### SQLite表结构

**files表**
```sql
CREATE TABLE files ( 
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    filename VARCHAR(255) NOT NULL, 
    filepath VARCHAR(1000) NOT NULL, 
    file_hash VARCHAR(64) UNIQUE, 
    filestate VARCHAR(20) DEFAULT 'pending', 
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, 
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP 
);
```

**processing_logs表**
```sql
CREATE TABLE processing_logs ( 
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    file_id INTEGER, status VARCHAR(20), 
    error_message TEXT, 
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY (file_id) REFERENCES files(id) 
);
```

### Milvus集合设计

**集合名**: rag_data

**Schema**:
```python
{ 
    "id": INT64 (主键), 
    "text_vector": FLOAT_VECTOR(512), 
    "clip_text_vector": FLOAT_VECTOR(1024), 
    "clip_image_vector": FLOAT_VECTOR(1024), 
    "text": VARCHAR(65535), 
    "db_id": INT64, 
    "file_name": VARCHAR(255), 
    "file_path": VARCHAR(1000), 
    "chunk_index": INT64 
}
```

**索引配置**:
- text_vector: IVF_FLAT, nlist=128, metric_type=COSINE
- clip_text_vector: IVF_FLAT, nlist=128, metric_type=COSINE
- clip_image_vector: IVF_FLAT, nlist=128, metric_type=COSINE

## 服务依赖关系
```mermaid
graph LR
    ┌─────────────────┐ │ Streamlit UI │ └────────┬────────┘ │ HTTP ┌────────▼────────┐ │ FastAPI │ │ Backend │ └───┬─────────┬───┘ │ │ ┌───▼──┐ ┌───▼──────┐ │Kafka │ │ SQLite │ │Queue │ │ │ └───┬──┘ └──────────┘ │ ┌───▼──────────┐ │ Offline │ │ Worker │ └───┬──────────┘ │ ┌───▼──────────────────┐ │ External Services │ │ • MinerU (:30000) │ │ • Milvus (:19530) │ │ • Qwen-VL │ └──────────────────────┘
```

## 部署架构

### Docker Compose服务
```yaml
services: zookeeper: image: confluentinc/cp-zookeeper:7.5.0 ports: ["2181:2181"]
kafka: image: confluentinc/cp-kafka:7.5.0 ports: ["9092:9092"] depends_on: [zookeeper]
milvus: image: milvusdb/milvus:v2.3.0 ports: ["19530:19530"]
mineru: image: opendatalab/mineru:latest ports: ["30000:30000"]
```

### 端口映射总览
| 服务 | 端口 | 用途 |
|------|------|------|
| Zookeeper | 2181 | Kafka协调 |
| Kafka | 9092 | 消息队列 |
| Milvus | 19530 | 向量检索 |
| MinerU | 30000 | 文档解析 |
| FastAPI | 8000 | REST API |
| Streamlit | 8501 | Web界面 |

## 性能指标

### 预期性能
- 文档解析: 1-2分钟/PDF (GPU)
- 向量检索: < 100ms (Top-5)
- 问答响应: 3-8秒 (含推理)
- 并发支持: 10用户上传 + 5人同时问答

### 资源需求
- CPU: 8核+
- 内存: 16GB+
- GPU: RTX 3090/A100 (显存12GB+)
- 磁盘: 100GB SSD

# 分阶段开发任务清单

## Phase 1: 基础设施搭建 (预计2天)

### Task 1.1: 项目初始化
- [ ] 创建目录结构
```
multimodal-rag/ 
├── app/ 
│ ├── api/ 
│ ├── models/ 
│ ├── services/ 
│ ├── worker/ 
│ └── utils/ 
├── uploads/ 
├── processed/ 
├── static/ 
├── tests/ 
├── docker-compose.yml 
├── requirements.txt 
├── .env.example 
└── README.md
```

- [ ] 编写requirements.txt
```text
fastapi==0.109.0 
uvicorn==0.27.0 
sqlalchemy==2.0.25 
kafka-python==2.0.2 
pymilvus==2.3.5 
streamlit==1.30.0 
sentence-transformers==2.3.1 
pillow==10.2.0 
python-multipart==0.0.6
```

- [ ] 创建.env.example
```env
DATABASE_URL=sqlite:///./db.db 
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 
KAFKA_TOPIC=rag-data 
MILVUS_URI=http://localhost:19530 
MILVUS_COLLECTION=rag_data 
MINERU_API_URL=http://localhost:30000 
BGE_MODEL_PATH=BAAI/bge-small-zh-v1.5 
CLIP_MODEL_PATH=jinaai/jina-clip-v2 
QWEN_VL_MODEL=Qwen/Qwen-VL-Chat
```

---

### Task 1.2: Docker服务配置
- [ ] 编写docker-compose.yml
```yaml
version: '3.8'
services: 
 zookeeper: 
  image: confluentinc/cp-zookeeper:7.5.0 
  ports: ["2181:2181"] 
  environment: 
    ZOOKEEPER_CLIENT_PORT: 2181
 kafka:
  image: confluentinc/cp-kafka:7.5.0
  ports: ["9092:9092"]
  depends_on: [zookeeper]
  environment:
    KAFKA_BROKER_ID: 1
    KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092

 milvus:
  image: milvusdb/milvus:v2.3.0
  ports: ["19530:19530"]
  volumes:
    - milvus-data:/var/lib/milvus
volumes: milvus-data:
```

---

### Task 1.3: 数据库模型定义
- [ ] 创建app/models/database.py
```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime 
from sqlalchemy.ext.declarative import declarative_base 
from datetime import datetime
Base = declarative_base()
class File(Base): tablename = 'files'
  id = Column(Integer, primary_key=True)
  filename = Column(String(255), nullable=False)
  filepath = Column(String(1000), nullable=False)
  file_hash = Column(String(64), unique=True)
  filestate = Column(String(20), default='pending')
  created_at = Column(DateTime, default=datetime.now)
  updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
```

---

## Phase 2: 核心功能开发 (预计5天)

### Task 2.1: 文件上传服务
- [ ] 实现app/api/upload.py
- [ ] 实现Kafka Producer
- [ ] 文件MD5去重逻辑
- [ ] 单元测试

---

### Task 2.2: 离线解析Worker
- [ ] 创建app/worker/parser.py
- [ ] Kafka Consumer实现
- [ ] MinerU集成调用
- [ ] Chunk分割算法
- [ ] 向量化与存储

---

### Task 2.3: 检索引擎
- [ ] 创建app/services/retriever.py
- [ ] BGE文本检索
- [ ] CLIP图像检索
- [ ] 结果合并排序

---

### Task 2.4: 问答服务
- [ ] 创建app/services/chat.py
- [ ] Qwen-VL集成
- [ ] Prompt工程
- [ ] 答案格式化

---

## Phase 3: 前端界面开发 (预计2天)

### Task 3.1: Streamlit应用
- [ ] 创建web_app.py
- [ ] 文件上传页面
- [ ] 聊天界面
- [ ] 参考来源展示

---

## Phase 4: 测试与优化 (预计2天)

### Task 4.1: 集成测试
- [ ] 编写tests/test_upload.py
- [ ] 编写tests/test_retrieval.py
- [ ] 编写tests/test_chat.py

### Task 4.2: 性能优化
- [ ] 添加缓存机制
- [ ] 优化Milvus索引
- [ ] 批量插入优化
- [ ] 监控日志完善

---

## 进度跟踪
```
Phase 1: [ ] 0% 
Phase 2: [ ] 0% 
Phase 3: [ ] 0% 
Phase 4: [ ] 0%
总体进度: 0%
```
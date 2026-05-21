---
name: multimodal-rag-dev
description: 开发多模态RAG问答系统的完整指南。支持PDF文档解析、向量检索、智能问答。技术栈：FastAPI + Kafka + Milvus + MinerU + BGE + CLIP + Qwen-VL + Streamlit。当用户需要开发文档解析系统、构建知识库问答、实现图文检索时使用此skill。
---

# 多模态RAG系统开发指南

## 核心目标
构建支持PDF图文解析、多模态向量检索、智能问答的完整系统。

## 技术栈（必须遵守）

| 层级 | 技术选型 |
|------|---------|
| **后端框架** | FastAPI + Uvicorn |
| **消息队列** | Kafka + Zookeeper |
| **向量数据库** | Milvus |
| **元数据存储** | SQLite/SQLAlchemy |
| **文档解析** | MinerU (vlm-http-client模式) |
| **文本编码** | BGE-small-zh-v1.5 (512维) |
| **图像编码** | Jina-CLIP-v2 (1024维) |
| **推理模型** | Qwen-VL-Chat |
| **前端界面** | Streamlit |

## 系统架构
用户上传 → FastAPI → Kafka → Worker解析 → Milvus存储 ↓ 用户提问 → 检索引擎 → Qwen-VL推理 → 返回答案+来源

### 关键约束
1. **异步处理**：MinerU解析耗时1分钟，必须用Kafka解耦
2. **GPU依赖**：MinerU和Qwen-VL需要GPU支持
3. **多模态检索**：同时检索文本和图像向量
4. **引用溯源**：答案必须标注信息来源（文件名+页码）

## 开发流程（严格按Phase顺序）

### Phase 1: 基础设施搭建
→ 项目初始化 + Docker配置(Kafka+Milvus) + 数据库模型

### Phase 2: 核心功能开发
→ 文件上传API → Worker解析服务 → 检索引擎 → 问答接口

### Phase 3: 前端界面开发
→ Streamlit应用(上传页面+聊天页面+文件管理)

### Phase 4: 测试与优化
→ 集成测试 + 性能优化 + 监控日志

## 快速开始指令

当用户说"开始开发多模态RAG系统"时：

1. **确认环境准备**
   - Docker Desktop已启动
   - GPU驱动正常
   - Python 3.10+环境

2. **从Phase 1开始**
   - 创建项目目录结构
   - 生成docker-compose.yml
   - 编写requirements.txt

3. **每完成一个Task汇报进度**
   - 列出已完成项✅
   - 等待用户确认后继续

## 详细文档索引

- **完整架构设计**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API接口规范**: [API_SPEC.md](API_SPEC.md)
- **分阶段任务清单**: [PHASE_TASKS.md](PHASE_TASKS.md)

## 常见问题处理

### MinerU启动失败
```bash
# 检查推理引擎
pip install lmdeploy
# 或使用Docker
docker run -d -p 30000:30000 opendatalab/mineru:latest
```

### Kafka连接失败
```bash
# 检查容器状态
docker-compose ps
# 查看日志
docker logs kafka
```

### Milvus初始化错误
```python
# 确保集合存在
client.create_collection("rag_data", dimension=512)
```

## 代码质量标准

- ✅ 所有函数有类型注解
- ✅ 关键逻辑有注释说明
- ✅ 异常处理完善
- ✅ 配置使用.env文件
- ✅ 符合PEP8规范

## 交付物要求

完成后提供：
1. 完整可运行的代码
2. requirements.txt依赖清单
3. docker-compose.yml配置文件
4. .env.example环境变量模板
5. README.md使用文档
6. 测试报告

---

**现在开始行动**：询问用户是否从Phase 1开始，或查看PHASE_TASKS.md了解完整任务清单。

# 多模态 RAG 图文问答平台 —— 完整实现

## 1. 需求澄清

根据项目文档和已有代码，需要**完整实现以下功能**：

| 模块                                     | 要求                                                         |
| :--------------------------------------- | :----------------------------------------------------------- |
| **文档上传接口** `POST /upload/document` | 接收 PDF/DOCX/TXT，落盘，发送 Kafka 消息触发异步解析         |
| **多模态问答接口** `POST /chat`          | 接收用户问题，执行**文本+图像混合检索**，调用 Qwen-VL 多模态模型生成图文并茂的答案 |
| **离线解析 Worker**                      | 消费 Kafka，调用 MinerU 解析 PDF → Markdown + 图片，用 BGE/CLIP 编码存入 Milvus |
| **测试逻辑**                             | 模拟上传、解析完成、问答全流程，评估页面/文件名/内容相似度   |

原有 Streamlit 界面（`web_page_upload.py` / `web_page_chat.py`）保留作为演示前端，但**核心接口必须通过 FastAPI 暴露**，便于集成与测试。



------

## 2. 项目文件结构（核心部分）

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口 + 路由注册
│   ├── api/
│   │   ├── upload.py           # POST /upload/document
│   │   └── chat.py             # POST /chat
│   ├── services/
│   │   ├── retriever.py        # 混合检索（文本+图像向量）
│   │   └── qwen_client.py      # Qwen-VL 多模态调用
│   └── static/                 # 挂载图片目录
├── worker/
│   └── offline_process_worker.py   # 增强版（修复图片路径、编码逻辑）
├── orm_model.py                # SQLite 文件表（不变）
├── web_demo.py                 # Streamlit 导航（不变）
├── web_page_upload.py          # Streamlit 上传（不变，但注意与 FastAPI 共享）
├── web_page_chat.py            # 可保留，但建议重定向到 FastAPI
├── test_api.py                 # 测试脚本（上传 + 问答 + 评分）
└── requirements.txt            # 依赖清单
```



------

##  3.总结

| 需求           | 实现状态                                  |
| :------------- | :---------------------------------------- |
| 文档上传接口   | ✅ FastAPI + Kafka 异步解析                |
| 多模态问答接口 | ✅ 混合检索（文本+图像）+ Qwen-VL 图文推理 |
| 离线 Worker    | ✅ MinerU + BGE/CLIP + Milvus              |
| 测试逻辑       | ✅ 上传→等待→问答→Jaccard 评分             |
| 图文关联推理   | ✅ 检索图片 URL 并传给多模态模型           |

整个系统完全可通过 `test_api.py` 快速验证。

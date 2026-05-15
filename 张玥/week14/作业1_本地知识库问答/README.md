# 作业 1：LangChain 本地知识库问答

本作业实现一个轻量版本地知识库问答流程，重点不是搭建复杂向量数据库，而是帮助你理解 RAG 的最小闭环。

## 学习目标

你需要理解这条链路：

```text
用户问题 -> 本地文档检索 -> 取出相关上下文 -> 交给大模型回答
```

从 Java 后端角度类比：

- 本地知识库：类似一批业务文档或数据库里的知识表。
- 检索器：类似根据关键词和相似度做查询的 DAO。
- Prompt 拼接：类似组装请求 DTO。
- LLM 回答：类似调用一个外部智能服务生成最终响应。

## 为什么不用向量数据库

本作业只要求“文档检索 + LLM 回答流程”，因此使用当前环境已有的 `scikit-learn` 做轻量检索：

- 不需要额外安装 `Chroma` 或 `FAISS`
- 更容易看懂检索过程
- 适合作业演示和初学阶段理解

## 运行方式

```powershell
cd D:\AI_study_env\files\study\Week14\homework\作业1_本地知识库问答
D:\AI_study_env\miniconda3\envs\py312\python.exe main.py
```

指定自己的问题：

```powershell
D:\AI_study_env\miniconda3\envs\py312\python.exe main.py --question "LangGraph 为什么适合做复杂 Agent 流程？"
```

## 启用 LLM 回答

程序默认读取这些环境变量：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:OPENAI_MODEL="qwen-flash"
```

如果没有配置 API Key，程序会跳过 LLM 调用，只展示检索结果。

## 示例问题

- LangChain 主要解决什么问题？
- LangGraph 和普通链式调用有什么区别？
- Deep Agents 适合什么样的任务？

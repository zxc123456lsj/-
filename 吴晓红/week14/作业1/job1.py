"""
基于LangChain的本地知识库问答系统
实现文档检索 + LLM回答流程

依赖安装：
pip install langchain langchain-openai langchain-community chromadb langchain-text-splitters langchain-core
# 注意：新版本LangChain中RetrievalQA模块可能缺失，本代码使用自定义实现替代
"""

import os
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


# 配置
DOCS_DIR = "./doc"  # 文档目录，请将文本文件放入此目录
CHROMA_PERSIST_DIR = "./chroma_db"  # 向量数据库持久化目录

# DashScope API配置（阿里云通义千问兼容模式）
# 请通过环境变量 DASHSCOPE_API_KEY 设置您的API密钥，或使用下面的默认值（可能受限）
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 设置基础URL环境变量，确保OpenAI兼容客户端正确连接到DashScope
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 模型设置
LLM_MODEL = "qwen-flash"  # 对话模型
# 嵌入模型设置：使用HuggingFace本地嵌入模型避免DashScope API错误
# 可选模型：'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'（多语言）
# 或 'sentence-transformers/all-MiniLM-L6-v2'（英文）
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
USE_HUGGINGFACE_EMBEDDINGS = True  # 设置为True使用HuggingFace嵌入，False使用DashScope嵌入


def load_and_split_documents(docs_dir: str):
    """加载文档并分割成 chunk"""
    if not os.path.exists(docs_dir):
        print(f"文档目录 {docs_dir} 不存在，请创建并放入文本文件。")
        print(f"将自动创建示例文档...")
        os.makedirs(docs_dir, exist_ok=True)
        sample_doc = os.path.join(docs_dir, "sample.txt")
        with open(sample_doc, "w", encoding="utf-8") as f:
            f.write("LangChain是一个用于开发大语言模型应用的框架。\n")
            f.write("它提供了模块化的组件，方便构建复杂的应用。\n")
            f.write("RAG（检索增强生成）是结合检索和生成的技术。\n")
            f.write("向量数据库用于存储文档的嵌入表示，便于快速检索。\n")
            f.write("检索器根据用户问题找到最相关的文档片段。\n")
        print(f"已创建示例文档: {sample_doc}")

    # 加载所有txt文件（可按需添加其他格式的loader）
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"已加载 {len(documents)} 个文档")

    if len(documents) == 0:
        print("未找到任何文档，请检查目录和文件格式。")
        return []

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"分割成 {len(splits)} 个 chunk")
    return splits


def create_vectorstore(splits, persist_dir: str, embedding_model_name: str, api_key: str):
    """创建或加载向量数据库"""
    if len(splits) == 0:
        raise ValueError("没有可处理的文档内容")

    # 根据标志选择嵌入模型
    if USE_HUGGINGFACE_EMBEDDINGS:
        print(f"使用HuggingFace嵌入模型: {embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # 使用CPU，如需GPU可改为 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        print(f"使用DashScope嵌入模型: {embedding_model_name}")
        # 使用 DashScope Embeddings (通过 OpenAIEmbeddings 兼容接口)
        # 模型名称必须是DashScope支持的，如 'text-embedding-v1' 或 'text-embedding-v2'
        embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            base_url=BASE_URL,
            api_key=api_key,  # type: ignore
        )

    # 创建向量存储（如果已存在则加载）
    if os.path.exists(persist_dir):
        print(f"加载已有向量数据库: {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        vectorstore.persist()
    return vectorstore


def create_llm(model_name: str, api_key: str):
    """创建LLM实例"""
    llm = ChatOpenAI(
        model=model_name,
        base_url=BASE_URL,
        api_key=api_key,  # type: ignore
        temperature=0.1,
        streaming=False,  # 如需流式输出可改为True
    )
    return llm


def create_qa_chain(vectorstore, llm):
    """创建检索问答链（自定义实现，不依赖RetrievalQA）"""
    # 自定义提示模板，可调整以提高回答质量
    prompt_template = """基于以下上下文回答问题。如果你不知道答案，就说不知道，不要编造。

上下文：
{context}

问题：{question}
回答："""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # 返回top 4个相关chunk
    )

    # 自定义问答链类
    class CustomRetrievalQA:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, inputs):
            """inputs 应包含 'query' 键"""
            query = inputs.get("query", "")
            # 检索相关文档
            source_documents = self.retriever.invoke(query)
            # 合并文档内容作为上下文
            context = "\n\n".join([doc.page_content for doc in source_documents])
            # 构建提示
            prompt_value = self.prompt.invoke({"context": context, "question": query})
            # 调用LLM
            response = self.llm.invoke(prompt_value)
            # 提取回答文本（假设response是AIMessage类型）
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            # 返回格式与RetrievalQA兼容
            return {
                "result": answer,
                "source_documents": source_documents
            }

    # 创建实例
    qa_chain = CustomRetrievalQA(llm, retriever, PROMPT)
    return qa_chain


def interactive_qa(qa_chain):
    """交互式问答循环"""
    print("\n问答系统已就绪，请输入问题（输入 'quit' 或 'q' 退出）")
    while True:
        try:
            question = input("\n问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n检测到退出信号，再见！")
            break

        if question.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
        if not question:
            continue

        # 执行检索与生成
        try:
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            source_docs = result["source_documents"]

            print(f"\n回答：{answer}")
            if source_docs:
                print(f"\n参考来源：")
                for i, doc in enumerate(source_docs, 1):
                    source = doc.metadata.get('source', '未知')
                    # 提取文件名
                    if isinstance(source, str):
                        filename = os.path.basename(source)
                    else:
                        filename = str(source)
                    print(f"  {i}. {filename}")
                    # 可选：显示前100个字符
                    # print(f"    内容片段: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"出错: {e}")


def main():
    """主流程"""
    print("=" * 50)
    print("本地知识库问答系统 (基于LangChain RAG)")
    print("=" * 50)

    # 1. 加载并分割文档
    splits = load_and_split_documents(DOCS_DIR)
    if len(splits) == 0:
        print("没有可处理的文档，退出。")
        return

    # 2. 创建向量数据库
    try:
        vectorstore = create_vectorstore(
            splits,
            CHROMA_PERSIST_DIR,
            EMBEDDING_MODEL_NAME,
            API_KEY
        )
    except Exception as e:
        print(f"创建向量数据库失败: {e}")
        if USE_HUGGINGFACE_EMBEDDINGS:
            print("请检查HuggingFace模型名称是否正确，或尝试其他模型。")
            print(f"当前使用的嵌入模型: {EMBEDDING_MODEL_NAME}")
            print("推荐模型: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'")
        else:
            print("请检查嵌入模型名称和API密钥是否正确。")
            print(f"当前使用的嵌入模型: {EMBEDDING_MODEL_NAME}")
            print("DashScope支持的嵌入模型: 'text-embedding-v1', 'text-embedding-v2'")
        return

    # 3. 创建LLM
    try:
        llm = create_llm(LLM_MODEL, API_KEY)
    except Exception as e:
        print(f"创建LLM失败: {e}")
        print("请检查API密钥和网络连接。")
        return

    # 4. 创建问答链
    qa_chain = create_qa_chain(vectorstore, llm)

    # 5. 交互式问答
    interactive_qa(qa_chain)


if __name__ == "__main__":
    main()
# rag_qa_system.py
# 依赖安装：pip install langchain chromadb openai tiktoken pypdf

import os
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ================== 配置 ==================
# 本地知识库文件夹路径（里面放 .txt .pdf 等文件）
KNOWLEDGE_DIR = "./knowledge_base"

# 持久化向量库存储路径
PERSIST_DIR = "./chroma_db"

# LLM 配置（这里使用 OpenAI 兼容接口，可替换为任意 API）
LLM_MODEL = "gpt-3.5-turbo"
LLM_BASE_URL = "https://api.openai.com/v1"  # 或你的代理地址
LLM_API_KEY = "sk-xxxx"


# ================== 步骤1：加载文档 ==================
def load_documents(directory):
    """支持 .txt .pdf .md 文件的加载"""
    loaders = []
    # 文本文件
    loaders.append(TextLoader)
    # PDF 文件需要额外安装 pypdf
    try:
        from langchain.document_loaders import PyPDFLoader
        loaders.append(PyPDFLoader)
    except:
        pass

    docs = []
    # 遍历目录下所有文件，根据扩展名选择 loader
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs


# ================== 步骤2：文档切分 ==================
def split_documents(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    return text_splitter.split_documents(docs)


# ================== 步骤3：构建向量库（首次运行） ==================
def build_vectorstore(docs, persist_directory):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_BASE_URL
    )
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore


# ================== 步骤4：检索 + LLM 问答 ==================
def get_qa_chain(vectorstore, llm_model, llm_base_url, llm_api_key):
    """返回一个可调用的问答链"""
    llm = ChatOpenAI(
        model=llm_model,
        openai_api_key=llm_api_key,
        openai_api_base=llm_base_url,
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 检索4个相关片段
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


# ================== 主流程 ==================
if __name__ == "__main__":
    # 1. 加载并切分文档
    print("加载文档...")
    raw_docs = load_documents(KNOWLEDGE_DIR)
    print(f"加载了 {len(raw_docs)} 个原始文档")

    print("切分文档...")
    split_docs = split_documents(raw_docs)
    print(f"切分为 {len(split_docs)} 个文本块")

    # 2. 构建向量库（如果已存在则加载，否则创建）
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        print("加载已有向量库...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL
        )
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        print("创建向量库...")
        vectorstore = build_vectorstore(split_docs, PERSIST_DIR)

    # 3. 构建问答链
    qa = get_qa_chain(vectorstore, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY)

    # 4. 交互式问答
    print("\n知识库问答系统已就绪，输入问题（输入 exit 退出）")
    while True:
        query = input("\n用户问题：")
        if query.lower() == "exit":
            break
        result = qa({"query": query})
        print(f"回答：{result['result']}")
        # 可选：打印引用来源
        # print(f"引用来源：{result['source_documents']}")
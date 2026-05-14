"""
本地知识库问答 (RAG) 实现
步骤：
1. 加载本地文档（支持 .txt, .md 等文件）
2. 将文档分割成小块
3. 生成嵌入向量并存储到 Chroma 向量数据库
4. 根据用户问题检索最相关的文档片段
5. 将检索结果作为上下文，调用 LLM 生成最终答案
"""

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# ========== 配置 ==========
# 本地知识库路径
KNOWLEDGE_DIR = "./knowledge_base"   # 存放 .txt 或 .md 文件的目录

# 通义千问模型配置（与示例保持一致）
LLM_CONFIG = {
    "model": "qwen-flash",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-4fedee4ece6541d3b17a7173f0b3c16f"  # 替换有效 API Key
}

# 嵌入模型（本地运行，无需 API Key）
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 持久化向量数据库路径
VECTOR_STORE_PATH = "./chroma_db"

# ========== 1. 加载文档 ==========
def load_documents(directory: str):
    """加载目录下所有 .txt 和 .md 文件"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"知识库目录不存在: {directory}")
    
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",   # 可根据需要添加 .md
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    documents = loader.load()
    print(f"成功加载 {len(documents)} 个文档")
    return documents

# ========== 2. 分割文档 ==========
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """将文档分割成适合检索的小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档分割为 {len(chunks)} 个文本块")
    return chunks

# ========== 3. 创建向量数据库 ==========
def create_vector_store(chunks, embedding_model, persist_dir):
    """生成嵌入并存储到 Chroma（如果已存在则直接加载）"""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # 如果已有持久化存储，直接加载
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"加载已有向量数据库: {persist_dir}")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vector_store.persist()
    return vector_store

# ========== 4. 构建 RAG 问答链 ==========
def build_qa_chain(vector_store, llm_config):
    """创建 RetrievalQA 链，包含提示模板和模型"""
    # 自定义提示模板：要求模型基于上下文回答，不知道就说不知道
    prompt_template = """你是一个知识渊博的助手，请基于以下【上下文】内容回答问题。
如果无法从上下文中找到答案，请直接说“根据现有资料无法回答”，不要编造信息。

【上下文】
{context}

【问题】
{question}

【回答】"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(**llm_config)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # 检索 4 个最相关块
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # 将所有检索到的片段一次性喂给 LLM
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # 可选，用于调试
    )
    return qa_chain

# ========== 5. 交互问答函数 ==========
def ask_question(qa_chain, question: str):
    """向知识库提问并打印回答"""
    result = qa_chain.invoke({"query": question})
    print(f"\n【用户问题】\n{question}")
    print(f"\n【助手回答】\n{result['result']}")
    # 可选：打印参考的源文档片段（调试用）
    # print("\n【参考文档】")
    # for doc in result['source_documents']:
    #     print(doc.page_content[:200] + "...\n")
    return result

# ========== 主流程 ==========
def main():
    # 1-3 准备向量数据库（如果已存在则跳过分割和索引步骤）
    try:
        # 先尝试直接加载已有向量库（加速启动）
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
            vector_store = Chroma(
                persist_directory=VECTOR_STORE_PATH,
                embedding_function=embeddings
            )
            print(f"直接加载已有向量数据库: {VECTOR_STORE_PATH}")
        else:
            # 首次运行：加载、分割、索引
            docs = load_documents(KNOWLEDGE_DIR)
            chunks = split_documents(docs)
            vector_store = create_vector_store(chunks, EMBEDDING_MODEL, VECTOR_STORE_PATH)
    except Exception as e:
        print(f"向量库准备失败: {e}")
        return
    
    # 构建问答链
    qa_chain = build_qa_chain(vector_store, LLM_CONFIG)
    
    # 交互式问答循环
    print("本地知识库问答系统已启动（输入 'exit' 退出）")
    while True:
        user_input = input("\n请输入问题: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        try:
            ask_question(qa_chain, user_input)
        except Exception as e:
            print(f"处理出错: {e}")

if __name__ == "__main__":
    main()

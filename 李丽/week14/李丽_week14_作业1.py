import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def local_knowledge_qa(file_path: str, query: str):
    """
    基于LangChain的本地知识库问答逻辑
    包含文档检索 + LLM回答流程
    """
    # 1. 加载文档
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # 2. 文档切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. 向量化与本地存储 (文档检索)
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", 
        dashscope_api_key=os.environ.get("ALIYUN_API_KEY")
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. 构建LLM回答流程
    llm = ChatTongyi(
        model="qwen-turbo", 
        temperature=0,
        dashscope_api_key=os.environ.get("ALIYUN_API_KEY")
    )
    
    # 构建Prompt模板
    system_prompt = (
        "你是一个有用的AI助手。请使用以下检索到的上下文来回答用户的问题。"
        "如果你不知道答案，请直接说不知道，不要编造答案。"
        "最多使用三句话，保持答案简洁。\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 组合检索与问答链
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. 执行查询
    print(f"正在基于本地文档检索并回答问题: '{query}'...")
    response = rag_chain.invoke({"input": query})
    
    print("\n【回答结果】:")
    print(response["answer"])
    
    return response["answer"]

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv() # 尝试从 .env 文件加载环境变量
    
    # 测试样例：
    sample_file = "sample_doc.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("LangChain是一个用于开发由大语言模型（LLMs）驱动的应用程序的框架。它使得应用程序能够连接到上下文数据，并依赖大语言模型进行推理。")
            
    test_query = "LangChain是什么？"
    local_knowledge_qa(sample_file, test_query)

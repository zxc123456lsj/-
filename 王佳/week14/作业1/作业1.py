"""
1:  基于今天讲解的langchain 的框架，开发对本地知识库进行问答的逻辑，只需要包括文档检索 + llm回答流程(参考项目2)；
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
import warnings
warnings.filterwarnings('ignore')

class LocalKnowledgeBaseQA:
    def __init__(self, pdf_path="汽车知识手册.pdf"):
        """
        初始化本地知识库问答系统
        :param pdf_path: 知识库 pdf 文件路径
        """
        self.pdf_path = pdf_path
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        self.vectorstore = None
        self.retriever = None
        self.prompt = None

    def _init_llm(self):
        """
        初始化大语言模型
        """
        llm = ChatOpenAI(
            model="qwen-flash",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-7458206891744b7aa46d6f7366fecdd5"
        )
        return llm

    def _init_embeddings(self):
        """
        初始化嵌入模型
        """
        embeddings = HuggingFaceEmbeddings(
            model_name="E:/models/BAAI/bge-small-zh-v1.5/",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": False}
        )
        return embeddings

    def load_and_split_documents(self):
        """
        加载PDF文档并进行文本分割
        """
        print("正在加载PDF文档...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"文档加载完成, 共{len(documents)}页")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
        )

        texts = text_splitter.split_documents(documents)
        print(f"文本分割完成，共{len(texts)}个片段")
        return texts

    def create_vector_store(self, texts):
        """
        创建向量存储
        :param texts: 分割后的文本片段
        """
        print("正在创建向量存储...")
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        print("向量存储创建完成")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def build_qa_chain(self):
        """
        构建问答链
        """
        prompt_template = """你是一个汽车专家，擅长回答汽车相关的问题。
请根据以下提供的参考资料来回答问题。如果资料中没有相关信息，请说明无法回答。

参考资料：
{context}

问题：{question}

请给出详细且准确的回答："""

        self.prompt = PromptTemplate.from_template(prompt_template)

    def initialize(self):
        """
        初始化整个系统
        """
        print("=== 初始化本地知识库问答系统 ===")
        texts = self.load_and_split_documents()
        self.create_vector_store(texts)
        self.build_qa_chain()
        print("=== 系统初始化完成 ===")

    def ask_question(self, question):
        """
        对问题进行问答
        :param question: 用户的问题
        :return: 系统的回答
        """
        if not self.retriever or not self.prompt:
            raise Exception("系统未初始化，请先调用initialize方法")

        print(f"用户提问：{question}")

        docs = self.retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        formatted_prompt = self.prompt.format(context=context, question=question)

        result = self.llm.invoke(formatted_prompt)

        answer = result.content

        print(f"AI回答：{answer}")
        print("\n参考来源：")
        for i, doc in enumerate(docs[:3]):
            print(f"  {i+1}. {doc.page_content[:100]}... (来源: {doc.metadata.get('source', '未知')})")
        print("-" * 50)

        return answer

def main():
    qa_system = LocalKnowledgeBaseQA()
    qa_system.initialize()

    questions = [
        "如何设置遥控钥匙的靠近解锁远离上锁功能？",
        "电动汽车充电需要注意什么？",
        "汽车保养的基本步骤有哪些？"
    ]

    for question in questions:
        qa_system.ask_question(question)
        print()

if __name__ == "__main__":
    main()

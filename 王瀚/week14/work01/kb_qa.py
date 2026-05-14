import os
from typing import Optional

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from kb_ingest import KnowledgeBaseIngestor


SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请基于以下提供的文档内容，回答用户的问题。

要求：
1. 如果文档内容中有相关信息，请用中文详细回答，并注明信息来源
2. 如果文档内容不足以回答问题，请明确说明"根据现有知识库无法回答"
3. 不要编造信息
4. 回答时保持客观准确

文档内容：
{context}"""


class KnowledgeBaseQA:
    def __init__(
        self,
        index_path: str = "knowledge/faiss_index",
        embedding_model: str = "shibing624/text2vec-base-chinese",
        llm_model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        top_k: int = 4,
    ):
        self.index_path = index_path
        self.top_k = top_k
        self._init_retriever(embedding_model)
        self._init_llm(llm_model, api_key, base_url)
        self._init_chain()

    def _init_retriever(self, embedding_model: str):
        ingestor = KnowledgeBaseIngestor(embedding_model)
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"索引不存在: {self.index_path}\n请先运行 kb_cli.py ingest 命令导入文档"
            )
        vectorstore = ingestor.load_index(self.index_path)
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

    def _init_llm(self, model: str, api_key: Optional[str], base_url: str):
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        self.llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=0.1,
        )

    def _init_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def ask(self, question: str) -> dict:
        return self.chain.invoke({"input": question})


if __name__ == '__main__':
    qa = KnowledgeBaseQA()
    print(qa.ask("知识库里有什么内容？")["answer"])
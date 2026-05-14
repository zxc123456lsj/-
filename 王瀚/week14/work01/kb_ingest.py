import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class KnowledgeBaseIngestor:
    def __init__(self, embedding_model: str = "shibing624/text2vec-base-chinese"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def load_documents(self, directory: str) -> List[Document]:
        docs = []
        supported = {".pdf": PyPDFLoader, ".txt": TextLoader}
        for fname in os.listdir(directory):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            path = os.path.join(directory, fname)
            loader = supported[ext](path, encoding="utf-8") if ext == ".txt" else supported[ext](path)
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = fname
            docs.extend(loaded)
            print(f"  [OK] {fname} ({len(loaded)} 页/段)")
        return docs

    def split(self, docs: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(docs)

    def build_index(self, docs: List[Document], index_path: str):
        print(f"  嵌入并建索引 ...")
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local(index_path)
        print(f"  [OK] 索引已保存至 {index_path}")

    def load_index(self, index_path: str) -> FAISS:
        return FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def ingest(self, doc_dir: str, index_path: str):
        print(f"加载文档: {doc_dir}")
        docs = self.load_documents(doc_dir)
        if not docs:
            print("  [ERR] 未找到支持的文档 (.pdf / .txt)")
            return False
        chunks = self.split(docs)
        print(f"  分割为 {len(chunks)} 个片段")
        self.build_index(chunks, index_path)
        return True


if __name__ == '__main__':
    ingestor = KnowledgeBaseIngestor()
    ingestor.ingest("knowledge", "knowledge/faiss_index")
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kb_ingest import KnowledgeBaseIngestor
from kb_qa import KnowledgeBaseQA


INDEX_PATH = "knowledge/faiss_index"
DOC_DIR = "knowledge"


def cmd_ingest():
    ingestor = KnowledgeBaseIngestor()
    ingestor.ingest(DOC_DIR, INDEX_PATH)


def cmd_ask(qa: KnowledgeBaseQA):
    print("知识库问答已就绪（输入 quit 退出）\n")
    while True:
        try:
            q = input("\n[提问] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("bye")
            break
        result = qa.ask(q)
        print(f"\n[回答] {result['answer']}\n")
        print(f"   [来源] {len(result['context'])} 个相关片段")
        for i, doc in enumerate(result['context']):
            src = doc.metadata.get("source", "未知")
            print(f"      {i+1}. {src} — {doc.page_content[:80]}...")
        print()


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python kb_cli.py ingest         导入知识库文档")
        print("  python kb_cli.py ask            进入问答交互")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "ask":
        qa = KnowledgeBaseQA(index_path=INDEX_PATH)
        cmd_ask(qa)
    else:
        print(f"未知命令: {cmd}")


if __name__ == '__main__':
    main()

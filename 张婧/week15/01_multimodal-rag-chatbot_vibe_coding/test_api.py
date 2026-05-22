import requests
import time
import os

BASE_URL = "http://localhost:8000/api"

def test_upload_and_chat():
    # 1. 上传测试 PDF（确保存在 test.pdf）
    test_pdf = "test.pdf"
    if not os.path.exists(test_pdf):
        print("请准备 test.pdf 文件")
        return

    with open(test_pdf, "rb") as f:
        resp = requests.post(f"{BASE_URL}/upload/document", files={"file": f})
    assert resp.status_code == 200
    file_id = resp.json()["file_id"]
    print(f"上传成功, file_id={file_id}")

    # 2. 等待 worker 解析完成（轮询 Milvus 或简单 sleep 60秒）
    print("等待 MinerU 解析和向量入库 (约60秒)...")
    time.sleep(60)

    # 3. 提问测试
    question = "根据图表，2019年哪种乳制品消费占比最高？"
    resp = requests.post(f"{BASE_URL}/chat", json={"question": question, "top_k": 3})
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    references = resp.json()["references"]
    print(f"答案: {answer}")
    print(f"引用: {references}")

    # 4. 简单评估（Jaccard 相似度，模拟 ground truth）
    ground_truth = "常温奶"
    pred_words = set(answer)
    true_words = set(ground_truth)
    jaccard = len(pred_words & true_words) / len(pred_words | true_words) if (pred_words | true_words) else 0
    print(f"答案内容相似度 (Jaccard): {jaccard:.2f} (满分0.5分，实际得分 {jaccard*0.5:.2f})")

if __name__ == "__main__":
    test_upload_and_chat()
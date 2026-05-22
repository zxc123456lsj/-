"""
文档处理 Worker
消费 Kafka 消息，解析 PDF，生成向量并存储
"""
import os
import glob
import subprocess
import traceback
from kafka import KafkaConsumer
import config
from models.file_model import update_file_state
from services.embedding_service import encode_chunk
from utils.milvus_client import insert_vectors


def split_text2chunks(lines: list, chunk_size: int = None) -> list:
    """
    将文本分割成多个 chunks

    Args:
        lines: 文本行列表
        chunk_size: 每个 chunk 的最大字符数

    Returns:
        chunks 列表
    """
    if chunk_size is None:
        chunk_size = config.RAG_CHUNK_SIZE

    chunks = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 跳过参考文献
        if line == "# References":
            continue

        # 跳过引用标记行
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        if len(chunks) == 0:
            chunks.append(line)
        else:
            if len(chunks[-1]) <= chunk_size:
                chunks[-1] += "\n" + line
            else:
                chunks.append(line)

    return chunks


def parse_document_with_mineru(file_path: str) -> str:
    """
    使用 MinerU 解析 PDF 文档

    Args:
        file_path: PDF 文件路径

    Returns:
        生成的 Markdown 文件路径
    """
    # 构建命令
    cmd = f"mineru -p {file_path} -o {config.PROCESSED_DIR} -b vlm-http-client -u {config.MINERU_ENDPOINT}"

    print(f"Executing: {cmd}")

    # 执行解析（最长等待10分钟）
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        print(f"MinerU error: {result.stderr}")
        raise Exception(f"MinerU parsing failed: {result.stderr}")

    # 查找生成的 Markdown 文件
    file_basename = os.path.basename(file_path).rsplit(".", 1)[0]
    pattern = os.path.join(config.PROCESSED_DIR, file_basename, "**", "*.md")
    md_files = glob.glob(pattern, recursive=True)

    if not md_files:
        raise Exception(f"No markdown file found after parsing {file_path}")

    return md_files[0]


def process_document(file_id: int, file_name: str, file_path: str):
    """
    处理单个文档

    Args:
        file_id: 文件ID
        file_name: 文件名
        file_path: 文件路径
    """
    print(f"\n{'='*60}")
    print(f"Processing document: {file_name} (ID: {file_id})")
    print(f"{'='*60}")

    try:
        # 1. 更新状态为"解析中"
        update_file_state(file_id, "解析中")

        # 2. 检查文件是否存在
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        # 3. 使用 MinerU 解析
        print("Step 1: Parsing document with MinerU...")
        markdown_path = parse_document_with_mineru(file_path)
        print(f"Markdown generated: {markdown_path}")

        # 4. 读取 Markdown 内容
        print("Step 2: Reading and chunking...")
        with open(markdown_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        chunks = split_text2chunks(lines)
        print(f"Created {len(chunks)} chunks")

        # 5. 编码并存储
        print("Step 3: Encoding and storing vectors...")
        inserted_count = 0

        for i, chunk in enumerate(chunks):
            try:
                # 编码
                text_bge_embedding, text_clip_embedding, image_clip_embedding = encode_chunk(
                    chunk, markdown_path
                )

                # 准备数据
                data = [{
                    "text_vector": text_bge_embedding,
                    "clip_text_vector": text_clip_embedding,
                    "clip_image_vector": image_clip_embedding,
                    "text": chunk,
                    "db_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path
                }]

                # 插入 Milvus
                insert_vectors(data)
                inserted_count += 1

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(chunks)}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing chunk {i}: {e}")
                continue

        print(f"Successfully inserted {inserted_count} chunks")

        # 6. 更新状态为"已完成"
        update_file_state(file_id, "已完成")
        print(f"Document processing completed: {file_name}")

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing document {file_name}: {e}")
        update_file_state(file_id, "失败")


def start_worker():
    """
    启动 Worker，消费 Kafka 消息
    """
    print("Starting Document Processor Worker...")
    print(f"Kafka bootstrap: {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {config.KAFKA_TOPIC_DOCUMENT}")
    print("Waiting for messages...")
    print("-" * 60)

    # 创建 Kafka Consumer
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC_DOCUMENT,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    # 消费消息
    for message in consumer:
        try:
            data = message.value
            print(f"\nReceived message: {data}")

            file_id = data.get('id')
            file_name = data.get('file_name')
            file_path = data.get('file_path')

            if not all([file_id, file_name, file_path]):
                print("Invalid message format, skipping...")
                continue

            # 处理文档
            process_document(file_id, file_name, file_path)

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing message: {e}")


if __name__ == "__main__":
    import json
    start_worker()

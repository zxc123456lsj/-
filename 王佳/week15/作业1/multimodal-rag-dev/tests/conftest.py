import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"
os.environ["KAFKA_TOPIC"] = "rag-data-test"
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_COLLECTION"] = "rag_data_test"
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="rag_test_")

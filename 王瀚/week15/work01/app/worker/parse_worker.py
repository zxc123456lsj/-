import json
import os
import time
import logging
import uuid
from typing import Optional

from app.config import settings
from app.core.database import Session
from app.core.models import Document, TextChunk, ImageRecord, ParseStatus
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.storage import StorageService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ParseWorker:
    def __init__(self):
        self.consumer: Optional = None
        self.running = False

    def _init_kafka(self):
        try:
            from confluent_kafka import Consumer
            self.consumer = Consumer({
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": settings.kafka_consumer_group,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            })
            self.consumer.subscribe([settings.kafka_parse_topic])
            logger.info("Kafka consumer subscribed to %s", settings.kafka_parse_topic)
        except ImportError:
            logger.warning("confluent-kafka not installed, using polling mode")
            self.consumer = None

    def process_document(self, doc_id: str, file_path: str):
        db = Session()
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                logger.error("Document %s not found in DB", doc_id)
                return

            doc.parse_status = ParseStatus.parsing
            db.commit()

            logger.info("Parsing document %s: %s", doc_id, file_path)

            result = DocumentParser.parse_with_pymupdf(file_path)

            if result.error:
                doc.parse_status = ParseStatus.failed
                doc.error_message = result.error
                db.commit()
                logger.error("Parse failed for %s: %s", doc_id, result.error)
                return

            md_path = StorageService.save_parsed_content(doc_id, result.markdown)
            doc.page_count = result.page_count

            chunks = DocumentParser.chunk_text(result.markdown)

            text_entities = []
            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                embedding = EmbeddingService.embed_text_bge(chunk["content"])
                text_record = TextChunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    kb_id=doc.kb_id,
                    content=chunk["content"],
                    page_num=chunk["page_num"],
                    chunk_index=idx,
                )
                db.add(text_record)
                text_entities.append({
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "kb_id": doc.kb_id,
                    "content": chunk["content"],
                    "page_num": chunk["page_num"],
                    "embedding": embedding,
                })

            image_entities = []
            for img_data in result.images:
                img_id = str(uuid.uuid4())
                embedding = EmbeddingService.embed_image_clip(img_data["file_path"])
                image_record = ImageRecord(
                    id=img_id,
                    doc_id=doc_id,
                    kb_id=doc.kb_id,
                    file_path=img_data["file_path"],
                    page_num=img_data["page_num"],
                    image_index=img_data["image_index"],
                )
                db.add(image_record)
                image_entities.append({
                    "id": img_id,
                    "image_id": img_id,
                    "doc_id": doc_id,
                    "kb_id": doc.kb_id,
                    "file_path": img_data["file_path"],
                    "page_num": img_data["page_num"],
                    "embedding": embedding,
                })

            if text_entities:
                VectorStore.insert_text_embeddings(text_entities)
            if image_entities:
                VectorStore.insert_image_embeddings(image_entities)

            doc.parse_status = ParseStatus.completed
            db.commit()
            logger.info("Document %s parsed successfully: %d chunks, %d images",
                        doc_id, len(chunks), len(result.images))

        except Exception as e:
            db.rollback()
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.parse_status = ParseStatus.failed
                doc.error_message = str(e)
                db.commit()
            logger.exception("Error processing document %s", doc_id)
        finally:
            db.close()

    def run_once(self, doc_id: str, file_path: str):
        self.process_document(doc_id, file_path)

    def run_forever(self):
        self._init_kafka()
        self.running = True
        logger.info("Parse worker started, waiting for messages...")

        while self.running:
            try:
                if self.consumer:
                    msg = self.consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        logger.error("Kafka error: %s", msg.error())
                        continue
                    data = json.loads(msg.value().decode("utf-8"))
                    self.process_document(data["doc_id"], data["file_path"])
                else:
                    time.sleep(5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception("Worker loop error: %s", e)

        if self.consumer:
            self.consumer.close()

    def stop(self):
        self.running = False

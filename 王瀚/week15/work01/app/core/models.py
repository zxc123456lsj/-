import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, DateTime, Enum, ForeignKey, JSON
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


def _utcnow():
    return datetime.now(timezone.utc)


class ParseStatus(str, enum.Enum):
    pending = "pending"
    parsing = "parsing"
    completed = "completed"
    failed = "failed"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    parse_status = Column(Enum(ParseStatus), default=ParseStatus.pending)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class TextChunk(Base):
    __tablename__ = "text_chunks"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    doc_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    page_num = Column(Integer, default=0)
    chunk_index = Column(Integer, default=0)
    extra_meta = Column(JSON, default=dict)


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    doc_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    page_num = Column(Integer, default=0)
    image_index = Column(Integer, default=0)
    caption = Column(Text, nullable=True)
    extra_meta = Column(JSON, default=dict)

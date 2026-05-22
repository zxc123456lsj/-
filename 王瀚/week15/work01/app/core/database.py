from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from app.config import settings
from app.core.models import Base


engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    pool_pre_ping=True,
)

SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Session = scoped_session(SessionFactory)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

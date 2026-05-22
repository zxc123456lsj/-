"""
文件数据模型 - SQLAlchemy ORM
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config

Base = declarative_base()


class File(Base):
    """
    文件表 - 存储上传文件的元信息
    """
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, comment='原始文件名')
    filepath = Column(String(1000), nullable=False, comment='本地存储路径')
    filestate = Column(String(20), nullable=False, default='已上传',
                       comment='处理状态：已上传/解析中/已完成/失败')
    created_at = Column(DateTime, default=datetime.now, comment='创建时间')
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now,
                        comment='更新时间')

    __table_args__ = (
        {'sqlite_autoincrement': True},
    )

    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'filename': self.filename,
            'filestate': self.filestate,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None,
        }


# 创建数据库引擎
engine = create_engine(f'sqlite:///{config.DB_PATH}')
Base.metadata.create_all(engine)

# 创建会话工厂
Session = sessionmaker(bind=engine)


def get_session():
    """获取数据库会话"""
    return Session()


def create_file_record(filename: str, filepath: str) -> int:
    """
    创建文件记录

    Args:
        filename: 原始文件名
        filepath: 存储路径

    Returns:
        file_id: 文件记录ID
    """
    with get_session() as session:
        file_record = File(
            filename=filename,
            filepath=filepath,
            filestate='已上传'
        )
        session.add(file_record)
        session.commit()
        session.refresh(file_record)
        return file_record.id


def update_file_state(file_id: int, state: str):
    """
    更新文件处理状态

    Args:
        file_id: 文件ID
        state: 新状态
    """
    with get_session() as session:
        file_record = session.query(File).filter_by(id=file_id).first()
        if file_record:
            file_record.filestate = state
            file_record.updated_at = datetime.now()
            session.commit()


def get_file_by_id(file_id: int) -> dict:
    """
    根据ID获取文件信息

    Args:
        file_id: 文件ID

    Returns:
        文件信息字典，不存在返回None
    """
    with get_session() as session:
        file_record = session.query(File).filter_by(id=file_id).first()
        return file_record.to_dict() if file_record else None


def list_files() -> list:
    """
    获取所有文件列表

    Returns:
        文件信息字典列表
    """
    with get_session() as session:
        files = session.query(File).order_by(File.created_at.desc()).all()
        return [f.to_dict() for f in files]


def delete_file(file_id: int) -> bool:
    """
    删除文件记录

    Args:
        file_id: 文件ID

    Returns:
        是否删除成功
    """
    with get_session() as session:
        file_record = session.query(File).filter_by(id=file_id).first()
        if file_record:
            session.delete(file_record)
            session.commit()
            return True
        return False

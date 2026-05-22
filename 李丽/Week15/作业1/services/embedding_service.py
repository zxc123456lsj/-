"""
向量编码服务
负责使用 BGE 和 CLIP 对文本和图像进行编码
"""
import os
import traceback
import numpy as np
from sentence_transformers import SentenceTransformer
import config

# 全局模型实例（延迟加载）
_bge_model = None
_clip_model = None


def get_bge_model():
    """获取 BGE 模型单例"""
    global _bge_model
    if _bge_model is None:
        print(f"Loading BGE model: {config.BGE_MODEL_PATH}")
        _bge_model = SentenceTransformer(config.BGE_MODEL_PATH)
        print("BGE model loaded successfully")
    return _bge_model


def get_clip_model():
    """获取 CLIP 模型单例"""
    global _clip_model
    if _clip_model is None:
        print(f"Loading CLIP model: {config.CLIP_MODEL_PATH}")
        _clip_model = SentenceTransformer(
            config.CLIP_MODEL_PATH,
            trust_remote_code=True,
            truncate_dim=config.CLIP_DIM
        )
        print("CLIP model loaded successfully")
    return _clip_model


def encode_text(text: str) -> list:
    """
    使用 BGE 编码文本

    Args:
        text: 输入文本

    Returns:
        512维向量
    """
    try:
        model = get_bge_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        traceback.print_exc()
        print(f"Error encoding text: {e}")
        return np.zeros(512).tolist()


def encode_text_with_clip(text: str) -> list:
    """
    使用 CLIP 编码文本

    Args:
        text: 输入文本

    Returns:
        1024维向量
    """
    try:
        model = get_clip_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        traceback.print_exc()
        print(f"Error encoding text with CLIP: {e}")
        return np.zeros(config.CLIP_DIM).tolist()


def encode_image(image_path: str) -> list:
    """
    使用 CLIP 编码图像

    Args:
        image_path: 图像文件路径

    Returns:
        1024维向量
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return np.zeros(config.CLIP_DIM).tolist()

        model = get_clip_model()
        embedding = model.encode(image_path, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        traceback.print_exc()
        print(f"Error encoding image {image_path}: {e}")
        return np.zeros(config.CLIP_DIM).tolist()


def encode_chunk(chunk: str, markdown_path: str = None) -> tuple:
    """
    对一个文本 chunk 进行多维度编码

    Args:
        chunk: 文本块
        markdown_path: Markdown 文件路径（用于定位关联图片）

    Returns:
        (bge_embedding, clip_text_embedding, clip_image_embedding)
    """
    # 分离文本和图片引用
    lines = chunk.split('\n')
    text_lines = [line for line in lines if not line.startswith('![')]
    image_lines = [line for line in lines if line.startswith('![')]

    text_content = '\n'.join(text_lines)

    # BGE 编码（纯文本）
    bge_embedding = encode_text(text_content)

    # CLIP 编码文本
    clip_text_embedding = encode_text_with_clip(text_content)

    # CLIP 编码图片（如果有）
    clip_image_embedding = None
    if image_lines and markdown_path:
        try:
            # 提取图片路径: ![alt](path)
            image_path_in_md = image_lines[0].split('](')[1].split(')')[0]
            image_filename = os.path.basename(image_path_in_md)
            image_dir = os.path.dirname(markdown_path)
            image_path = os.path.join(image_dir, image_filename)

            clip_image_embedding = encode_image(image_path)
        except Exception as e:
            print(f"Error extracting image path: {e}")
            clip_image_embedding = np.zeros(config.CLIP_DIM).tolist()
    else:
        clip_image_embedding = np.zeros(config.CLIP_DIM).tolist()

    return bge_embedding, clip_text_embedding, clip_image_embedding

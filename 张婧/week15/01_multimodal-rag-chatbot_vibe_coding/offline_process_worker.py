# 以下只展示关键修改部分，完整文件请参考原代码并应用 diff

def encode_document(path, file_id, file_name, file_path):
    lines = open(path, 'r', encoding='utf-8').readlines()
    chunks = split_text2chunks(lines)
    base_dir = os.path.dirname(path)  # 例如 ./processed/xxx/
    for chunk in chunks:
        # 将 chunk 中的图片链接替换为可访问的相对路径
        # 原图片路径如 images/abc.png ，实际位于 base_dir/images/abc.png
        # 改写为 ./processed/filename_dir/images/abc.png
        if "images/" in chunk:
            chunk = chunk.replace("images/", f"./processed/{os.path.basename(file_path).split('.')[0]}/images/")

        text_bge, text_clip, image_clip = encode_text_and_image(chunk, path)
        # 插入 Milvus（同原逻辑）
        ...
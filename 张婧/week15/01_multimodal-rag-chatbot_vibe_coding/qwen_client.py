import openai

client = openai.OpenAI(
    api_key="sk-711c186f74494136ba26035be25a7cb8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def qwen_vl_chat(question, context_texts, image_urls):
    # 构造消息内容（文本 + 图片）
    content = []
    # 添加上下文文本
    if context_texts:
        content.append({
            "type": "text",
            "text": f"基于以下资料回答问题：\n{chr(10).join(context_texts)}\n\n问题：{question}"
        })
    else:
        content.append({"type": "text", "text": question})

    # 添加图片（最多3张，避免超限）
    for img_url in image_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": img_url}})

    messages = [{"role": "user", "content": content}]

    completion = client.chat.completions.create(
        model="qwen-vl-plus",   # 多模态模型
        messages=messages,
        temperature=0.1
    )
    return completion.choices[0].message.content
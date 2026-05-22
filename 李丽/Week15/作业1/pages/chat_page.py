"""
聊天问答页面 - Streamlit
"""
import re
import os
import streamlit as st
from services.chat_service import chat


def render_markdown_with_images(markdown_text: str):
    """
    渲染 Markdown，处理图片显示
    """
    # 匹配 Markdown 图片语法 ![alt](url)
    pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    last_pos = 0
    for match in pattern.finditer(markdown_text):
        # 显示匹配前的文本
        if match.start() > last_pos:
            st.markdown(markdown_text[last_pos:match.start()], unsafe_allow_html=True)

        # 显示图片
        alt_text = match.group(1)
        img_url = match.group(2)

        # 如果是本地路径，转换为绝对路径
        if img_url.startswith('./') or img_url.startswith('../'):
            img_url = os.path.abspath(img_url)

        try:
            st.image(img_url, caption=alt_text if alt_text else None)
        except Exception as e:
            st.write(f"[图片加载失败: {img_url}]")

        last_pos = match.end()

    # 显示剩余文本
    if last_pos < len(markdown_text):
        st.markdown(markdown_text[last_pos:], unsafe_allow_html=True)


def clear_chat_history():
    """清空聊天记录"""
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是多模态 RAG 助手，可以帮你查询已上传文档的内容。"}
    ]


def show_chat_page():
    """显示聊天页面"""
    st.title("💬 多模态问答")

    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是多模态 RAG 助手，可以帮你查询已上传文档的内容。\n\n请先在上传页面添加 PDF 文档，然后在这里提问。"}
        ]

    # 侧边栏
    with st.sidebar:
        st.header("设置")
        st.button("🗑️ 清空对话", on_click=clear_chat_history)

        st.divider()
        st.header("使用说明")
        st.markdown("""
        1. 在**文件管理**页面上传 PDF 文档
        2. 等待文档解析完成（状态变为"已完成"）
        3. 在此页面输入问题进行提问
        4. 系统会基于文档内容回答
        """)

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_markdown_with_images(message["content"])
            else:
                st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 RAG 服务
        with st.chat_message("assistant"):
            with st.spinner("🔍 检索相关资料..."):
                response = chat(prompt)

            if response['code'] == 0:
                answer = response['data']['answer']
                sources = response['data']['sources']

                # 显示答案
                render_markdown_with_images(answer)

                # 显示来源（可折叠）
                if sources:
                    with st.expander("📚 参考来源"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**{i}. {source['file_name']}** (相关度: {source['score']})")
                            st.caption(source['text'])

                # 保存到历史
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"❌ 查询失败: {response.get('message', '未知错误')}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    show_chat_page()

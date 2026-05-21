import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api")

st.set_page_config(page_title="多模态RAG问答系统", page_icon="📚", layout="wide")

st.title("📚 多模态RAG问答系统")
st.caption("支持PDF文档解析、多模态向量检索、智能问答")

tab1, tab2, tab3 = st.tabs(["📤 文件上传", "💬 智能问答", "📋 文件管理"])

# ==================== Tab 1: 文件上传 ====================
with tab1:
    st.header("上传PDF文档")
    uploaded_file = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        help="支持单个PDF文件上传，最大50MB",
    )
    if uploaded_file:
        if st.button("上传并解析", type="primary"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                resp = httpx.post(f"{API_BASE}/upload", files=files, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ {data.get('message')}")
                    st.json(data)
                else:
                    st.error(f"上传失败: {resp.json().get('detail', '未知错误')}")
            except Exception as e:
                st.error(f"连接失败: {e}")

# ==================== Tab 2: 智能问答 ====================
with tab2:
    st.header("智能问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📎 参考来源"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"- **{src['file_name']}** (Chunk {src['chunk_index']}) "
                            f"— 相关度: {src['relevance_score']:.2f}\n"
                            f"  > {src['text'][:150]}..."
                        )

    if query := st.chat_input("输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成答案..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/chat",
                        json={"query": query, "top_k": 5},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(data["answer"])
                        sources = data.get("sources", [])
                        msg_data = {"role": "assistant", "content": data["answer"], "sources": sources}
                        with st.expander("📎 参考来源"):
                            for src in sources:
                                st.markdown(
                                    f"- **{src['file_name']}** (Chunk {src['chunk_index']}) "
                                    f"— 相关度: {src['relevance_score']:.2f}"
                                )
                                st.caption(src["text"][:200] + "...")
                        st.caption(f"⏱ 响应时间: {data['response_time_ms']}ms | 模型: {data['model']}")
                    else:
                        error_text = resp.json().get("detail", "请求失败")
                        st.error(error_text)
                        msg_data = {"role": "assistant", "content": f"错误: {error_text}"}
                except Exception as e:
                    st.error(f"连接失败: {e}")
                    msg_data = {"role": "assistant", "content": f"连接失败: {e}"}

        st.session_state.messages.append(msg_data)

# ==================== Tab 3: 文件管理 ====================
with tab3:
    st.header("文件列表")

    col1, col2 = st.columns([3, 1])
    with col1:
        status_filter = st.selectbox("状态筛选", ["全部", "pending", "processing", "completed", "failed"])
    with col2:
        page = st.number_input("页码", min_value=1, value=1)

    try:
        params: dict = {"page": page, "limit": 20}
        if status_filter != "全部":
            params["status"] = status_filter
        resp = httpx.get(f"{API_BASE}/files", params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            st.info(f"共 {data['total']} 个文件")

            for f in data["files"]:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(f["filename"])
                with col2:
                    status_map = {
                        "pending": "⏳ 等待",
                        "processing": "🔄 处理中",
                        "completed": "✅ 完成",
                        "failed": "❌ 失败",
                    }
                    st.text(status_map.get(f["status"], f["status"]))
                with col3:
                    st.caption(f["created_at"][:19] if f["created_at"] else "")
    except Exception as e:
        st.warning(f"无法连接到后端服务: {e}")

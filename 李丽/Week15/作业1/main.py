"""
多模态 RAG 聊天机器人 - 主入口
使用 Streamlit 构建 Web 界面
"""
import streamlit as st
from pages.upload_page import show_upload_page
from pages.chat_page import show_chat_page

# 页面配置
st.set_page_config(
    page_title="多模态 RAG 聊天机器人",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """主函数 - 页面路由"""

    # 侧边栏导航
    with st.sidebar:
        st.title("🤖 多模态 RAG")
        st.markdown("---")

        # 页面选择
        page = st.radio(
            "选择页面",
            options=["📁 文件管理", "💬 图文问答"],
            index=0
        )

        st.markdown("---")

        # 系统状态
        st.header("系统状态")

        # 检查各组件状态
        try:
            from utils.milvus_client import get_client
            get_client()
            st.success("✅ Milvus 连接正常")
        except Exception as e:
            st.error(f"❌ Milvus 连接失败: {e}")

        try:
            from models.file_model import get_session
            get_session()
            st.success("✅ SQLite 连接正常")
        except Exception as e:
            st.error(f"❌ SQLite 连接失败: {e}")

        st.markdown("---")
        st.caption("基于 BGE + CLIP + Qwen 的多模态 RAG 系统")

    # 根据选择显示页面
    if page == "📁 文件管理":
        show_upload_page()
    else:
        show_chat_page()


if __name__ == "__main__":
    main()

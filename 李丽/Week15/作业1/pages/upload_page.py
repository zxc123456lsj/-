"""
文件上传页面 - Streamlit
"""
import os
import streamlit as st
from models.file_model import list_files, delete_file as delete_file_record, get_session, File
from services.upload_service import handle_file_upload, delete_file_and_data
from utils.milvus_client import delete_by_file_id


def show_upload_page():
    """显示上传页面"""
    st.title("📁 文件管理")

    # ========== 文件列表 ==========
    st.header("已上传文件")

    files = list_files()

    if not files:
        st.info("暂无文件，请上传 PDF 文档")
    else:
        for file in files:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.write(f"**{file['filename']}**")

            with col2:
                # 根据状态显示不同颜色
                status = file['filestate']
                if status == "已完成":
                    st.success(status)
                elif status == "解析中":
                    st.info(status)
                elif status == "失败":
                    st.error(status)
                else:
                    st.write(status)

            with col3:
                st.write(file.get('created_at', ''))

            with col4:
                if st.button("删除", key=f"del_{file['id']}"):
                    result = delete_file_and_data(file['id'])
                    if result['code'] == 0:
                        st.success("删除成功")
                        st.rerun()
                    else:
                        st.error("删除失败")

    # ========== 文件上传 ==========
    st.header("上传新文件")

    uploaded_file = st.file_uploader(
        "选择 PDF 文件",
        type=["pdf"],
        help="支持 PDF 格式文档"
    )

    if uploaded_file is not None:
        st.write(f"文件名: {uploaded_file.name}")
        st.write(f"文件大小: {uploaded_file.size / 1024:.1f} KB")

        if st.button("确认上传", type="primary"):
            with st.spinner("上传中..."):
                # 读取文件内容
                file_content = uploaded_file.getvalue()

                # 处理上传
                result = handle_file_upload(file_content, uploaded_file.name)

                if result['code'] == 0:
                    st.success(f"✅ 上传成功！文件ID: {result['data']['file_id']}")
                    st.info("📋 文档正在后台解析，请稍后刷新查看状态")
                    st.rerun()
                else:
                    st.error(f"上传失败: {result.get('message', '未知错误')}")


if __name__ == "__main__":
    show_upload_page()

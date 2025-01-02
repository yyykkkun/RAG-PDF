import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from main import load_pdf_and_index, zhipu_glm_4_long, base_llm_chain, index

# 初始化模型
model = SentenceTransformer(r'F:\prompt-engineering-master\PDF文件理解\all-MiniLM-L6-v2')
dim = 384
index = faiss.IndexFlatL2(dim)

# 设置上传文件的保存目录
UPLOAD_DIRECTORY = "uploaded_pdfs"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# 初始化会话状态
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # 用于存储对话历史

if 'answered_queries' not in st.session_state:
    st.session_state.answered_queries = []  # 用于存储已回答的查询

# Streamlit 前端界面
st.title('PDF 内容分析与问答系统')

# 页面选择器
page = st.sidebar.selectbox("选择页面", ("对话页面", "历史记录页面"))

# ------------------------- 对话页面 -------------------------
if page == "对话页面":
    # 显示上传的PDF文件列表
    uploaded_files = os.listdir(UPLOAD_DIRECTORY)
    st.sidebar.header("已上传的 PDF 文件")

    # 显示已经上传的PDF文件并允许删除
    for pdf_file in uploaded_files:
        if st.sidebar.button(f"删除 {pdf_file}", key=pdf_file):
            os.remove(os.path.join(UPLOAD_DIRECTORY, pdf_file))
            st.sidebar.success(f"{pdf_file} 已删除！")
            st.rerun()  # 使用 st.rerun() 代替 st.experimental_rerun()

    # 文件上传组件，允许上传多个PDF文件
    uploaded_files = st.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=True)

    # 处理上传的PDF文件
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(UPLOAD_DIRECTORY, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("PDF 文件上传成功！")

    # 文件选择器：选择要提问的 PDF 文件
    selected_pdf = st.selectbox("选择你想提问的 PDF 文件", uploaded_files)

    # 加载所选 PDF 文件并建立 FAISS 索引
    pdf_texts = []
    if selected_pdf:
        file_path = os.path.join(UPLOAD_DIRECTORY, selected_pdf.name)
        pdf_texts += load_pdf_and_index(file_path)

    # 输入查询问题
    query = st.text_input("请输入您的问题:")

    if query:
        # 如果该问题已经回答过，则跳过
        if query in st.session_state.answered_queries:
            st.warning("这个问题已经被回答过了！")
        else:
            # 将用户问题转化为向量
            query_vector = model.encode([query])

            # 在FAISS中进行查询，获取最相似的文本段落
            _, indices = index.search(np.array(query_vector), k=3)  # 返回前3个相似的段落

            # 展示最相关的文本
            relevant_texts = [pdf_texts[i] for i in indices[0]]
            combined_text = "\n".join(relevant_texts)

            # 将用户问题和相关文本合并为提示词
            prompt = f"""
            用户的问题：{query}
            以下是与问题相关的内容：
            ----------------------------------------------------------
            {combined_text}
            ----------------------------------------------------------
            请根据上述信息，给出一个完整的回答：
            """

            # 获取智谱GLM-4模型
            llm = zhipu_glm_4_long(temperature=0.9)

            # 获取生成的回答
            response = base_llm_chain(llm, prompt)
            st.write("生成的回答：")
            st.write(response)

            # 记录已回答的问题，避免重复回答
            st.session_state.answered_queries.append(query)

            # 存储新的对话历史
            st.session_state.conversation_history.append(("用户: " + query, "AI: " + response))

# ------------------------- 历史记录页面 -------------------------
elif page == "历史记录页面":
    st.title("历史对话记录")

    # 显示历史记录
    if st.session_state.conversation_history:
        for entry in st.session_state.conversation_history:
            user_query, ai_response = entry
            st.markdown(f"**用户:** {user_query}")
            st.markdown(f"**AI:** {ai_response}")
    else:
        st.write("没有对话历史记录。")

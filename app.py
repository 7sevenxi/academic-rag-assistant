import streamlit as st
import ollama
import os
from rag_utils import multi_stage_retrieval, add_pdf_to_vector_db, init_vector_db, clear_vector_db

# 设置页面标题和图标
st.set_page_config(page_title="学术科研智能助手", page_icon="📚")

# 初始化向量数据库（确保应用启动时就完成）
if "vector_db" not in st.session_state:
    st.session_state.vector_db = init_vector_db()
    st.success("向量数据库初始化成功！")

# 页面标题和欢迎语
st.title("📚 学术科研智能助手")
st.subheader("基于RAG技术的文献解读与科研辅助工具")
st.write("—— 支持文献上传、科研问答 ——")

# 初始化对话历史（用session_state存储，实现多轮对话）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 显示历史对话
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户输入框
user_input = st.chat_input("请输入您的问题（如：这篇文献的研究方法是什么？）")


# 侧边栏：文献上传+向量库状态
with st.sidebar:
    st.header("📤 文献上传")
    uploaded_pdf = st.file_uploader("选择PDF文献", type="pdf", accept_multiple_files=False, key="pdf_uploader_unique")

    if uploaded_pdf:
        # 保存上传的PDF到临时路径
        pdf_save_path = f"./temp_{uploaded_pdf.name}"
        with open(pdf_save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        # 点击按钮入库
        if st.button("📥 上传并入库"):
            with st.spinner("正在解析文献并入库..."):
                success, msg = add_pdf_to_vector_db(pdf_save_path, st.session_state.vector_db)
                if success:
                    st.success(msg)
                    os.remove(pdf_save_path)  # 入库后删除临时文件
                else:
                    st.error(msg)

        # 显示向量库状态（原有功能，无修改）
        st.divider()
        st.info(f"向量库当前文献页数：{st.session_state.vector_db.count()}")

        # （原有向量库状态显示，注意：这里之前有重复，已保留1个）
        st.divider()
        st.info(f"向量库当前文献页数：{st.session_state.vector_db.count()}")

    # -------------------------- 新增：清空向量库按钮 --------------------------
    # 清空功能逻辑
    st.divider()
    st.warning("⚠️ 清空操作不可逆，请谨慎使用！")
    confirm_clear = st.checkbox("我已确认要清空所有文献数据")
    if confirm_clear:
        if st.button("🗑️ 清空当前向量库", type="primary"):
            with st.spinner("正在清空向量库..."):
                success, msg = clear_vector_db(st.session_state.vector_db)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    else:
        st.info("请勾选确认框以启用清空功能")

    # 显示向量库状态（移到侧边栏内部，缩进正确）
    st.divider()
    st.info(f"向量库当前文献页数：{st.session_state.vector_db.count()}")

# 当用户输入问题时，执行RAG流程（修改后完整代码）
if user_input:
    # 1. 添加用户消息到历史
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 2. 拼接历史对话上下文（新增：关联最近3轮对话）
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-3:]])

    # 3. 多轮RAG检索（替换原retrieve_relevant_documents）
    with st.spinner("检索相关文献..."):
        relevant_docs = multi_stage_retrieval(  # 这里替换为新增的多轮检索函数
            user_input,
            st.session_state.vector_db,
            top_k_coarse=10,
            top_k_final=3
        )

    # 4. 拼接上下文与问题，生成prompt（新增历史上下文）
    context = "\n".join(
        [f"[文献片段{idx + 1}] {doc['text']}（来源：{doc['metadata']['title']} 第{doc['metadata']['page_num']}页）"
         for idx, doc in enumerate(relevant_docs)])

    prompt = f"""基于以下文献信息，回答用户问题，需严格遵循以下规则：
    1. 仅能基于提供的文献片段回答，**文献中无相关信息时，直接说明“未从上传文献中找到相关信息”，禁止编造内容**；
    2. 学术严谨，标注文献引用（如[文献片段1]）；
    3. 语言简洁，逻辑清晰。

    提供的文献片段：
    {context}

    用户问题：{user_input}
    """

    # 5. 调用大模型生成回答（无修改）
    with st.spinner("生成回答..."):
        response = ollama.generate(
            model="deepseek-r1:1.5b",
            prompt=prompt,
            options={"temperature": 0.1}
        )
    assistant_msg = response["response"]

    # 6. 添加助手消息到历史并显示（无修改）
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
    st.chat_message("assistant").write(assistant_msg)
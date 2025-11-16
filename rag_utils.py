import os
import chromadb
import ollama
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# 加载环境变量（后续可存API密钥，现在为空也不影响）
load_dotenv()


# -------------------------- 1. 初始化向量数据库和嵌入模型 --------------------------
def init_vector_db():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # 使用Sentence-BERT嵌入模型
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # 轻量高效的开源模型
    )
    collection = chroma_client.get_or_create_collection(
        name="academic_papers",
        embedding_function=sentence_transformer_ef,
        metadata={"description": "存储学术文献的向量数据库"}
    )
    return collection


# -------------------------- 2. 解析PDF文献（提取文本和元数据） --------------------------
def parse_pdf(pdf_file_path):
    """解析PDF文件，提取标题、页码、文本内容（按页分割）"""
    reader = PdfReader(pdf_file_path)
    pdf_metadata = reader.metadata  # 获取PDF元数据（作者、标题等）
    documents = []

    # 按页提取文本，生成结构化文档
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:  # 跳过空白页
            documents.append({
                "text": page_text,
                "metadata": {
                    "file_name": os.path.basename(pdf_file_path),
                    "page_num": page_num,
                    "author": pdf_metadata.get("/Author", "未知"),
                    "title": pdf_metadata.get("/Title", "未知标题")
                }
            })
    return documents


# -------------------------- 3. 文献向量入库 --------------------------
def add_pdf_to_vector_db(pdf_file_path, collection):
    """将解析后的PDF文本向量化，存入Chroma向量库"""
    # 1. 解析PDF
    documents = parse_pdf(pdf_file_path)
    if not documents:
        return False, "PDF解析失败，未提取到文本"

    # 2. 准备入库数据（Chroma需要的格式：ids、documents、metadatas）
    ids = [f"{doc['metadata']['file_name']}_page{doc['metadata']['page_num']}" for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    # 3. 存入向量库
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    return True, f"成功入库！共{len(documents)}页文本"

# -------------------------- 新增：多轮RAG检索（粗检索→精排→上下文补全） --------------------------
def multi_stage_retrieval(query, collection, top_k_coarse=10, top_k_final=3):
    """多轮RAG检索：粗检索→精排→上下文补全（替代原retrieve_relevant_documents）"""
    # 1. 粗检索：召回Top10相关片段
    coarse_results = collection.query(
        query_texts=[query],
        n_results=top_k_coarse,
        include=["documents", "metadatas", "distances"]  # 包含相似度分数
    )
    coarse_docs = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            coarse_results["documents"][0],
            coarse_results["metadatas"][0],
            coarse_results["distances"][0]
        )
    ]

    # 2. 精排：用大模型对片段打分（0-10分，筛选≥6分的片段）
    ranked_docs = []
    for doc in coarse_docs:
        score_prompt = f"""
        请判断以下文献片段与用户问题的相关性，仅返回分数（0-10分，分数越高相关性越强），不要其他内容：
        用户问题：{query}
        文献片段：{doc['text'][:300]}...
        相关性分数：
        """
        response = ollama.generate(
            model="deepseek-r1:1.5b",
            prompt=score_prompt,
            options={"temperature": 0.1}
        )
        try:
            score = float(response["response"].strip())
            if score >= 6:
                ranked_docs.append({"doc": doc, "score": score})
        except:
            continue  # 分数解析失败则跳过

    # 3. 按分数排序，取Top3
    ranked_docs.sort(key=lambda x: x["score"], reverse=True)
    final_docs = [item["doc"] for item in ranked_docs[:top_k_final]]

    # 4. 上下文补全：补充相邻页码文本（若存在）
    final_docs_with_context = complete_adjacent_pages(final_docs, collection)
    return final_docs_with_context


def complete_adjacent_pages(selected_docs, collection):
    """补全选中片段的相邻页码文本（避免信息断裂）—— 辅助多轮检索函数"""
    completed_docs = []
    # 先获取所有文献的元数据和文本，用于内存筛选
    all_docs = collection.get(include=["documents", "metadatas"])
    all_metadatas = all_docs["metadatas"]
    all_documents = all_docs["documents"]

    for doc in selected_docs:
        file_name = doc["metadata"]["file_name"]
        current_page = doc["metadata"]["page_num"]
        # 查找前1页和后1页的文献
        for offset in [-1, 1]:
            target_page = current_page + offset
            if target_page < 1:
                continue
            # 内存中筛选符合文件名和目标页码的文献
            for meta, text in zip(all_metadatas, all_documents):
                if meta.get("file_name") == file_name and meta.get("page_num") == target_page:
                    doc["text"] += f"\n\n【补充上下文（第{target_page}页）】：{text[:200]}..."
                    break  # 找到后跳出循环
        completed_docs.append(doc)
    return completed_docs

# # -------------------------- 测试代码（运行该文件可验证功能） --------------------------
# if __name__ == "__main__":
#     # 1. 初始化向量库
#     db_collection = init_vector_db()
#     print("向量库初始化成功！")
#
#     # 2. 测试PDF解析和入库（替换为你本地的PDF文献路径）
#     test_pdf_path = "test_paper.pdf"  # 把你的测试文献放在项目文件夹，修改文件名
#     if os.path.exists(test_pdf_path):
#         success, msg = add_pdf_to_vector_db(test_pdf_path, db_collection)
#         print(msg)
#     else:
#         print(f"测试PDF文件不存在：{test_pdf_path}")
#
#     # 3. 验证入库结果（查询向量库中的文档数量）
#     print(f"向量库中当前文档总数：{db_collection.count()}")

# -------------------------- 新增：清空向量库功能 --------------------------
def clear_vector_db(collection):
    """清空向量库中所有文献数据（保留集合结构，可重新入库）"""
    try:
        # 获取向量库中所有文档的ID
        all_ids = collection.get()["ids"]
        if all_ids:  # 若存在数据则删除
            collection.delete(ids=all_ids)
            return True, f"成功清空向量库！共删除{len(all_ids)}条数据"
        else:
            return True, "向量库已为空，无需清空"
    except Exception as e:
        return False, f"清空失败：{str(e)}"
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# 环境变量配置，确保在生产环境中使用正确的API密钥
ZHIPUAI_API_KEY = 'ee7e8ba57eae412c933a4eac24619c74.RoCr0aGU0v4SdMkI'

# 初始化SentenceTransformer模型，用于文本向量化
model = SentenceTransformer('all-MiniLM-L6-v2')  # 你可以选择不同的模型
dim = 384  # 这个模型的向量维度
index = faiss.IndexFlatL2(dim)  # 使用L2距离的FAISS索引


# 加载和处理PDF文件
def pdf_loader(filepath):
    """
    加载PDF文件内容并提取文本。
    :param filepath: PDF文件路径
    :return: 返回提取的文本内容
    """
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    return [page.page_content for page in pages]


# 将PDF文件内容加载并转化为向量，存储到FAISS向量数据库中
def load_pdf_and_index(filepath):
    """
    加载PDF并将其内容转化为向量，存入FAISS索引。
    :param filepath: PDF文件路径
    :return: 返回PDF的文本内容列表
    """
    pages_content = pdf_loader(filepath)
    embeddings = model.encode(pages_content)
    index.add(np.array(embeddings))  # 将文本向量加入FAISS索引
    return pages_content


# 初始化智谱GLM-4模型
def zhipu_glm_4_long(temperature=0.9):
    """
    获取智谱的GLM-4长文本模型。
    :param temperature: 模型的温度
    :return: ChatOpenAI 实例
    """
    model = ChatOpenAI(temperature=temperature, model="glm-4-long",
                       openai_api_key=ZHIPUAI_API_KEY,
                       openai_api_base="https://open.bigmodel.cn/api/paas/v4/")
    return model


# 基于LLM链生成对话式回答
def base_llm_chain(model, prompt, **kwargs):
    """
    创建并运行LLM链，生成模型的响应。
    :param model: 使用的LLM模型
    :param prompt: 模板中的prompt
    :param kwargs: 提供给模板的参数
    :return: 模型的响应结果
    """
    prompt_template = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=model, prompt=prompt_template)
    result = chain.run(kwargs)
    return result


# 在FAISS索引中查询与用户问题最相似的文本段落
def search_similar_texts(query, k=3):
    """
    根据用户的查询问题，在FAISS索引中查找最相似的文本。
    :param query: 用户的查询问题
    :param k: 返回最相似的k个段落
    :return: 返回最相关的文本段落
    """
    query_vector = model.encode([query])  # 将查询问题转化为向量
    _, indices = index.search(np.array(query_vector), k=k)  # 查询FAISS索引
    return indices[0]  # 返回最相似的k个文本段落的索引


# 示例函数：从PDF文件加载并根据查询生成回答
def get_answer_from_pdf(pdf_filepath, query):
    """
    从PDF文件加载内容并回答问题。
    :param pdf_filepath: PDF文件路径
    :param query: 用户的查询问题
    :return: 返回生成的回答
    """
    # 加载PDF并建立FAISS索引
    load_pdf_and_index(pdf_filepath)

    # 根据查询问题，查询FAISS索引并获取相关文本
    relevant_indices = search_similar_texts(query, k=3)

    # 提取相关的文本内容
    relevant_texts = []
    for i in relevant_indices:
        relevant_texts.append(pdf_loader(pdf_filepath)[i])

    # 格式化对话样式的prompt
    prompt = f"""
    给你提供 PDF 电子书内容，请认真阅读并理解每一页的核心内容，并回答问题，以下是电子书的每一页内容：
    ----------------------------------------------------------
    问题：总结这部分内容推荐的日常行为原则，以及它们的使用场景。

    请以对话的形式回复，每一段内容后都要有一个简短、自然的对话式回答，避免直接列举。示例回答可以像这样：
    用户：根据这本书，推荐的行为原则是什么？
    AI：书中提到的一个行为原则是：保持每日反思。使用场景：在忙碌的工作日结束后，每晚花些时间思考今天做得如何。

    请开始你的回答：
    """

    # 获取智谱GLM-4模型
    llm = zhipu_glm_4_long(temperature=0.9)

    # 获取生成的回答
    response = base_llm_chain(llm, prompt)
    return response
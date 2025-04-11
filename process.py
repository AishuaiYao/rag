from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unsloth import FastLanguageModel
import torch
import re

# ----------------------
# 1. 智能分块预处理
# ----------------------
class SmartDocumentProcessor:
    def __init__(self):
        # 初始化嵌入模型，使用HuggingFace的BAAI/bge-small-zh-v1.5模型-这个模型专为RAG而生

        # 模型下载
        # from modelscope import snapshot_download
        # model_dir = snapshot_download('BAAI/bge-small-zh-v1.5')
        # import os
        # os.environ["HF_HOME"] = r"E:\3-软件\PyCharm Community Edition 2024.3.5\PythonProject\RAG"  # 这里是包含模型的目录路径
        self.embed_model = HuggingFaceEmbeddings(
            model_name="./BAAI/x",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 16}
        )

        # import torch
        # from transformers import AutoModelForSequenceClassification, AutoTokenizer
        #
        # tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        # model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        # model.eval()

    def _detect_content_type(self, text):
        """动态内容类型检测"""
        # 如果文本包含代码相关模式（如def、import、print或代码示例）标记为代码
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text:  # 如果文本包含表格相关模式（如|和百分比），标记为表格
            return "table"
        return "normal"  # 如果不满足上述条件，标记为普通文本

    def process_documents(self):
        # 加载文档
        # 创建加载器列表，处理知识库中的PDF和文本文件
        loaders = [
            DirectoryLoader("./dataset", glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader("./dataset", glob="**/*.txt", loader_cls=lambda x: TextLoader(x, encoding='utf-8'))
        ]
        # 初始化空列表，用于存储加载的所有文档
        documents = []
        # 遍历每个加载器，加载文档并添加到documents列表
        for loader in loaders:
            documents.extend(loader.load())

        # 创建语义分块器，使用嵌入模型进行语义分块
        chunker = SemanticChunker(
            embeddings=self.embed_model,  # 使用我们的嵌入模型
            breakpoint_threshold_amount=82,  # 设置断点阈值
            add_start_index=True  # 启用添加起始索引功能
        )
        base_chunks = chunker.split_documents(documents)  # 使用语义分块器将文档分割为基本块

        # 二次动态分块
        # 初始化最终分块列表，用于存储二次分块结果
        final_chunks = []
        # 遍历每个基本块，进行二次动态分块
        for chunk in base_chunks:
            content_type = self._detect_content_type(chunk.page_content)
            if content_type == "code":
                # 如果是代码，设置较小的块大小和重叠，用于保持上下文
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=256, chunk_overlap=64)
            elif content_type == "table":
                # 如果是表格，设置中等块大小和重叠
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=384, chunk_overlap=96)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512, chunk_overlap=128)
                # 如果是普通文本，设置较大的块大小和重叠
            final_chunks.extend(splitter.split_documents([chunk]))
            # 使用适当的分割器将块分割为最终块，并添加到列表
        # 遍历最终块列表，为每个块添加元数据
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "content_type": self._detect_content_type(chunk.page_content)
            })  # 更新块的元数据，添加唯一ID和内容类型

        return final_chunks


# ----------------------
# 2. 混合检索系统
# ----------------------
class HybridRetriever:
    def __init__(self, chunks):
        # 创建向量数据库，使用Chroma存储文档块，嵌入模型为BAAI/bge-large-zh-v1.5
        self.vector_db = Chroma.from_documents(
            chunks,
            embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5"),
            persist_directory="./vector_db"
        )

        # 创建BM25检索器，从文档块中初始化，初始检索数量为5
        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=5  # 初始检索数量多于最终需要
        )

        # 创建混合检索器，结合向量和BM25检索，权重分别为0.6和0.4
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )

        # 初始化重排序模型，使用BAAI/bge-reranker-large
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-large",
            device="cuda" if torch.has_cuda else "cpu"
        )

    def retrieve(self, query, top_k=3):
        # 第一阶段：使用混合检索器获取相关文档
        docs = self.ensemble_retriever.get_relevant_documents(query)

        # 第二阶段：为查询和每个文档创建配对，用于重排序
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        # 使用重排序模型预测配对的分数
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # 返回top_k结果
        return [doc for doc, _ in ranked_docs[:top_k]]


# ----------------------
# 3. RAG系统集成
# ----------------------
class EnhancedRAG:
    def __init__(self):
        # 文档处理
        processor = SmartDocumentProcessor()
        chunks = processor.process_documents()  # 整合检索和生成功能

        # 初始化混合检索器，使用处理后的分块
        self.retriever = HybridRetriever(chunks)

        # 加载微调后的语言模型，用于生成回答
        # 我使用DeepSeek-R1-Distill-Qwen-14B在知乎推理数据集上进行微调
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            "DeepSeek-R1-Distill-Qwen-7B",
            max_seq_length=4096
        )

        # 设置随机种子
        torch.manual_seed(666)

        # 将模型设置为推理模式
        FastLanguageModel.for_inference(self.model)

    def generate_prompt(self, question, contexts):
        # 格式化上下文，包含来源和类型信息
        context_str = "\n\n".join([
            f"[来源：{doc.metadata['source']}，类型：{doc.metadata['content_type']}]\n{doc.page_content}"
            for doc in contexts
        ])
        # 创建提示模板，要求基于上下文回答问题
        return f"""你是一个专业助手，请严格根据以下带来源的上下文：
        {context_str}

        按步骤思考并回答：{question}
        如果上下文信息不足，请明确指出缺失的信息。最后用中文给出结构化答案。"""

    def ask(self, question):
        # 使用检索器获取与问题相关的上下文
        contexts = self.retriever.retrieve(question)

        # 根据问题和上下文生成提示
        prompt = self.generate_prompt(question, contexts)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # 使用语言模型生成回答
        generated_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=2048,
            temperature=0.3,
            top_p=0.9,

            do_sample=True
        )
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = {'choices': [{'text': generated_text}]}
        return response['choices'][0]['text']



rag = EnhancedRAG()

complex_question = "给我讲讲沙漠中的彼岸花"
answer = rag.ask(complex_question)
print(f"问题：{complex_question}")
print("答案：")
print(answer)















































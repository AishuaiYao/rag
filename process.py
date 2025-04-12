from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unsloth import FastLanguageModel
import torch
import os
import re
from tqdm import tqdm

# ----------------------
# 1. 智能分块预处理
# ----------------------
class SmartDocumentProcessor:
    def __init__(self, file_name):
        self.file_name = file_name
        print("正在初始化嵌入模型...")
        self.embed_model = HuggingFaceEmbeddings(
            model_name=r"models/BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 16}
        )
        print("嵌入模型初始化完成。")

    def _detect_content_type(self, text):
        """动态内容类型检测"""
        # 如果文本包含代码相关模式（如def、import、print或代码示例）标记为代码
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text:  # 如果文本包含表格相关模式（如|和百分比），标记为表格
            return "table"
        return "normal"  # 如果不满足上述条件，标记为普通文本

    def process_documents(self):
        print("开始加载文档...")
        # 创建加载器列表，处理知识库中的PDF和文本文件
        loaders = [
            DirectoryLoader("./dataset", glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader("./dataset", glob=f"**/{self.file_name}", loader_cls=lambda x: TextLoader(x, encoding='utf-8'))
        ]
        # 初始化空列表，用于存储加载的所有文档
        documents = []
        # 遍历每个加载器，加载文档并添加到documents列表
        for loader in loaders:
            documents.extend(loader.load())
        print("文档加载完成。")

        print("开始进行语义分块...")
        # 创建语义分块器，使用嵌入模型进行语义分块
        chunker = SemanticChunker(
            embeddings=self.embed_model,  # 使用我们的嵌入模型
            breakpoint_threshold_amount=82,  # 设置断点阈值
            add_start_index=True  # 启用添加起始索引功能
        )
        # 手动处理每个文档并添加进度条
        base_chunks = []
        for doc in tqdm(documents, desc="语义分块进度"):
            base_chunks.extend(chunker.split_documents([doc]))
        print("语义分块完成。")

        print("开始进行二次动态分块...")
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
        print("二次动态分块完成。")

        print("开始为每个块添加元数据...")
        # 遍历最终块列表，为每个块添加元数据
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "content_type": self._detect_content_type(chunk.page_content)
            })  # 更新块的元数据，添加唯一ID和内容类型
        print("元数据添加完成。")

        return final_chunks


# ----------------------
# 2. 混合检索系统
# ----------------------
class HybridRetriever:
    def __init__(self, file_name, chunks):
        embedding = HuggingFaceEmbeddings(model_name="models/BAAI/bge-large-zh-v1.5")
        persist_directory = "./vector_db/"+file_name.replace('.txt', '')
        os.makedirs(persist_directory, exist_ok=True)
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print("发现本地已存在向量数据库，正在加载...")
            self.vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            print("本地向量数据库加载完成。")
        else:
            print("未发现本地向量数据库，开始创建...")
            # 手动添加文档并展示进度条
            self.vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            for chunk in tqdm(chunks, desc="向量数据库创建进度"):
                self.vector_db.add_documents([chunk])
            print("向量数据库创建完成。")
        print("开始创建BM25检索器...")
        # 创建BM25检索器，从文档块中初始化，初始检索数量为5
        self.bm25_retriever = BM25Retriever.from_documents(
            chunks,
            k=5  # 初始检索数量多于最终需要
        )
        print("BM25检索器创建完成。")

        print("开始创建混合检索器...")
        # 创建混合检索器，结合向量和BM25检索，权重分别为0.6和0.4
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )
        print("混合检索器创建完成。")

        print("开始初始化重排序模型...")
        self.reranker = CrossEncoder(
            "models/BAAI/bge-reranker-large",
            # device="cuda" if torch.has_cuda else "cpu"
            device="cpu"
        )
        print("重排序模型初始化完成。")

    def retrieve(self, query, top_k=3):
        print("开始使用混合检索器获取相关文档...")
        # 第一阶段：使用混合检索器获取相关文档
        docs = self.ensemble_retriever.get_relevant_documents(query)
        print("混合检索器获取相关文档完成。")

        print("开始使用重排序模型进行重排序...")
        # 第二阶段：为查询和每个文档创建配对，用于重排序
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        # 使用重排序模型预测配对的分数
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        print("重排序完成。")

        # 返回top_k结果
        return [doc for doc, _ in ranked_docs[:top_k]]


# ----------------------
# 3. RAG系统集成
# ----------------------
class EnhancedRAG:
    def __init__(self, file_name):
        print("开始进行文档处理...")
        processor = SmartDocumentProcessor(file_name)
        chunks = processor.process_documents()  # 整合检索和生成功能
        print("文档处理完成。")

        print("开始初始化混合检索器...")
        # 初始化混合检索器，使用处理后的分块
        self.retriever = HybridRetriever(file_name, chunks)
        print("混合检索器初始化完成。")

        print("开始加载微调后的语言模型...")
        device_map = {
            "model.embeddings": 0,
            "model.layers": 0,
            "model.norm": 0,
            "lm_head": 0,
            "model.embed_tokens.weight":0
        }
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            "models/DeepSeek-R1-Distill-Qwen-7B",
            max_seq_length=4096,
            offload_buffers = True,
            # llm_int8_enable_fp32_cpu_offload = True,
            device_map = device_map
        )
        print("微调后的语言模型加载完成。")

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
        return f">你是一个专业助手，请严格根据以下来源的上下文:\n  {context_str} \n>按步骤思考并回答：{question}\n>如果上下文信息不足，请明确指出缺失的信息。最后用中文给出结构化答案。"

    def ask(self, question):
        print("开始使用检索器获取与问题相关的上下文...")
        contexts = self.retriever.retrieve(question)
        print("与问题相关的上下文获取完成。")

        print("开始根据问题和上下文生成提示...")
        # 根据问题和上下文生成提示
        prompt = self.generate_prompt(question, contexts)
        print("提示生成完成。")

        print("开始使用语言模型生成回答...")
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
        print("语言模型回答生成完成。")
        return response['choices'][0]['text']


if __name__ == '__main__':
    book_name = '巴伐利亚玫瑰.txt'
    print("开始初始化RAG系统...")
    rag = EnhancedRAG(book_name)
    print("RAG系统初始化完成。")

    complex_question = "汉诺威柏林的高速公路上发生了什么"
    print(f"开始处理问题：{complex_question}")
    answer = rag.ask(complex_question)
    print("问题处理完成。")
    print(f"[问题：{complex_question}]")
    print(">答案：")
    print(answer)
    print("问题2")
    complex_question = "伊丽莎没穿越之前多大了，不是问伊丽莎白"
    print(f"开始处理问题：{complex_question}")
    answer = rag.ask(complex_question)
    print("问题处理完成。")
    print(f"[问题：{complex_question}]")
    print(">答案：")
    print(answer)
# rag-demo - LangChain RAG 文档问答

基于 **LangChain** 的完整 RAG（检索增强生成）系统，演示从 PDF 文档到智能问答的完整流程。

## 📝 项目简介

本项目演示如何使用 LangChain 构建生产级 RAG 系统，包含：

- ✅ **PDF 文档处理**: 自动分块，保持语义完整性
- ✅ **向量索引构建**: ChromaDB 持久化存储
- ✅ **智能问答**: LCEL (LangChain Expression Language) 链式调用
- ✅ **多查询检索**: Multi-Query 提高召回率
- ✅ **ai-starter 集成**: 使用工厂类简化配置

## 🏗️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **框架** | LangChain 0.3+ | LLM 应用开发框架 |
| **向量数据库** | ChromaDB | 嵌入式向量数据库 |
| **LLM** | 智谱AI glm-4-flash | 对话生成 |
| **Embedding** | 智谱AI embedding-2 | 文本向量化（1024维） |
| **文档处理** | LangChain Text Splitters | PDF 分块 |
| **工具库** | ai-starter | 配置、日志、LangChain 集成 |

## 📁 项目结构

```
rag-demo/
├── rag_demo/
│   ├── retriever/
│   │   └── langchain_qa_retriever.py    # RAG 问答检索器（LCEL）
│   ├── stage_test/                      # 阶段测试（单元测试）
│   │   ├── pdf_chunker_test.py          # PDF 分块测试
│   │   ├── embedding_test.py            # Embedding 测试
│   │   ├── vector_storage_test.py       # 向量存储测试
│   │   └── qa_retriever_test.py         # 问答检索测试
│   └── workflow_test/                   # 完整流程测试
│       ├── test_rag_pipeline.py         # RAG 完整流程
│       ├── test_multi_query_traditional.py  # Multi-Query（传统方式）
│       └── test_multi_query_lcel.py     # Multi-Query（LCEL 方式）
├── data/                                # PDF 文档目录
├── chroma_db/                           # ChromaDB 持久化数据（自动生成）
├── config.yaml                          # 配置文件（gitignore）
├── pyproject.toml
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd rag-demo
uv sync
```

### 2. 配置

创建 `config.yaml`：

```yaml
# ==================== RAG 配置 ====================
rag:
  # PDF 文件路径（相对于 rag_demo 目录）
  pdf_path: "项目经理资格考试题库.pdf"

  # 文档分块配置
  chunk_size: 500          # 文本块大小
  chunk_overlap: 50        # 文本块重叠大小

  # 检索配置
  top_k: 3                 # 检索返回的文档数量

# ==================== 智谱AI 配置 ====================
zhipu:
  # API Key（必填）
  api_key: "your_api_key_here"

  # API Base URL（可选）
  api_base: "https://open.bigmodel.cn/api/paas/v4/"

  # LLM 配置
  llm:
    model: "glm-4-flash"   # 模型名称
    temperature: 0.7       # 温度参数

  # Embedding 配置
  embedding:
    model: "embedding-2"   # embedding-2 或 embedding-3
    dimension: 1024        # 向量维度

# ==================== ChromaDB 配置 ====================
database:
  chromadb:
    collection_name: "rag_demo_collection"  # Collection 名称
    persist_directory: "./chroma_db"        # 持久化目录

# ==================== HTTP 客户端配置 ====================
http:
  proxy:
    enabled: true
    http: "http://proxy:8080"
  verify_ssl: false

# ==================== 日志配置 ====================
logging:
  level: "INFO"
```

### 3. 准备文档

将 PDF 文档放入 `rag_demo/` 目录（或修改 `config.yaml` 中的路径）：

```bash
rag-demo/
└── rag_demo/
    └── 项目经理资格考试题库.pdf
```

### 4. 运行

#### 方式1: 完整 RAG 流程

```bash
python rag_demo/workflow_test/test_rag_pipeline.py
```

#### 方式2: Multi-Query 检索（LCEL）

```bash
python rag_demo/workflow_test/test_multi_query_lcel.py
```

#### 方式3: 阶段测试（分步测试）

```bash
# 测试 PDF 分块
python rag_demo/stage_test/pdf_chunker_test.py

# 测试 Embedding
python rag_demo/stage_test/embedding_test.py

# 测试向量存储
python rag_demo/stage_test/vector_storage_test.py

# 测试问答检索
python rag_demo/stage_test/qa_retriever_test.py
```

## 💻 核心代码说明

### RAGPipeline 类（完整流程）

```python
from rag_demo.workflow_test.test_rag_pipeline import RAGPipeline

# 初始化（自动从 config.yaml 读取配置）
pipeline = RAGPipeline()

# 构建知识库（首次运行或文档更新后）
pipeline.build_knowledge_base()

# 提问
response = pipeline.ask("项目经理的主要职责有哪些？")
print(f"答案: {response['result']}")
print(f"来源: {len(response['source_documents'])} 个文档")
```

### 完整流程

```
PDF 文档
    ↓
[PDFChunker: 文档分块]
    ↓
chunks: [chunk1, chunk2, ...]  (500 tokens, overlap 50)
    ↓
[LangChainGLMEmbedding: 文本向量化]
    ↓
vectors: [[0.12, -0.34, ...], ...]  (1024维)
    ↓
[LangchainChromadb: 存储到向量数据库]
    ↓
持久化到 chroma_db/

--- 查询阶段 ---

用户问题: "项目经理的主要职责有哪些？"
    ↓
[Retriever: 向量相似度检索]
    ↓
Top-K 相关文档 (k=3)
    ↓
[LCEL Chain: 构建 Prompt]
    ↓
Prompt: "上下文: [文档1]... [文档2]... [文档3]...\n问题: ..."
    ↓
[LangChainChatZhipuAI: 生成答案]
    ↓
答案 + 来源文档
```

## 🎯 核心组件

### 1. PDFChunker（文档分块）

```python
from ai_starter.langchain import PDFChunker

chunker = PDFChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.load_and_split("document.pdf")

# 每个 chunk 是 LangChain Document 对象
# - page_content: 文本内容
# - metadata: {"source": "document.pdf", "page": 1}
```

**特点**:
- RecursiveCharacterTextSplitter - 递归切分，保持语义完整
- 保留页码等元数据
- 可配置 chunk_size 和 overlap

### 2. LangchainChromadb（向量存储）

```python
from ai_starter.langchain import LangchainChromadb, LangChainGLMEmbedding

embeddings = LangChainGLMEmbedding()
storage = LangchainChromadb(embeddings=embeddings)

# 添加文档（自动向量化）
storage.add_texts(chunks)

# 获取检索器
retriever = storage.get_retriever(k=3)

# 清空 collection
storage.clear_collection()
```

**特点**:
- 自动持久化到本地文件
- 支持增量添加
- 集成 Embedding 自动处理

### 3. LangchainQARetriever（问答检索）

使用 **LCEL (LangChain Expression Language)** 构建 RAG 链：

```python
from rag_demo.retriever.langchain_qa_retriever import LangchainQARetriever

qa_retriever = LangchainQARetriever(retriever)
response = qa_retriever.ask("问题")

# response = {
#     "result": "答案文本",
#     "source_documents": [doc1, doc2, ...]
# }
```

**LCEL 链定义**:
```python
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)
```

## 🔍 高级特性

### Multi-Query 检索

生成多个查询变体提高召回率：

```python
# 传统方式（使用 MultiQueryRetriever）
from rag_demo.workflow_test.test_multi_query_traditional import MultiQueryRAG

rag = MultiQueryRAG()
rag.build_knowledge_base()
response = rag.ask("问题")

# LCEL 方式（使用 LCEL + MultiQueryRetriever）
from rag_demo.workflow_test.test_multi_query_lcel import MultiQueryLCEL

rag = MultiQueryLCEL()
rag.build_knowledge_base()
response = rag.ask("问题")
```

**原理**:
```
用户问题: "项目经理的职责有哪些？"
    ↓
[LLM 生成多个查询变体]
    ↓
变体1: "项目经理的职责有哪些？"
变体2: "项目经理需要做什么工作？"
变体3: "项目经理的工作内容包括哪些方面？"
    ↓
[并行检索] → [合并去重] → [返回文档]
```

## 📊 运行输出示例

```
============================================================
开始构建知识库
============================================================

[Step 0] 清空现有数据
✓ 数据清空完成

[Step 1] PDF 文本分割
✓ 分割完成: 234 个文本块

[Step 2+3] 文本向量化并存储到向量数据库
✓ 存储完成: 234 个文档

[Step 4] 初始化问答检索器
✓ 问答系统初始化完成

============================================================
知识库构建完成！
============================================================

============================================================
开始测试问答系统
============================================================

问题: 项目经理的主要职责有哪些？

[检索到 3 个相关文档]

答案:
项目经理的主要职责包括：
1. 项目计划和组织
2. 团队管理和协调
3. 风险识别和控制
4. 进度跟踪和汇报
5. 资源分配和优化

来源文档:
  [1] 项目经理资格考试题库.pdf (第 12 页)
  [2] 项目经理资格考试题库.pdf (第 15 页)
  [3] 项目经理资格考试题库.pdf (第 23 页)
```

## 📚 与其他项目对比

| 特性 | rag-demo | llama-demo | chromadb-demo |
|------|----------|------------|---------------|
| **框架** | LangChain | LlamaIndex | Pure Python |
| **向量DB** | ChromaDB | Qdrant | ChromaDB |
| **检索策略** | Multi-Query | RAG Fusion + Rerank | 基础检索 |
| **问答链** | LCEL Chain | Chat Engine | ❌ |
| **复杂度** | ⭐⭐⭐ 完整 | ⭐⭐⭐⭐ 高级 | ⭐ 简单 |
| **文档处理** | ✅ PDF 分块 | ✅ PDF 分块 | ❌ |
| **适合人群** | LangChain 开发者 | LlamaIndex 开发者 | 初学者 |

**选择建议**:
- 学习向量数据库基础 → **chromadb-demo**
- 学习 LangChain RAG → **rag-demo**
- 学习 LlamaIndex 高级 RAG → **llama-demo**

## 🔧 配置调优

### 文档分块参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `chunk_size` | 500 | 文本块大小 | 增大 → 更多上下文，但检索精度下降 |
| `chunk_overlap` | 50 | 文本块重叠 | 10-20% 的 chunk_size，避免语义断裂 |

### 检索参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `top_k` | 3 | 检索返回数量 | 增大 → 召回率提高，但上下文过长 |

### LLM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.7 | 温度参数（0-1） |
| `model` | glm-4-flash | 模型名称 |

## 📦 依赖

```toml
dependencies = [
    "ai-starter[langchain,chromadb,embedding]",  # LangChain + ChromaDB + Embedding
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "pypdf>=5.1.0",                               # PDF 解析
]
```

**ai-starter 提供**:
- `PDFChunker` - PDF 文档分块
- `LangChainGLMEmbedding` - LangChain Embedding 实现
- `LangchainChromadb` - LangChain ChromaDB 适配器
- `LangChainChatZhipuAI` - LangChain LLM 实现
- `Config` - 配置管理
- `get_logger()` - 日志工具

## 🛠️ 常见问题

### 1. ChromaDB 数据如何清空？

```python
storage = LangchainChromadb(embeddings=embeddings)
storage.clear_collection()  # 清空当前 collection
```

或删除 `chroma_db/` 目录：
```bash
rm -rf chroma_db/
```

### 2. 如何切换不同的 PDF 文档？

修改 `config.yaml`:
```yaml
rag:
  pdf_path: "new_document.pdf"  # 更改路径
```

然后重新构建知识库：
```python
pipeline.build_knowledge_base()
```

### 3. 答案不准确怎么办？

尝试：
1. **增加检索数量**: `top_k: 5`
2. **调整分块大小**: `chunk_size: 1000`
3. **降低温度**: `temperature: 0.3`（更确定性）
4. **使用 Multi-Query**: 提高召回率

## 📖 参考资源

### 官方文档
- [LangChain 文档](https://python.langchain.com/)
- [LangChain RAG 教程](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain LCEL 指南](https://python.langchain.com/docs/expression_language/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [智谱AI 开放平台](https://open.bigmodel.cn/)

### 项目内文档
- [ai-starter/README.md](../ai-starter/README.md) - 工具库详细说明
- [总结.md](../总结.md) - AI 框架对比分析
- [编码规范.md](../编码规范.md) - Python 编码标准

---

**最后更新**: 2026-03-04

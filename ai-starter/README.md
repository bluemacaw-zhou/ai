# AI Starter - AI 工作区共享工具包

AI Starter 是一个 Python AI 工作区共享工具包，提供通用的工具函数和 AI 相关组件。设计目标是简化 AI 应用的开发，提供统一的配置管理、日志记录和 AI 框架集成。

---

## 目录结构

```
ai_starter/
├── __init__.py                 # 主入口，导出所有公共 API
├── core/                       # 核心模块（无外部依赖）
│   ├── config/
│   │   └── config.py          # Config 类，自动加载项目配置
│   └── log/
│       └── logging_utils.py   # 日志工具，支持 trace ID
├── http_client/
│   └── http_client_factory.py # HttpClientFactory，支持代理和 SSL
├── chromadb/
│   └── chromadb.py            # ChromaDB 类，集成 Embedding
├── embedding/
│   ├── embedding_interface.py # EmbeddingInterface 抽象基类
│   ├── embedding_glm.py       # GLMEmbedding（智谱 AI）
│   └── embedding_openai.py    # OpenAIEmbedding（待实现）
├── langchain/                  # LangChain 集成
│   ├── langchain_chat_zhipuai.py       # LangChain ChatModel 适配器
│   ├── langchain_chromadb.py          # LangChain ChromaDB 适配器
│   ├── langchain_embedding_interface.py # LangChain Embedding 接口
│   ├── langchain_glm_embedding.py      # LangChain GLM Embedding
│   └── pdf_chunker.py         # PDF 文档分块处理器
├── llama_index/                # LlamaIndex 集成
│   ├── llm_factory.py         # ZhipuLLMFactory
│   ├── embedding_factory.py   # ZhipuEmbeddingFactory
│   └── global_settings.py     # ZhipuGlobalSettings
├── llm/
│   └── zhipuai_base.py        # ZhipuAIBase（HTTP API 调用基类）
└── qwen_agent/                 # Qwen-Agent 集成
    └── qwen_agent_chat_zhipuai.py # Qwen-Agent ChatModel 适配器
```

---

## 核心概念

### LLM (Large Language Model)
大型语言模型，如 GPT、GLM 等，用于文本生成和理解。

**AI Starter 提供**：
- `ZhipuAIBase`：智谱 AI HTTP API 调用的公共基类
- `LangChainChatZhipuAI`：LangChain 兼容的智谱 AI 适配器
- `QwenAgentChatZhipuAI`：Qwen-Agent 兼容的智谱 AI 适配器

### Agent
智能体，能够使用工具并执行复杂任务的 AI 系统。

**AI Starter 提供**：
- Qwen-Agent 集成，支持构建具有工具调用能力的 Agent

### LangChain
用于开发由语言模型驱动的应用程序的框架。

**AI Starter 提供**：
- LangChain ChatModel 集成
- LangChain Embedding 集成
- LangChain ChromaDB 向量存储集成
- PDF 文档加载和分块

### LlamaIndex
用于连接大语言模型与私有数据的框架。

**AI Starter 提供**：
- `ZhipuLLMFactory`：创建智谱 AI LLM
- `ZhipuEmbeddingFactory`：创建智谱 AI Embedding
- `ZhipuGlobalSettings`：全局设置配置

### RAG (Retrieval-Augmented Generation)
检索增强生成，结合检索和生成技术提高回答质量。

**AI Starter 提供**：
- ChromaDB 向量数据库客户端
- PDF 文档处理
- 与 LangChain 和 Qwen-Agent 的集成

### Embedding
文本向量化，将文本转换为数值向量表示。

**AI Starter 提供**：
- `EmbeddingInterface`：统一 Embedding 接口
- `GLMEmbedding`：智谱 AI Embedding 实现
- `LangChainGLMEmbedding`：LangChain 兼容的智谱 AI Embedding

### Vector Database
向量数据库，用于存储和检索向量化的文本。

**AI Starter 提供**：
- `ChromaDB`：ChromaDB 客户端，集成 Embedding 自动文本向量化
- `LangchainChromadb`：LangChain 兼容的 ChromaDB 适配器

---

## 模块说明

### 核心模块 (core)

无需任何可选依赖，提供基础功能：

**配置管理 (Config)**：
```python
from ai_starter import Config

config = Config()  # 自动从项目根目录加载 config.yaml
api_key = config.get("zhipu.api_key")
```

**日志管理 (get_logger)**：
```python
from ai_starter import get_logger, with_trace

logger = get_logger(__name__)

@with_trace  # 自动分配 trace_id
def process():
    logger.info("Processing...")  # [trace-id] Processing...
```

**HTTP 客户端 (HttpClientFactory)**：
```python
from ai_starter import HttpClientFactory

http_client = HttpClientFactory.create()  # 自动读取代理和 SSL 配置
```

### ChromaDB 模块

向量数据库客户端，自动处理文本向量化：

```python
from ai_starter import ChromaDB, GLMEmbedding

embedding = GLMEmbedding()
db = ChromaDB(embedding=embedding)

# 添加文本（自动向量化）
db.add(collection_name="docs", texts=["文本1", "文本2"])

# 搜索（自动向量化查询）
results = db.search(collection_name="docs", query_text="查询文本")
```

### Embedding 模块

统一的文本向量化接口：

```python
from ai_starter.embedding import GLMEmbedding

# 自动从 config.yaml 读取配置
embedding = GLMEmbedding()
vector = embedding.get_embedding("文本")
print(embedding.get_model_name())  # "GLM-embedding-2"
print(embedding.get_vector_dimension())  # 1024（embedding-2 和 embedding-3 都是1024维）

# 或手动指定模型
embedding = GLMEmbedding(model="embedding-3")
```

### LangChain 模块

LangChain 框架集成：

```python
from ai_starter import LangChainChatZhipuAI, LangChainGLMEmbedding, PDFChunker

# ChatModel
llm = LangChainChatZhipuAI()
response = llm.invoke("Hello!")

# Embedding
embedding = LangChainGLMEmbedding()

# PDF 处理
chunker = PDFChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.load_and_split("document.pdf")
```

### LlamaIndex 模块

LlamaIndex 框架集成（工厂模式）：

```python
from ai_starter.llama_index import ZhipuLLMFactory, ZhipuEmbeddingFactory, ZhipuGlobalSettings

# 方式1: 全局设置（推荐，自动配置 Settings.llm 和 Settings.embed_model）
ZhipuGlobalSettings.setup()  # 自动从 config.yaml 读取所有配置

# 方式2: 单独创建
llm = ZhipuLLMFactory.create()  # 自动从 config.yaml 读取配置
embedding = ZhipuEmbeddingFactory.create()

# 方式3: 参数覆盖
llm = ZhipuLLMFactory.create(model="glm-4-plus")  # 覆盖模型名称
```

### Qwen-Agent 模块

Qwen-Agent 框架集成：

```python
from ai_starter import QwenAgentChatZhipuAI

llm = QwenAgentChatZhipuAI()
response = llm.chat(messages=[{"role": "user", "content": "Hello!"}])
```

---

## 安装

### 基础安装（无可选依赖）
```bash
pip install ai-starter
```

### 安装特定功能
```bash
# Embedding 功能（智谱 AI）
pip install ai-starter[embedding]

# ChromaDB 功能（自动包含 Embedding）
pip install ai-starter[chromadb]

# LangChain 功能
pip install ai-starter[langchain]

# LlamaIndex 功能
pip install ai-starter[llama-index]

# Qwen-Agent 功能
pip install ai-starter[qwen-agent]

# 全部功能
pip install ai-starter[all]
```

### 依赖关系说明

| 安装命令 | 安装的依赖 |
|---------|-----------|
| `ai-starter[embedding]` | zhipuai, sniffio |
| `ai-starter[chromadb]` | chromadb, zhipuai, sniffio（包含 embedding） |
| `ai-starter[langchain]` | langchain-core, langchain-community, langchain-text-splitters, pypdf |
| `ai-starter[llama-index]` | llama-index-core, llama-index-llms-openai-like, llama-index-embeddings-openai-like |
| `ai-starter[qwen-agent]` | qwen-agent[rag] |

---

## 配置

在项目根目录创建 `config.yaml`（已 gitignore）：

```yaml
# ==================== 智谱AI 配置 ====================
zhipu:
  # API Key（必填）
  api_key: "your_api_key_here"

  # API Base URL（可选，默认值如下）
  api_base: "https://open.bigmodel.cn/api/paas/v4/"

  # LLM 配置
  llm:
    model: "glm-4-flash"     # 模型名称
    temperature: 0.7         # 温度参数（可选）

  # Embedding 配置
  embedding:
    model: "embedding-2"     # embedding-2 或 embedding-3（都是1024维）
    dimension: 1024          # 向量维度

# ==================== ChromaDB 配置 ====================
database:
  chromadb:
    host: "localhost"
    port: 8000
    username: "admin"       # HTTP 基础认证（可选）
    password: "admin"

# ==================== HTTP 客户端配置 ====================
http:
  # 代理配置
  proxy:
    enabled: true           # 是否启用代理
    http: "http://proxy:8080"
    https: "http://proxy:8080"

  # 超时配置（秒）
  timeout: 30

  # SSL 验证
  verify_ssl: false         # 是否验证 SSL 证书

# ==================== 日志配置 ====================
logging:
  level: "INFO"             # DEBUG, INFO, WARNING, ERROR
```

---

## 打包

### 方式 1: 不切换目录（推荐）

在工作区根目录 `ai/` 下执行：

```bash
# 指定项目目录
uv build ai-starter

# 或使用绝对路径
uv build /d/workspace/ai/ai-starter
```

### 方式 2: 切换到项目目录

```bash
cd /d/workspace/ai/ai-starter
uv build
```

### 打包产物位置

```
ai/
└── ai-starter/
    └── dist/
        ├── ai-starter-0.1.0.dev-py3-none-any.whl    # Python 包（类似 .jar）
        └── ai-starter-0.1.0.dev.tar.gz             # 源码包
```

---

## 被其他项目引用

### 方式 1: 开发阶段 - 引用本地路径

在子项目的 `pyproject.toml` 中添加：

```toml
[project]
dependencies = [
    "ai-starter @ file:///${PROJECT_ROOT}/../ai-starter",
]
```

### 方式 2: 打包后引用

```toml
[project]
dependencies = [
    "ai-starter @ file:///${PROJECT_ROOT}/../ai-starter/dist/ai-starter-0.1.0.dev-py3-none-any.whl",
]
```

### 使用示例

```python
from ai_starter import say_hello

result = say_hello()  # "Hello, World!"
```

---

## 版本管理

| 版本类型 | 版本号 | 说明 |
|---------|--------|------|
| 开发版本 | `0.1.0.dev` | 开发中，随时变化 |
| 正式版本 | `0.1.0` | 稳定版本，可发布 |

修改版本号需同步更新：
- `pyproject.toml` 中的 `version` 字段
- `__init__.py` 中的 `__version__` 字段

# AI 工作区

AI 相关开发项目的 Python monorepo，包含基于 LangChain、LlamaIndex 等框架的多个演示项目。

## 技术栈

- **Python** 3.13
- **包管理**: uv
- **LLM 框架**: LangChain, LlamaIndex, Qwen-Agent
- **向量数据库**: ChromaDB, Qdrant
- **LLM/Embedding**: 智谱 AI (GLM-4, Embedding-2/3)

---

## 📁 项目分类

### 🔧 基础设施

#### [ai-starter](./ai-starter/README.md) - 共享工具库
跨所有项目的统一工具库，提供配置管理、日志、HTTP 客户端、LLM/Embedding 工厂等。

**核心功能**:
- `Config` - 配置管理（自动查找项目根目录）
- `get_logger()` - 标准日志 + 自动 Trace ID
- `HttpClientFactory` - HTTP 客户端（支持代理、SSL）
- `ZhipuLLMFactory` / `ZhipuEmbeddingFactory` - LlamaIndex 工厂
- `LangChainChatZhipuAI` - LangChain 集成
- `QwenAgentChatZhipuAI` - Qwen-Agent 集成

### 🦜 LangChain 项目

#### [rag-demo](./rag-demo/README.md) - RAG 文档问答
基于 LangChain 的完整 RAG（检索增强生成）系统。

**技术栈**: LangChain + ChromaDB + 智谱AI
**功能**: PDF 文档处理、向量索引、多轮对话、Multi-Query 检索

### 🦙 LlamaIndex 项目

#### [workflow-demo](./workflow-demo/README.md) - Text-to-SQL
自然语言查询数据库，支持多表 JOIN 和外键推断。

**技术栈**: LlamaIndex Workflow + SQLite + 智谱AI
**功能**: CSV 导入、表结构推断、SQL 生成、事件驱动流程

#### [llama-demo](./llama-demo/README.md) - RAG Pipeline
LlamaIndex RAG 完整流程演示，包含 RAG Fusion 和 Rerank。

**技术栈**: LlamaIndex + Qdrant + 智谱AI
**功能**: PDF 文档索引、多查询融合、LLM 重排序、多轮对话

### 🗄️ 向量数据库

#### [chromadb-demo](./chromadb-demo/README.md) - ChromaDB 操作
纯 Python 的 ChromaDB 向量数据库操作演示。

**技术栈**: ChromaDB + 智谱AI Embedding
**功能**: 向量存储、相似度搜索、持久化

### 🐍 学习项目

#### [python-practice](./python-practice/README.md) - Python 练习
Python 语法和编码规范示例。

**功能**: Config 使用、Logging 使用、编码规范示例

---

## 🚀 快速开始

### 前置要求

- Python 3.12+ (推荐 3.13)
- [uv](https://docs.astral.sh/uv/) 包管理器

### 1. 安装依赖

```bash
# 克隆仓库
cd ai

# 安装基础库
uv sync --project ai-starter

# 安装具体项目（按需选择）
uv sync --project rag-demo        # LangChain RAG
uv sync --project workflow-demo   # LlamaIndex Text-to-SQL
uv sync --project llama-demo      # LlamaIndex RAG
```

### 2. 配置

每个项目根目录下创建 `config.yaml`:

```bash
# 复制示例配置
cd <项目目录>
cp config.example.yaml config.yaml

# 编辑配置，填入智谱AI API Key
vim config.yaml
```

配置示例：

```yaml
zhipu:
  api_key: "your_api_key_here"
  llm:
    model: "glm-4-flash"
  embedding:
    model: "embedding-2"

http:
  proxy:
    enabled: true
    http: "http://proxy:8080"
  verify_ssl: false
```

### 3. 运行

```bash
# 进入项目目录
cd <项目目录>

# 运行测试脚本
python <项目包名>/test_*.py

# 例如：
cd llama-demo
python llama_demo/test_llama_pipeline.py
```

---

## 📖 学习路径

### 入门 RAG（文档问答）
1. [chromadb-demo](./chromadb-demo/README.md) - 理解向量数据库基础
2. [rag-demo](./rag-demo/README.md) - LangChain RAG 完整流程
3. [llama-demo](./llama-demo/README.md) - LlamaIndex RAG 高级特性

### 入门 Text-to-SQL
- [workflow-demo](./workflow-demo/README.md) - 从 CSV 到自然语言查询

### 理解项目架构
1. [编码规范.md](./编码规范.md) - 项目编码标准
2. [ai-starter/README.md](./ai-starter/README.md) - 工具库设计
3. [总结.md](./总结.md) - AI 框架对比分析

---

## 🛠️ 常用命令

```bash
# 同步依赖
uv sync --project <项目名>

# 添加新依赖
cd <项目目录>
uv add <包名>

# 添加 ai-starter 可选功能
uv add "ai-starter[langchain]"     # LangChain 集成
uv add "ai-starter[llama-index]"   # LlamaIndex 集成
uv add "ai-starter[all]"           # 全部功能

# 构建 ai-starter
uv build ai-starter
```

---

## 📚 参考文档

- [总结.md](./总结.md) - AI 框架深度对比与分析
- [CLAUDE.md](./CLAUDE.md) - AI 辅助开发指南
- [编码规范.md](./编码规范.md) - Python 编码标准

### 官方文档
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- [智谱 AI](https://open.bigmodel.cn/)
- [uv 文档](https://docs.astral.sh/uv/)

---

## 📄 许可证

仅供内部使用。

# AI 工作区

AI 相关开发项目的 Python monorepo，各顶层项目独立使用 uv 管理依赖。

## 环境管理约定

- 仓库根目录不声明 uv workspace，也不创建统一的 Python 环境。
- 每个顶层项目拥有自己的 `pyproject.toml`、`.python-version`、`uv.lock` 和 `.venv/`。
- 当前各项目开发解释器固定为 Python 3.13；兼容范围仍由各自的 `requires-python` 声明。
- `nn-lab` 内部使用 uv workspace：章节共享 `nn-lab/.venv` 和 `nn-lab/uv.lock`。
- 默认使用官方 PyPI，根 `uv.toml` 保留已注释的历史公司代理配置。
- `ai-starter` 是可编辑安装的共享库，使用它的项目通过本地路径声明依赖；不需要先单独安装它。
- 提交各项目的 `uv.lock` 和 `.python-version`；忽略 `.venv/` 和本地配置文件。

## 项目入口

以下命令从仓库根目录执行；首次运行前使用 `uv sync --project <项目名> --locked` 安装该项目依赖。

| 项目 | 用途 | 运行或阅读入口 |
|---|---|---|
| `ai-starter` | 共享工具库 | [说明](ai-starter/README.md)，由其他项目引用 |
| `rag-demo` | LangChain 文档问答 | [说明](rag-demo/README.md) |
| `llama-demo` | LlamaIndex RAG | [说明](llama-demo/README.md) |
| `workflow-demo` | LlamaIndex Text-to-SQL | [说明](workflow-demo/README.md) |
| `chromadb-demo` | 向量数据库操作 | [说明](chromadb-demo/README.md) |
| `fastmcp-demo` | MCP 服务 | [说明](fastmcp-demo/readme.md) |
| `function-call-demo` | 工具调用实验 | `function-call-demo/function_call_demo/` |
| `huggingface-demo` | Hugging Face API、模型与 Transformers 实验 | `huggingface-demo/API_test/`、`huggingface-demo/trsanformers_test/` |
| `bert-demo` | BERT 模型训练与评估 | [说明](bert-demo/README.md) |
| `pytorch-demo` | PyTorch 训练示例 | `uv run --project pytorch-demo python pytorch-demo/main.py` |
| `chat-agent` | LangGraph + FastAPI 聊天服务 | `uv run --project chat-agent python chat-agent/src/main.py`，见[说明](chat-agent/README.md) |
| `python-practice` | Python 基础练习 | [说明](python-practice/README.md) |
| `nn-lab` | 从线性回归到 Transformer 的学习实验 | `cd nn-lab` 后执行 `uv sync --all-packages --locked` 和 `uv run jupyter lab` |

根目录执行 `bash clean-venvs.sh` 可递归查找虚拟环境，确认后删除。

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

#### [nn-lab](./nn-lab/README.md) - 神经网络实验
使用 uv workspace 统一管理从线性回归到 Transformer 的学习章节。首个章节为 [linear-demo](./nn-lab/linear-demo/README.md)，包含 NumPy 手写梯度下降和拟合可视化。

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

# 使用当前项目的虚拟环境运行脚本
uv run python <项目包名>/test_*.py

# 例如：
cd llama-demo
uv run python llama_demo/test_llama_pipeline.py
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

# llama-demo - LlamaIndex RAG Pipeline

LlamaIndex 完整 RAG（检索增强生成）流程演示，包含 **RAG Fusion**（多查询融合）和 **LLM Rerank**（智能重排序）等高级特性。

## 📝 项目简介

本项目演示如何使用 LlamaIndex 构建生产级 RAG 系统，相比简单的向量检索，本项目实现了：

- ✅ **RAG Fusion**: 生成多个查询变体，融合检索结果
- ✅ **LLM Rerank**: 使用 LLM 对检索结果进行智能重排序
- ✅ **相似度过滤**: 过滤低质量检索结果
- ✅ **多轮对话**: 支持上下文对话历史
- ✅ **ai-starter 集成**: 使用工厂类简化配置

## 🎯 核心特性

### RAG Fusion（查询融合）

传统 RAG 只使用用户原始问题检索，容易遗漏相关内容。RAG Fusion 通过生成多个查询变体提高召回率：

```
用户问题: "项目迭代流程有哪些？"
    ↓
[QueryFusionRetriever 生成 3 个变体]
    ↓
变体1: "项目迭代流程有哪些？"
变体2: "软件开发迭代的步骤是什么？"
变体3: "敏捷开发的迭代流程包括哪些环节？"
    ↓
[分别检索] → [融合排序] → [返回 Top K]
```

### LLM Rerank（智能重排序）

向量检索基于语义相似度，但可能不够精准。LLM Rerank 使用大模型理解上下文，重新排序：

```
[向量检索 Top 5]
    ↓
[LLM 理解查询意图 + 评估每个文档的相关性]
    ↓
[重新排序] → [返回 Top 2]
```

### 完整流程

```
PDF 文档
    ↓
[文档分块 512 tokens, overlap 200]
    ↓
[Embedding 向量化] (embedding-2, 1024维)
    ↓
[存储到 Qdrant 向量数据库]
    ↓
用户问题
    ↓
[RAG Fusion: 生成 3 个查询变体]
    ↓
[向量检索 Top 5]
    ↓
[LLM Rerank: 重排序保留 Top 2]
    ↓
[相似度过滤: 过滤 < 0.6 的结果]
    ↓
[CondenseQuestionChatEngine: 多轮对话]
    ↓
[Response Synthesizer: REFINE 模式生成答案]
    ↓
返回答案
```

## 🏗️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **框架** | LlamaIndex 0.12+ | 数据索引与检索框架 |
| **向量数据库** | Qdrant | 高性能向量检索（本地文件存储） |
| **LLM** | 智谱AI glm-4-flash | 对话生成、重排序 |
| **Embedding** | 智谱AI embedding-2 | 文档向量化（1024 维） |
| **工具库** | ai-starter | 配置、日志、HTTP 客户端 |

## 📁 项目结构

```
llama-demo/
├── llama_demo/
│   ├── __init__.py
│   └── test_llama_pipeline.py    # RAG Pipeline 主程序
├── data/                          # 存放 PDF 文档
├── qdrant_db/                     # Qdrant 本地数据（自动生成）
├── config.yaml                    # 配置文件（gitignore）
├── config.example.yaml            # 配置模板
├── pyproject.toml
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd llama-demo
uv sync
```

### 2. 配置

创建 `config.yaml`（复制 `config.example.yaml`）：

```yaml
# ==================== LlamaIndex 配置 ====================
llama:
  # 文档数据路径
  data_path: "./data"

  # 是否重建知识库（true: 重新构建, false: 从现有 collection 加载）
  build_knowledge_base: true

  # 文档分割配置
  chunk_size: 512          # 文本块大小
  chunk_overlap: 200       # 文本块重叠大小

  # 检索配置
  similarity_top_k: 5      # 检索召回 top k 结果
  num_queries: 3           # RAG Fusion 生成的查询数量

  # Rerank 配置
  rerank_top_n: 2          # Rerank 后保留的文档数
  similarity_cutoff: 0.6   # 相似度过滤阈值（低于此分数的文档被过滤）

# ==================== 智谱AI 配置 ====================
zhipu:
  # API Key（必填）
  api_key: "your_api_key_here"

  # API Base URL（可选）
  api_base: "https://open.bigmodel.cn/api/paas/v4/"

  # LLM 配置
  llm:
    model: "glm-4-flash"     # 智谱AI 模型名称

  # Embedding 配置
  embedding:
    model: "embedding-2"     # 智谱AI 嵌入模型名称
    dimension: 1024          # 向量维度

# ==================== Qdrant 向量数据库配置 ====================
qdrant:
  # 本地存储路径
  path: "./qdrant_db"

# ==================== HTTP 客户端配置 ====================
http:
  # 代理配置
  proxy:
    enabled: true
    http: "http://10.200.86.85:8080"
    https: "http://10.200.86.85:8080"

  # 超时配置（秒）
  timeout: 30

  # SSL 验证
  verify_ssl: false

# ==================== 日志配置 ====================
logging:
  level: "INFO"
```

### 3. 准备文档

将 PDF 文档放入 `data/` 目录：

```bash
mkdir -p data
cp your_documents.pdf data/
```

### 4. 运行

```bash
# 方式1: 直接运行
python llama_demo/test_llama_pipeline.py

# 方式2: 使用虚拟环境
.venv/Scripts/python.exe llama_demo/test_llama_pipeline.py
```

## 💻 代码说明

### LlamaRAGPipeline 类

```python
from llama_demo.test_llama_pipeline import LlamaRAGPipeline

# 初始化（自动从 config.yaml 读取配置）
pipeline = LlamaRAGPipeline()

# 对话
response = pipeline.chat("项目迭代流程有哪些？")
print(response)

# 重置对话历史
pipeline.reset_chat()

# 关闭资源
if pipeline.qdrant_client:
    pipeline.qdrant_client.close()
```

### 核心方法

| 方法 | 说明 |
|------|------|
| `__init__()` | 初始化：配置模型、创建向量数据库、构建/加载知识库 |
| `build_knowledge_base()` | 构建知识库：加载 PDF → 分块 → 向量化 → 存储 |
| `load_existing_index()` | 从现有 Qdrant collection 加载索引 |
| `chat(message: str)` | 对话接口：输入问题，返回答案 |
| `reset_chat()` | 重置对话历史 |

### 使用 ai-starter 工厂类

代码使用 `ai-starter` 提供的工厂类简化配置：

```python
from ai_starter.llama_index import ZhipuGlobalSettings

# 一行代码配置 LLM 和 Embedding（自动读取 config.yaml）
ZhipuGlobalSettings.setup()

# 等价于：
# Settings.llm = ZhipuLLMFactory.create()
# Settings.embed_model = ZhipuEmbeddingFactory.create()
```

**优势**：
- ✅ 无需手动配置 API Key、Base URL
- ✅ 自动创建 HTTP 客户端（代理、SSL）
- ✅ 配置优先级：参数 > config.yaml > 默认值

## 🔧 配置说明

### RAG 参数调优

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `chunk_size` | 512 | 文本块大小 | 增大 → 更多上下文，但检索精度下降 |
| `chunk_overlap` | 200 | 文本块重叠 | 增大 → 避免语义断裂 |
| `similarity_top_k` | 5 | 检索召回数量 | 增大 → 召回率提高，但噪音增加 |
| `num_queries` | 3 | RAG Fusion 查询数 | 增大 → 召回率提高，但速度变慢 |
| `rerank_top_n` | 2 | Rerank 保留数量 | 调整为最终需要的文档数 |
| `similarity_cutoff` | 0.6 | 相似度阈值 | 提高 → 更精准，但可能过滤掉相关文档 |

### 知识库重建

**首次运行或文档更新后**，设置：

```yaml
llama:
  build_knowledge_base: true  # 重建知识库
```

**后续查询时**，设置：

```yaml
llama:
  build_knowledge_base: false  # 从现有索引加载
```

这样可以避免每次启动都重新构建索引，提高启动速度。

## 📊 性能优化

### 1. 向量维度选择

```yaml
zhipu:
  embedding:
    model: "embedding-2"  # 1024 维
    dimension: 1024
```

- `embedding-2` 和 `embedding-3` 都是 **1024 维**
- 维度必须与 Qdrant collection 配置一致
- 更改维度需要删除 `qdrant_db/` 并重建

### 2. Qdrant 本地存储

```yaml
qdrant:
  path: "./qdrant_db"  # 本地文件存储
```

- 适合开发和中小规模数据
- 生产环境建议使用 Qdrant Server

### 3. 代理和 SSL

```yaml
http:
  proxy:
    enabled: true  # 是否启用代理
  verify_ssl: false  # 是否验证 SSL 证书
```

## 🔍 调试技巧

### 1. 查看日志

日志自动包含 Trace ID，方便追踪请求：

```
2026-03-04 13:28:37 - [eac41e66-c860-455e-8e81-de596e073001] - INFO - ...
```

### 2. 检查向量数据库

```bash
# 查看 Qdrant 存储大小
du -sh qdrant_db/

# 删除并重建
rm -rf qdrant_db/
python llama_demo/test_llama_pipeline.py
```

### 3. 调整检索参数

如果答案不准确，尝试：

1. **增加召回**：`similarity_top_k: 10`
2. **增加 Rerank**：`rerank_top_n: 5`
3. **降低阈值**：`similarity_cutoff: 0.5`
4. **增加查询变体**：`num_queries: 5`

## 📚 与其他项目对比

| 项目 | 框架 | 向量DB | 特点 |
|------|------|--------|------|
| **llama-demo** | LlamaIndex | Qdrant | RAG Fusion + Rerank |
| [rag-demo](../rag-demo/README.md) | LangChain | ChromaDB | Multi-Query 检索 |
| [workflow-demo](../workflow-demo/README.md) | LlamaIndex | 无 | Text-to-SQL Workflow |

**选择建议**：
- 学习 LlamaIndex RAG → **llama-demo**
- 学习 LangChain RAG → **rag-demo**
- 学习 Text-to-SQL → **workflow-demo**

## 🛠️ 常见问题

### 1. 向量维度错误

```
ValueError: could not broadcast input array from shape (2048,) into shape (1024,)
```

**原因**：embedding 模型维度与 Qdrant collection 不匹配

**解决**：
1. 确认配置中 `zhipu.embedding.dimension` 为 1024
2. 删除 `qdrant_db/` 目录
3. 重新运行程序

### 2. 模块导入错误

```
ModuleNotFoundError: No module named 'llama_index.vector_stores'
```

**原因**：虚拟环境未选对或依赖未安装

**解决**：
```bash
# 确认使用项目虚拟环境
which python  # 应该指向 .venv/Scripts/python.exe

# 重新安装依赖
uv sync
```

### 3. 代理问题

如果 API 调用失败，检查代理配置：

```yaml
http:
  proxy:
    enabled: true  # 确认是否需要代理
    http: "http://your_proxy:8080"
```

## 📖 参考资源

### 官方文档
- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [LlamaIndex RAG 指南](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)
- [Qdrant 文档](https://qdrant.tech/documentation/)
- [智谱AI 开放平台](https://open.bigmodel.cn/)

### 项目内文档
- [总结.md](../总结.md) - AI 框架对比分析
- [ai-starter/README.md](../ai-starter/README.md) - 工具库详细说明
- [编码规范.md](../编码规范.md) - Python 编码标准

---

**最后更新**: 2026-03-04

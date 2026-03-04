# chromadb-demo - ChromaDB 向量数据库演示

纯 Python 的 ChromaDB 向量数据库操作演示，不依赖 LangChain/LlamaIndex，适合学习向量数据库基础概念。

## 📝 项目简介

本项目演示如何使用 ChromaDB 进行向量存储和相似度检索，通过纯 Python 实现帮助理解：

- ✅ 文本向量化（Embedding）
- ✅ 向量存储到 ChromaDB
- ✅ 相似度搜索（余弦相似度）
- ✅ 元数据过滤
- ✅ ai-starter 集成

## 🏗️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **向量数据库** | ChromaDB | 嵌入式向量数据库 |
| **Embedding** | 智谱AI embedding-2 | 文本向量化（1024维） |
| **工具库** | ai-starter | ChromaDB 和 Embedding 封装 |
| **语言** | Pure Python | 不依赖 LangChain/LlamaIndex |

## 📁 项目结构

```
chromadb-demo/
├── chromadb_demo/
│   ├── __init__.py
│   └── test_chromadb.py       # ChromaDB 测试和演示
├── config.yaml                # 配置文件（gitignore）
├── pyproject.toml
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd chromadb-demo
uv sync
```

### 2. 配置

创建 `config.yaml`：

```yaml
# ==================== 智谱AI 配置 ====================
zhipu:
  # API Key（必填）
  api_key: "your_api_key_here"

  # Embedding 配置
  embedding:
    model: "embedding-2"     # embedding-2 或 embedding-3
    dimension: 1024          # 向量维度

# ==================== ChromaDB 配置 ====================
database:
  chromadb:
    host: "localhost"
    port: 8000
    # HTTP 基础认证（可选，远程 ChromaDB 需要）
    username: "admin"
    password: "admin"

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

### 3. 运行

```bash
python chromadb_demo/test_chromadb.py
```

## 💻 代码说明

### test_chromadb.py

演示两个核心功能：

#### 1. 向量存储和检索

```python
from ai_starter.chromadb import ChromaDB
from ai_starter.embedding import GLMEmbedding

# 初始化（ChromaDB 自动集成 Embedding）
db = ChromaDB(embedding=GLMEmbedding())

# 添加文本（自动向量化）
texts = ["雨伞是最常见的雨具", "雨衣能够保护全身"]
metadatas = [{"type": "umbrella"}, {"type": "raincoat"}]
db.add(
    collection_name="test_collection",
    texts=texts,
    metadatas=metadatas
)

# 相似度搜索（自动向量化查询）
results = db.search(
    collection_name="test_collection",
    query_text="我需要一个能遮雨的工具",
    n_results=3
)

for doc, distance, metadata in results:
    print(f"文档: {doc}")
    print(f"距离: {distance}")
    print(f"元数据: {metadata}")
```

#### 2. 文本相似度计算

```python
import numpy as np
from ai_starter.embedding import GLMEmbedding

embedding = GLMEmbedding()

# 获取向量
vec1 = np.array(embedding.get_embedding("雨伞"))
vec2 = np.array(embedding.get_embedding("雨具"))

# 计算余弦相似度
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"相似度: {similarity:.4f}")
```

## 🎯 核心功能演示

### 1. 添加文本并搜索

```python
def test_add_and_search(collection_name: str):
    """测试添加文本和相似度搜索"""
    db = ChromaDB(embedding=GLMEmbedding())

    # 准备测试数据
    texts = [
        "雨伞是最常见的雨具，可以遮挡雨水",
        "雨衣能够保护全身不被雨水淋湿",
        "雨靴防水性能好，适合在雨天穿着",
        "雨帽可以保护头部免受雨水",
        "防水包能保护包内物品不受潮"
    ]

    metadatas = [
        {"type": "umbrella", "waterproof": True},
        {"type": "raincoat", "waterproof": True},
        {"type": "rain_boots", "waterproof": True},
        {"type": "rain_hat", "waterproof": True},
        {"type": "waterproof_bag", "waterproof": True}
    ]

    # 添加到数据库
    db.add(collection_name, texts=texts, metadatas=metadatas)

    # 搜索相似内容
    query = "我需要一个能遮雨的工具"
    results = db.search(collection_name, query_text=query, n_results=3)

    # 输出结果
    for doc, distance, metadata in results:
        print(f"  - {doc}")
        print(f"    距离: {distance:.4f}")
        print(f"    类型: {metadata['type']}")
```

### 2. 元数据过滤

```python
# 搜索时过滤元数据
results = db.search(
    collection_name="test_collection",
    query_text="防水工具",
    n_results=3,
    where={"waterproof": True}  # 只返回 waterproof=True 的文档
)
```

## 📊 运行输出示例

```
2026-03-04 14:30:15 - [abc123] - INFO - 测试: 添加文本并搜索相似内容
2026-03-04 14:30:16 - [abc123] - INFO - 添加 5 个文档到 collection 'test_collection'
2026-03-04 14:30:17 - [abc123] - INFO - 搜索相似文本: '我需要一个能遮雨的工具'

查询: '我需要一个能遮雨的工具'
最相似的内容:
  - 雨伞是最常见的雨具，可以遮挡雨水
    距离: 0.2345
    类型: umbrella
  - 雨衣能够保护全身不被雨水淋湿
    距离: 0.3456
    类型: raincoat
  - 雨帽可以保护头部免受雨水
    距离: 0.4567
    类型: rain_hat

2026-03-04 14:30:18 - [abc123] - INFO - ✓ 测试完成
```

## 🔍 技术原理

### 向量相似度检索流程

```
用户查询: "我需要一个能遮雨的工具"
    ↓
[GLMEmbedding.get_embedding()]  ← 调用智谱AI API
    ↓
查询向量: [0.12, -0.34, 0.56, ...]  (1024维)
    ↓
[ChromaDB 余弦相似度检索]
    ↓
文档向量库:
  - "雨伞是最常见的雨具" → [0.11, -0.32, 0.54, ...]  距离: 0.2345
  - "雨衣能够保护全身" → [0.09, -0.28, 0.51, ...]  距离: 0.3456
  - "雨靴防水性能好" → [0.05, -0.20, 0.42, ...]  距离: 0.5123
    ↓
[返回 Top K 最相似结果]
```

### 余弦相似度计算

```python
similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

- 结果范围: [-1, 1]
- 1.0 表示完全相同
- 0.0 表示无关
- -1.0 表示完全相反

## 🎓 设计理念

### 为什么不用 LangChain/LlamaIndex？

1. **学习目的**: 理解向量数据库底层操作
2. **轻量化**: 只需要向量存储和检索功能
3. **灵活性**: 自定义接口，便于扩展

### 架构特点

- **接口抽象**: `EmbeddingInterface` 支持切换不同 Embedding 实现
- **自动集成**: ChromaDB 自动处理文本向量化
- **配置驱动**: 所有配置从 `config.yaml` 读取

## 📚 与其他项目对比

| 特性 | chromadb-demo | rag-demo | llama-demo |
|------|---------------|----------|------------|
| **框架** | Pure Python | LangChain | LlamaIndex |
| **向量DB** | ChromaDB | ChromaDB | Qdrant |
| **适用场景** | 向量数据库学习 | 生产级 RAG | 高级 RAG |
| **复杂度** | ⭐ 简单 | ⭐⭐⭐ 完整 | ⭐⭐⭐⭐ 高级 |
| **文档处理** | ❌ | ✅ PDF 切分 | ✅ PDF 切分 |
| **检索策略** | ❌ 基础 | ✅ Multi-Query | ✅ RAG Fusion + Rerank |
| **问答链** | ❌ | ✅ RetrievalQA | ✅ Chat Engine |
| **适合人群** | 初学者 | 实战开发者 | 高级开发者 |

**学习路径建议**:
1. **chromadb-demo** - 理解向量数据库基础
2. **rag-demo** - 学习完整 RAG 流程（LangChain）
3. **llama-demo** - 掌握高级检索技术（LlamaIndex）

## 📦 依赖

```toml
dependencies = [
    "ai-starter[chromadb,embedding]",  # ChromaDB + GLMEmbedding
    "numpy>=2.2.3",                     # 向量计算
]
```

**ai-starter 提供**:
- `ChromaDB` - ChromaDB 客户端封装
- `GLMEmbedding` - 智谱AI Embedding 实现
- `Config` - 配置管理
- `get_logger()` - 日志工具

## 🛠️ 扩展

### 切换 Embedding 实现

```python
# 使用 OpenAI Embedding（需要自己实现）
from ai_starter.embedding import EmbeddingInterface

class OpenAIEmbedding(EmbeddingInterface):
    def get_embedding(self, text: str) -> list[float]:
        # 调用 OpenAI API
        pass

    def get_model_name(self) -> str:
        return "text-embedding-ada-002"

    def get_vector_dimension(self) -> int:
        return 1536

# 使用自定义 Embedding
db = ChromaDB(embedding=OpenAIEmbedding())
```

## 📖 参考资源

### 官方文档
- [ChromaDB 文档](https://docs.trychroma.com/)
- [智谱AI 开放平台](https://open.bigmodel.cn/)

### 项目内文档
- [ai-starter/README.md](../ai-starter/README.md) - 工具库详细说明
- [总结.md](../总结.md) - AI 框架对比分析

---

**最后更新**: 2026-03-04

# workflow-demo - Text-to-SQL Workflow

基于 **LlamaIndex Workflow** 的 Text-to-SQL 系统，演示如何将自然语言转换为 SQL 查询并执行，支持多表 JOIN 和外键推断。

## 📝 项目简介

本项目演示如何使用 LlamaIndex 构建生产级 Text-to-SQL 系统，包含：

- ✅ **CSV 数据加载**: 自动加载和解析 CSV 文件
- ✅ **表结构推断**: LLM 自动推断主键/外键关系
- ✅ **向量索引**: 基于表描述的向量检索
- ✅ **SQL 生成**: 支持 JOIN、聚合、子查询
- ✅ **事件驱动 Workflow**: 类型安全的执行流程
- ✅ **ai-starter 集成**: 使用工厂类简化配置

## 🏗️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **框架** | LlamaIndex 0.12+ | 数据索引与检索框架 |
| **Workflow** | LlamaIndex Workflows | 事件驱动架构 |
| **数据库** | SQLite | 关系型数据库（WAL 模式） |
| **LLM** | 智谱AI glm-4-flash | SQL 生成、答案合成 |
| **Embedding** | 智谱AI embedding-2 | 表描述向量化（1024维） |
| **工具库** | ai-starter | 配置、日志、LlamaIndex 集成 |

## 📁 项目结构

```
workflow-demo/
├── workflow_demo/
│   ├── model/
│   │   ├── table_info.py           # TableInfo 数据模型
│   │   └── events.py               # Workflow 事件定义
│   ├── utils/
│   │   ├── csv_loader.py           # CSV 数据加载器
│   │   ├── table_info_generator.py # LLM 生成表描述
│   │   ├── sqlite_loader.py        # SQLite 数据库加载器
│   │   ├── text2sql_index_builder.py    # 向量索引构建
│   │   ├── text2sql_workflow.py    # Workflow 入口
│   │   └── text2sql_workflow_impl.py    # Workflow 实现
│   ├── test_workflow.py            # 完整测试（11个查询）
│   └── __init__.py
├── data/
│   └── RelationalTestData/         # 测试数据（users/orders/products）
├── db/
│   ├── workflow-demo.db            # SQLite 数据库（自动生成）
│   └── table_info/relational/      # TableInfo JSON 缓存
├── config.yaml                     # 配置文件（gitignore）
├── pyproject.toml
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd workflow-demo
uv sync
```

### 2. 配置

创建 `config.yaml`：

```yaml
# ==================== 智谱AI 配置 ====================
zhipu:
  # API Key（必填）
  api_key: "your_api_key_here"

  # API Base URL（可选）
  api_base: "https://open.bigmodel.cn/api/paas/v4/"

  # LLM 配置
  llm:
    model: "glm-4-flash"     # 模型名称
    temperature: 0.0         # Text-to-SQL 建议使用 0.0

  # Embedding 配置
  embedding:
    model: "embedding-2"     # embedding-2 或 embedding-3
    dimension: 1024          # 向量维度

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

### 3. 准备数据

测试数据已包含在 `data/RelationalTestData/`：

**users.csv**:
```csv
user_id,name,age
1,Alice,25
2,Bob,30
3,Charlie,35
```

**orders.csv**:
```csv
order_id,user_id,product_id,quantity,order_date
101,1,201,2,2024-01-15
102,1,202,1,2024-01-16
103,2,201,3,2024-01-17
104,3,203,1,2024-01-18
```

**products.csv**:
```csv
product_id,product_name,price
201,Laptop,999.99
202,Mouse,29.99
203,Keyboard,59.99
```

### 4. 运行

```bash
python workflow_demo/test_workflow.py
```

## 💻 核心代码说明

### WikiTableQAPipeline 类

```python
from workflow_demo.test_workflow import WikiTableQAPipeline
import asyncio

# 初始化（自动从 config.yaml 读取配置）
pipeline = WikiTableQAPipeline(
    data_dir="data/RelationalTestData",
    db_path="db/workflow-demo.db",
    table_info_dir="db/table_info/relational",
    similarity_top_k=3,
    use_cache=True  # 使用缓存加速
)

# 查询
response = asyncio.run(pipeline.query("Show me all orders with user names"))
print(response)
```

### 完整流程

```
CSV 文件
    ↓
[Step 1: CSVLoader 加载数据]
    ↓
DataFrames: {users: df1, orders: df2, products: df3}
    ↓
[Step 2: TableInfoGenerator 生成表描述]
    ↓
TableInfo: {
  table_name: "users",
  description: "Users table. Primary key: user_id. Contains user profile...",
  sample_data: [[1, "Alice", 25], ...]
}
    ↓
[Step 3: SQLiteLoader 加载到数据库]
    ↓
SQLite tables: users, orders, products
    ↓
[Step 4: Text2SQLIndexBuilder 构建向量索引]
    ↓
ObjectIndex: 向量化表描述，支持相似度检索
    ↓
【查询阶段】
    ↓
用户问题: "Show me all orders with user names"
    ↓
[Workflow Step 1: 向量检索相关表]
    ↓
检索结果: orders (score: 0.85), users (score: 0.78)
    ↓
[Workflow Step 2: LLM 生成 SQL]
    ↓
SQL: SELECT o.*, u.name FROM orders o JOIN users u ON o.user_id = u.user_id;
    ↓
[Workflow Step 3: 执行 SQL]
    ↓
结果: [(101, 1, 201, 2, "2024-01-15", "Alice"), ...]
    ↓
[Workflow Step 4: LLM 生成自然语言答案]
    ↓
答案: "Here are all orders with user names: Order 101 by Alice..."
```

## 🎯 核心组件

### 1. CSVLoader（CSV 加载器）

```python
from workflow_demo.utils import CSVLoader

loader = CSVLoader(data_dir="data/RelationalTestData")
dfs = loader.load_csvs()

# dfs = {
#     "users": DataFrame(...),
#     "orders": DataFrame(...),
#     "products": DataFrame(...)
# }
```

### 2. TableInfoGenerator（表描述生成器）

使用 LLM 自动推断主键/外键关系：

```python
from workflow_demo.utils import TableInfoGenerator

generator = TableInfoGenerator(llm=llm)
table_infos = generator.generate_table_infos(
    dfs=dfs,
    table_info_dir="db/table_info/relational",
    use_cache=True
)

# 生成的 TableInfo 包含:
# - table_name: 表名
# - description: LLM 生成的表描述（含主键/外键信息）
# - sample_data: 前 5 行样本数据
```

**外键推断原理**:
```
LLM Prompt:
"Given the column 'user_id' in 'orders' table, and knowing there's a 'users' table,
 infer that user_id is likely a foreign key referencing users.user_id"

生成的描述:
"Orders table. Primary key: order_id. Foreign keys: user_id (references users),
 product_id (references products). Contains order transaction details."
```

### 3. SQLiteLoader（数据库加载器）

```python
from workflow_demo.utils import SQLiteLoader

loader = SQLiteLoader(db_path="db/workflow-demo.db")
engine = loader.load_to_sqlite(dfs=dfs, table_infos=table_infos)

# 特性:
# - 启用 WAL 模式（支持并发读写）
# - 自动创建表结构
# - 批量插入数据
```

### 4. Text2SQLIndexBuilder（索引构建器）

```python
from workflow_demo.utils import Text2SQLIndexBuilder

builder = Text2SQLIndexBuilder(
    engine=engine,
    table_infos=table_infos,
    similarity_top_k=3
)

# 功能:
# - 构建 ObjectIndex（向量化表描述）
# - 提供 obj_retriever（表检索器）
# - 提供 sql_retriever（SQL 执行器）
```

### 5. Text2SQLWorkflowRunner（Workflow 执行器）

```python
from workflow_demo.utils import Text2SQLWorkflowRunner

runner = Text2SQLWorkflowRunner(builder=builder)
response = await runner.run(query="Show me all orders")

# Workflow 包含 4 个步骤:
# 1. retrieve_tables - 检索相关表
# 2. generate_sql - 生成 SQL
# 3. execute_sql - 执行 SQL
# 4. generate_response - 生成答案
```

## 🔍 Workflow 实现细节

### Workflow 事件定义

```python
from workflow_demo.model.events import (
    StartEvent,          # 启动事件（包含用户问题）
    TableRetrieveEvent,  # 表检索事件（包含检索到的表）
    TextToSQLEvent,      # SQL 生成事件（包含生成的 SQL）
    StopEvent,           # 停止事件（包含最终答案）
)
```

### Workflow 步骤

```python
@step
async def retrieve_tables(self, ctx: Context, ev: StartEvent) -> TableRetrieveEvent:
    """步骤1: 向量检索相关表"""
    table_schema_objs = self._builder.obj_retriever.retrieve(ev.query)
    return TableRetrieveEvent(query=ev.query, table_context=...)

@step
async def generate_sql(self, ctx: Context, ev: TableRetrieveEvent) -> TextToSQLEvent:
    """步骤2: LLM 生成 SQL"""
    sql = self._generate_sql_from_context(ev.table_context, ev.query)
    return TextToSQLEvent(sql=sql, ...)

@step
async def generate_response(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
    """步骤3+4: 执行 SQL 并生成答案"""
    # 执行 SQL
    results = self._builder.sql_retriever.retrieve(ev.sql)

    # 生成答案
    answer = self._llm.chat(...)
    return StopEvent(result=answer)
```

## 📊 运行输出示例

```
=== Text-to-SQL JOIN 查询完整测试 ===
初始化 Pipeline...
✓ 使用 ZhipuGlobalSettings 配置 LLM 和 Embedding
✓ 加载 CSV 数据: 3 个表
✓ 生成 TableInfo（使用缓存）
✓ 加载到 SQLite 数据库
✓ 构建向量索引
✓ 初始化 Workflow Runner
✓ Pipeline 初始化完成

======================================================================
[测试 1/11] What are the names of all users?
======================================================================
[步骤1] 检索相关表: users
[步骤2] 生成 SQL: SELECT name FROM users;
[步骤3] 执行 SQL 并获取结果: 3 行数据
[步骤4] 生成答案:
The names of all users are Alice, Bob, and Charlie.

======================================================================
[测试 2/11] Show me all orders with user names
======================================================================
[步骤1] 检索相关表: orders, users
[步骤2] 生成 SQL:
SELECT o.order_id, o.user_id, o.product_id, o.quantity, o.order_date, u.name
FROM orders o
JOIN users u ON o.user_id = u.user_id;
[步骤3] 执行 SQL 并获取结果: 4 行数据
[步骤4] 生成答案:
Order 101: Alice ordered 2 items on 2024-01-15
Order 102: Alice ordered 1 item on 2024-01-16
Order 103: Bob ordered 3 items on 2024-01-17
Order 104: Charlie ordered 1 item on 2024-01-18

======================================================================
[测试 3/11] List all orders with user names and product names
======================================================================
[步骤1] 检索相关表: orders, users, products
[步骤2] 生成 SQL:
SELECT o.order_id, u.name AS user_name, p.product_name, o.quantity
FROM orders o
JOIN users u ON o.user_id = u.user_id
JOIN products p ON o.product_id = p.product_id;
[步骤3] 执行 SQL 并获取结果: 4 行数据
[步骤4] 生成答案:
Order 101: Alice ordered 2 Laptops
Order 102: Alice ordered 1 Mouse
...

✓ 所有测试通过: 11/11
```

## 🎓 核心特性详解

### 1. 向量表检索

使用向量相似度找到最相关的表：

```
问题: "Show me all orders with user names"
    ↓
[Embedding 向量化]
    ↓
查询向量: [0.12, -0.34, 0.56, ...]
    ↓
[与所有表描述计算相似度]
    ↓
表相似度:
  - orders: 0.85  ✓ 选中
  - users: 0.78   ✓ 选中
  - products: 0.42 ✗ 过滤
    ↓
返回 Top-K 表（k=2）
```

### 2. 外键自动推断

LLM 根据列名和表名推断关系：

```
输入:
  - orders.csv 包含列: order_id, user_id, product_id, quantity
  - 存在 users.csv 和 products.csv

LLM 推断:
  - order_id → 主键
  - user_id → 外键，可能引用 users.user_id
  - product_id → 外键，可能引用 products.product_id

生成描述:
  "Orders table. Primary key: order_id.
   Foreign keys: user_id (likely references users table),
   product_id (likely references products table).
   Contains order transaction details."
```

### 3. 缓存机制

TableInfo 自动缓存到 JSON 文件：

```
db/table_info/relational/
├── 0_orders.json
├── 1_users.json
└── 2_products.json
```

首次运行会调用 LLM 生成，后续运行直接加载缓存。

## 📚 与其他项目对比

| 特性 | workflow-demo | rag-demo | llama-demo |
|------|---------------|----------|------------|
| **框架** | LlamaIndex | LangChain | LlamaIndex |
| **数据类型** | 结构化（表格） | 非结构化（文档） | 非结构化（文档） |
| **查询方式** | Text-to-SQL | 文档检索 | 文档检索 |
| **核心技术** | Workflow + SQL | LCEL + Retriever | RAG Fusion + Rerank |
| **输出** | SQL 结果 | 文档片段+答案 | 答案 |
| **适用场景** | 数据分析、报表 | 知识库问答 | 高级文档问答 |
| **复杂度** | ⭐⭐⭐⭐ 高级 | ⭐⭐⭐ 完整 | ⭐⭐⭐⭐ 高级 |

**选择建议**:
- 查询结构化数据（SQL） → **workflow-demo**
- 文档问答（LangChain） → **rag-demo**
- 文档问答（LlamaIndex） → **llama-demo**

## 🔧 配置调优

### LLM 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `temperature` | 0.0 | Text-to-SQL 需要确定性输出 |
| `model` | glm-4-flash | 快速推理 |

### 检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `similarity_top_k` | 3 | 检索表数量 |

增大 `similarity_top_k` 可以提高召回率，但可能引入不相关的表。

## 📦 依赖

```toml
dependencies = [
    "ai-starter[llama-index]",        # LlamaIndex 集成
    "llama-index>=0.12.0",
    "llama-index-core>=0.12.0",
    "llama-index-llms-openai-like>=0.1.0",
    "llama-index-embeddings-openai-like>=0.1.0",
    "pandas>=2.2.3",                  # DataFrame 处理
    "sqlalchemy>=2.0.0",              # SQL 引擎
]
```

**ai-starter 提供**:
- `ZhipuGlobalSettings` - 全局 LLM/Embedding 配置
- `Config` - 配置管理
- `get_logger()` - 日志工具

## 🛠️ 常见问题

### 1. SQL 生成不准确怎么办？

尝试：
1. **降低温度**: `temperature: 0.0`（更确定性）
2. **增加表检索数**: `similarity_top_k: 5`
3. **改进表描述**: 手动编辑 `db/table_info/` 中的 JSON 文件

### 2. 如何清除缓存重新生成？

```bash
# 删除 TableInfo 缓存
rm -rf db/table_info/relational/

# 删除数据库
rm db/workflow-demo.db
```

### 3. 如何查看生成的 SQL？

日志中会显示生成的 SQL：

```
[步骤2] 生成 SQL: SELECT ... FROM ... JOIN ...
```

或直接查看代码中的 logger 输出。

### 4. 支持哪些 SQL 特性？

- ✅ SELECT、WHERE、JOIN
- ✅ 聚合函数（COUNT、SUM、AVG 等）
- ✅ GROUP BY、ORDER BY
- ✅ 子查询
- ❌ INSERT、UPDATE、DELETE（只读查询）

## 🔍 调试技巧

### 查看数据库内容

```bash
sqlite3 db/workflow-demo.db

# 查看所有表
.tables

# 查看表结构
.schema orders

# 查询数据
SELECT * FROM orders LIMIT 5;
```

### 查看 TableInfo 缓存

```bash
cat db/table_info/relational/0_orders.json
```

### 修改日志级别

```yaml
logging:
  level: "DEBUG"  # 查看详细日志
```

## 📖 参考资源

### 官方文档
- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
- [Text-to-SQL 教程](https://docs.llamaindex.ai/en/stable/examples/query_engine/text_to_sql_guide/)
- [智谱AI 开放平台](https://open.bigmodel.cn/)

### 项目内文档
- [ai-starter/README.md](../ai-starter/README.md) - 工具库详细说明
- [总结.md](../总结.md) - AI 框架对比分析
- [编码规范.md](../编码规范.md) - Python 编码标准

---

**最后更新**: 2026-03-04

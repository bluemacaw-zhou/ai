# AI 工作区

AI 相关开发项目的 Python monorepo，包含共享工具库。

## 概述

本工作区包含多个专注于 AI 开发、向量数据库、嵌入和 LangChain 集成的独立 Python 项目。所有项目共享一个通用工具库（`ai-starter`），提供配置管理、日志记录、数据库客户端和嵌入接口。

**核心技术栈：**
- Python 3.13
- uv（现代包管理器）
- ChromaDB（向量数据库）
- 智谱 AI GLM（嵌入 & 大语言模型）
- LangChain（RAG 框架）
- FastMCP（模型上下文协议）

## 项目

### ai-starter
跨所有项目的共享工具库，提供通用功能。

**功能特性：**
- `Config` - 统一配置管理（支持 YAML/JSON/TOML）
- `get_logger()` - 标准 Python 日志，支持文件/控制台输出
- `ChromaDB` - 向量数据库客户端，支持认证
- `GLMEmbedding` - 智谱 AI 嵌入接口

**可选依赖：**
- `[chromadb]` - ChromaDB 支持
- `[embedding]` - 智谱 AI 嵌入
- `[langchain]` - LangChain 集成
- `[all]` - 所有功能

### chromadb-demo
演示 ChromaDB 向量数据库操作和自定义嵌入实现。

**功能特性：**
- 向量存储和检索
- 相似度搜索
- 自定义嵌入接口（GLM、OpenAI）
- 文本向量化

### langchain-demo
使用 LangChain 的完整 RAG（检索增强生成）流水线。

**功能特性：**
- PDF 文档处理
- 文本分块和拆分
- 向量存储创建
- 问答链
- 交互式 QA 模式

### fastmcp-demo
FastMCP 服务端实现，演示模型上下文协议。

### python-practice
Python 语法和特性练习场。

## 快速开始

### 前置要求

- Python 3.12+（推荐 3.13）
- uv 包管理器

### 安装

```bash
# 克隆仓库
cd ai

# 同步所有项目依赖
uv sync --project ai-starter
uv sync --project chromadb-demo
uv sync --project rag-demo

# 或同步单个项目
uv sync --project <项目名>
```

### 配置

每个项目使用 `config.yaml` 进行配置：

```bash
# 复制示例配置
cp config.example.yaml config.yaml

# 编辑配置，填入 API 密钥和设置
vim config.yaml
```

配置示例：
```yaml
api:
  zhipuai:
    key: "your_api_key_here"
    proxy: "http://10.200.86.85:8080"
    verify_ssl: false

database:
  chromadb:
    host: "10.106.51.218"
    port: 8000
    username: "admin"
    password: "admin"

logging:
  level: "INFO"
  file: "logs/app.log"
```

## 开发指南

### 项目结构

```
ai/
├── ai-starter/           # 共享库
│   ├── ai_starter/       # 包源码
│   │   ├── __init__.py
│   │   ├── config.py         # 配置管理
│   │   ├── logging_utils.py  # 日志工具
│   │   ├── chromadb_client.py
│   │   ├── embedding_glm.py
│   │   └── ...
│   └── pyproject.toml
│
├── chromadb-demo/            # ChromaDB 演示
│   ├── chromadb_demo/
│   │   └── vector_operations.py
│   └── pyproject.toml
│
├── langchain-demo/           # LangChain 演示
│   ├── langchain_demo/
│   │   ├── pdf_processor.py
│   │   ├── vector_store.py
│   │   ├── qa_chain.py
│   │   └── main.py
│   └── pyproject.toml
│
├── python-practice/          # Python 练习
│   ├── python_practice/
│   │   ├── config_usage_example.py
│   │   ├── logging_usage_example.py
│   │   └── say_hello.py
│   └── pyproject.toml
│
├── config.example.yaml       # 配置模板
├── uv.toml                   # uv 工作区配置
└── README.md
```

### 添加依赖

```bash
# 添加到特定项目
cd <项目目录>
uv add <包名>

# 添加开发依赖
uv add --dev <包名>

# 添加可选依赖
uv add "ai-starter[chromadb,embedding]"
```

### 修改 ai-starter

通过可编辑安装，对共享库的更改会立即对依赖项目生效：

```bash
# 1. 编辑 ai-starter/ai_starter/ 中的代码
vim ai-starter/ai_starter/logging_utils.py

# 2. 测试更改（无需重新构建）
cd chromadb-demo
python chromadb_demo/vector_operations.py

# 3. 构建分发包（可选）
cd ai-starter
uv build
```

### 可选依赖模式

`ai-starter` 使用可选依赖来避免安装不必要的包：

```toml
[project]
dependencies = ["pyyaml>=6.0"]  # 仅核心依赖

[project.optional-dependencies]
chromadb = ["chromadb>=1.2.0"]
embedding = ["zhipuai>=2.1.0", "httpx>=0.27.0"]
langchain = ["langchain>=0.1.0", "langchain-community>=0.1.0"]
all = [...]  # 所有可选依赖
```

**在项目中使用：**
```toml
# 只安装需要的功能
dependencies = ["ai-starter[chromadb,embedding]"]
```

## 架构设计

### 共享库模式

`ai-starter` 作为可编辑依赖安装：

```toml
[tool.uv.sources]
ai-starter = { path = "../ai-starter", editable = true }
```

**优势：**
- 代码更改立即生效
- 断点调试正常工作
- 开发时无需重新构建
- 单一真相来源

### 配置文件优于环境变量

使用 `config.yaml` 而不是环境变量：

**原因：**
- 集中化配置
- 支持嵌套结构
- 易于切换环境
- 类型安全访问
- 版本控制的模板

### 标准 Python 日志

使用 Python 内置的 `logging` 模块（类似 log4j）：

**特性：**
- 行业标准
- 线程安全
- 多处理器（控制台、文件）
- 可配置格式化器
- 日志级别：DEBUG、INFO、WARNING、ERROR、CRITICAL
- **自动链路追踪**：每个线程/请求自动获得独立 trace_id

## 常用命令

### 依赖管理
```bash
# 同步依赖
uv sync --project <项目名>

# 添加依赖
cd <项目目录>
uv add <包名>

# 添加可选功能
uv add "ai-starter[all]"
```

### 构建
```bash
# 构建共享库
uv build ai-starter

# 输出: dist/ai_starter-0.1.0.dev-py3-none-any.whl
```

### 运行
```bash
# 使用项目虚拟环境运行
cd <项目目录>
python <脚本>.py

# 或使用 uv run
uv run python <脚本>.py
```

### 测试
```bash
# 运行示例
cd python-practice
python python_practice/config_usage_example.py
python python_practice/logging_usage_example.py
```

## IDE 设置

### PyCharm

1. 打开项目目录
2. 设置 → 项目 → Python 解释器
3. 选择：`<项目目录>/.venv/Scripts/python.exe`
4. 可编辑安装允许调试进入 `ai-starter`

### VS Code

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
```

## 参考链接

- [uv 文档](https://docs.astral.sh/uv/)
- [Python Logging](https://docs.python.org/3/library/logging.html)
- [ChromaDB](https://docs.trychroma.com/)
- [LangChain](https://python.langchain.com/)
- [智谱 AI](https://open.bigmodel.cn/)

## 许可证

仅供内部使用。

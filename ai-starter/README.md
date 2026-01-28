# AI Starter - 工作区共享工具包

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

## 打包产物位置

```
ai/
└── ai-starter/
    └── dist/
        ├── ai-starter-0.1.0.dev-py3-none-any.whl    # Python 包（类似 .jar）
        └── ai-starter-0.1.0.dev.tar.gz             # 源码包
```

## 被其他项目引用

### 方式 1: 开发阶段 - 引用本地路径

在子项目的 `pyproject.toml` 中添加：

```toml
# ai/fastmcp-demo/pyproject.toml
[project]
dependencies = [
    "fastmcp>=2.14.0",
    "ai-starter @ file:///${PROJECT_ROOT}/../ai-starter",
]
```

然后在子项目目录下运行：
```bash
cd /d/workspace/ai/fastmcp-demo
uv sync
```

### 方式 2: 打包后引用

在子项目的 `pyproject.toml` 中添加：

```toml
# ai/fastmcp-demo/pyproject.toml
[project]
dependencies = [
    "fastmcp>=2.14.0",
    "ai-starter @ file:///${PROJECT_ROOT}/../ai-starter/dist/ai-starter-0.1.0.dev-py3-none-any.whl",
]
```

### 使用示例

```python
# ai/fastmcp-demo/server.py
from ai_starter import say_hello

@mcp.tool()
def test():
    result = say_hello()
    return result  # "Hello, World!"
```

## 版本管理

| 版本类型 | 版本号 | 说明 |
|---------|--------|------|
| 开发版本 | `0.1.0.dev` | 开发中，随时变化 |
| 正式版本 | `0.1.0` | 稳定版本，可发布 |

修改版本号：
- `pyproject.toml` 中的 `version` 字段
- `__init__.py` 中的 `__version__` 字段

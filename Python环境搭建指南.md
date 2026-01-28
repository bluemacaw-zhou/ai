# Python 环境搭建指南

本文档介绍如何从零开始搭建 Python 开发环境，包括工具安装、代理配置、项目打包等完整流程。

所有示例基于当前工作区的实际项目结构进行说明。

## 当前工作区项目结构

```
D:\workspace\ai\
├── ai-starter/          # 共享库项目（可打包）
│   ├── ai_starter/      # 包目录
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── pyproject.toml
│   ├── README.md
│   └── dist/                # 打包输出目录
│       ├── ai_starter-0.1.0.dev0-py3-none-any.whl
│       └── ai_starter-0.1.0.dev0.tar.gz
│
├── python-practice/         # 应用项目（不打包）
│   ├── say_hello.py         # 测试脚本
│   ├── pyproject.toml
│   └── .venv/               # 项目虚拟环境
│
├── chromadb-demo/           # 其他项目...
├── fastmcp-demo/
└── Python环境搭建指南.md   # 本文档
```

**项目说明：**
- **ai-starter**: 共享工具库，提供通用函数供其他项目使用
- **python-practice**: 应用项目，引用并测试 ai-starter 的功能

---

## 一、安装 Python

### 1. 下载 Python

访问 Python 官网下载最新版本：
- 官网地址：https://www.python.org/downloads/
- 推荐版本：**Python 3.12+**

### 2. 安装 Python (Windows)

1. 运行下载的安装程序（`.exe` 文件）
2. **重要：** 勾选 **"Add Python to PATH"**
3. 选择 **"Install Now"** 或自定义安装路径
4. 安装完成后，打开命令行验证：

```bash
python --version
pip --version
```

---

## 二、安装 uv（现代化的 Python 包管理器）

### 1. uv 简介

**uv** 是 Astral 开发的极速 Python 包管理器，比 pip 快 10-100 倍，提供更好的依赖解析和项目管理能力。

### 2. 使用 pip 安装 uv (Windows)

**前提：** 已安装 Python 和 pip

```bash
# 使用 pip 安装 uv
pip install uv

# 验证安装
uv --version
```

---

## 三、配置代理（内网环境必需）

在企业内网环境中，通常需要配置私有 PyPI 仓库或代理来下载依赖包。

### 1. 配置 pip 代理 (Windows)

创建或编辑 pip 配置文件：

**配置文件位置：**
- `%APPDATA%\pip\pip.ini`
- 或：`C:\Users\<用户名>\AppData\Roaming\pip\pip.ini`

**配置内容示例：**

```ini
[global]
# 内网 PyPI 镜像地址
index-url = http://10.100.1.27:8688/repository/Pypi-group/simple

# 可选：信任内网仓库（避免 SSL 警告）
trusted-host = 10.100.1.27

# 可选：超时时间
timeout = 60

# 可选：本地缓存目录
cache-dir = D:\pip-cache
```

**验证配置：**
```bash
pip config list
```

---

### 2. 配置 uv 代理 (Windows)

创建 uv 配置文件：

**配置文件位置：**
- `%APPDATA%\uv\uv.toml`
- 或：`C:\Users\<用户名>\AppData\Roaming\uv\uv.toml`

**配置内容示例：**

```toml
# 全局索引源配置
[index]
url = "http://10.100.1.27:8688/repository/Pypi-group/simple"

# 可选：额外的索引源
[[index-extra]]
url = "https://pypi.org/simple"
```

---

## 四、项目初始化与依赖管理

### 1. 应用项目的 pyproject.toml 配置

以 `python-practice` 项目为例：

```toml
[project]
name = "python-practice"
version = "0.1.0"
description = "Python 语法练习项目"
requires-python = ">=3.12"
dependencies = [
    "ai-starter",  # 引用本地库项目
]

[tool.uv]
# 项目级别的索引源（覆盖全局配置）
index-url = "http://10.100.1.27:8688/repository/Pypi-group/simple"
# package = false 表示这是应用项目，不打包为库
package = false

# 配置本地依赖源
[tool.uv.sources]
ai-starter = { path = "../ai-starter", editable = true }
```

### 2. 安装依赖

```bash
# 创建虚拟环境并安装依赖
uv sync

# 添加新依赖
uv add requests

# 添加开发依赖
uv add --dev pytest
```

---

## 五、打包库项目（使用 hatchling）

### 1. hatchling 简介

**hatchling** 是现代化的 Python 打包工具，符合 PEP 517/PEP 660 规范，提供简洁的配置和强大的功能。

### 2. 标准库项目结构

以 `ai-starter` 项目为例：

```
ai-starter/
├── ai_starter/          # 包目录（使用下划线）
│   ├── __init__.py          # 包入口
│   └── utils.py             # 工具函数模块
├── pyproject.toml           # 项目配置
├── README.md                # 说明文档
├── uv.lock                  # 依赖锁定文件
├── dist/                    # 构建输出目录（自动生成）
└── .venv/                   # 虚拟环境（自动生成）
```

### 3. 库项目的 pyproject.toml 配置

以 `ai-starter` 项目为例：

```toml
# 构建系统配置（必需）
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# 项目元数据
[project]
name = "ai-starter"
version = "0.1.0.dev"
description = "Wind WM 工作区共享工具包"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

# uv 工具配置
[tool.uv]
index-url = "http://10.100.1.27:8688/repository/Pypi-group/simple"
# package = true 表示这是一个库项目，可以打包
package = true
```

**说明：**
- `[build-system]` 指定使用 hatchling 作为构建后端
- `package = true` 允许项目被打包
- hatchling 会自动识别 `ai_starter/` 目录作为包

### 4. 构建打包

```bash
# 进入库项目目录
cd ai-starter

# 构建包（生成 wheel 和 tar.gz）
uv build

# 构建结果在 dist/ 目录：
# - ai_starter-0.1.0.dev0-py3-none-any.whl  (wheel 格式)
# - ai_starter-0.1.0.dev0.tar.gz             (源码包)
```

---

## 六、常用命令速查

### uv 命令

```bash
# 项目管理
uv sync                      # 同步依赖（创建/更新虚拟环境）
uv add <package>             # 添加依赖
uv remove <package>          # 移除依赖
uv lock                      # 生成/更新 uv.lock 文件

# 运行命令
uv run python script.py      # 在虚拟环境中运行 Python 脚本
uv run pytest                # 在虚拟环境中运行测试

# 打包
uv build                     # 构建项目（生成 wheel 和 tar.gz）

# pip 兼容命令
uv pip install <package>     # 安装包
uv pip list                  # 列出已安装的包
uv pip show <package>        # 显示包信息
uv pip uninstall <package>   # 卸载包

# 缓存管理
uv cache dir                 # 显示缓存目录
uv cache info                # 显示缓存信息
uv cache clean               # 清理缓存
```

### pip 命令

```bash
pip install <package>        # 安装包
pip install -r requirements.txt  # 从文件安装依赖
pip uninstall <package>      # 卸载包
pip list                     # 列出已安装的包
pip freeze > requirements.txt    # 导出依赖列表
pip config list              # 查看配置
```

---

## 七、最佳实践

### 1. 项目组织

- **应用项目**（不打包）：设置 `package = false`
- **库项目**（需打包）：设置 `package = true`，添加 `[build-system]`
- 统一使用 `pyproject.toml`，避免 `setup.py`

### 2. 依赖管理

- 运行时依赖 → `dependencies`
- 开发依赖 → `uv add --dev`
- 本地依赖 → `[tool.uv.sources]` + `editable = true`

### 3. 版本控制

**提交到 Git：**
- ✅ `pyproject.toml`
- ✅ `uv.lock`
- ✅ `README.md`
- ✅ 源码目录

**忽略文件（.gitignore）：**
- ❌ `.venv/`
- ❌ `dist/`
- ❌ `__pycache__/`
- ❌ `*.pyc`

### 4. 缓存管理

- 设置 `UV_CACHE_DIR` 统一管理所有项目的缓存
- 类似 Maven 的 `.m2/repository`，一次下载，全局共享
- 定期清理：`uv cache clean`

---

## 八、总结

本文档涵盖了从 Python 安装到项目打包的完整流程：

1. ✅ **安装 Python** - 提供基础运行环境
2. ✅ **安装 uv** - 使用 pip 安装现代化包管理器
3. ✅ **配置代理** - pip.ini 和 uv.toml 支持内网环境
4. ✅ **项目管理** - 使用 pyproject.toml 统一配置
5. ✅ **打包发布** - hatchling 构建标准化包
6. ✅ **本地开发** - editable 模式支持实时调试
7. ✅ **实战演练** - 基于 ai-starter 和 python-practice 的完整示例

**关键要点：**
- ai-starter：共享库项目，使用 hatchling 打包
- python-practice：应用项目，使用 editable 模式引用共享库
- 工作区级别的 uv 缓存：类似 Maven 仓库，全局共享依赖
- PyCharm 断点调试：editable 模式让 IDE 追踪到源码

遵循本指南，你可以快速搭建标准化、高效的 Python 开发环境。

---

**参考文档：**
- uv 官方文档：https://docs.astral.sh/uv/
- Python Packaging User Guide：https://packaging.python.org/
- Hatchling 文档：https://hatch.pypa.io/latest/

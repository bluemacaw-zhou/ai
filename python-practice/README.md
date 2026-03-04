# python-practice - Python 编码规范和最佳实践

Python 语法特性练习场，提供 **ai-starter** 核心功能的使用示例和 **编码规范** 的完整可运行代码。

## 📝 项目简介

本项目作为 AI 工作区的学习和参考资源，包含：

- ✅ **ai-starter 核心功能示例**: Config、Logging、ChromaDB、Embedding
- ✅ **编码规范实践**: @property 封装、类型注解、命名规范
- ✅ **多线程日志追踪**: trace_id 自动分配和传递
- ✅ **完整可运行代码**: 每个示例都可以独立运行

## 📁 项目结构

```
python-practice/
├── python_practice/
│   ├── __init__.py
│   ├── say_hello.py                  # Hello World 示例
│   ├── config_usage_example.py       # Config 配置管理示例
│   ├── logging_usage_example.py      # 日志系统完整示例（11 个示例）
│   ├── test_thread_logging.py        # 多线程日志追踪示例
│   └── property_example.py           # @property 装饰器最佳实践
├── pyproject.toml
└── README.md
```

## 🎯 示例列表

### 核心功能示例

| 文件 | 说明 | 关键知识点 |
|------|------|-----------|
| **config_usage_example.py** | Config 配置管理 | 自动加载、懒加载、参数覆盖 |
| **logging_usage_example.py** | 日志系统（11 个示例） | trace_id、多线程、异常记录、结构化日志 |
| **test_thread_logging.py** | 多线程日志追踪 | 线程独立 trace_id、@with_trace 装饰器 |

### 编码规范示例

| 文件 | 说明 | 对应规范章节 |
|------|------|-------------|
| **property_example.py** | @property 最佳实践 | 编码规范.md - 属性封装 |

**完整示例位置**: `python-practice/python_practice/` 目录下

## 🚀 快速开始

### 1. 安装依赖

```bash
cd python-practice
uv sync
```

### 2. 配置（可选）

大部分示例不需要配置文件即可运行。如需测试 Config 和组件，创建 `config.yaml`：

```yaml
# ==================== 智谱AI 配置 ====================
zhipu:
  api_key: "your_api_key_here"

  llm:
    model: "glm-4-flash"

  embedding:
    model: "embedding-2"
    dimension: 1024

# ==================== ChromaDB 配置 ====================
database:
  chromadb:
    host: "localhost"
    port: 8000

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

### 3. 运行示例

```bash
# 配置管理示例
python python_practice/config_usage_example.py

# 日志系统示例（11 个完整示例）
python python_practice/logging_usage_example.py

# 多线程日志追踪
python python_practice/test_thread_logging.py

# @property 装饰器最佳实践
python python_practice/property_example.py
```

## 💻 示例说明

### 1. Config 配置管理 (config_usage_example.py)

演示 Spring Boot 风格的配置管理：

```python
from ai_starter import Config

# 方式1: 自动加载（推荐）
Config.load()  # 自动查找项目根目录的 config.yaml
api_key = Config().get("zhipu.api_key")

# 方式2: 组件自动加载（懒加载）
db = ChromaDB()  # 内部自动加载配置
embedding = GLMEmbedding()

# 方式3: 参数覆盖配置
embedding = GLMEmbedding(model="embedding-3")  # 参数优先级最高
```

**3 个示例**：
- 自动加载配置
- 组件自动读取配置
- 覆盖配置文件中的值

**完整代码**: `python_practice/config_usage_example.py`

### 2. 日志系统 (logging_usage_example.py)

基于 Python 标准 logging 模块的完整示例（11 个）：

```python
from ai_starter import get_logger, with_trace

# 基础用法
logger = get_logger(__name__)
logger.info("处理请求")  # 自动包含 trace_id

# 多线程场景（每个线程独立 trace_id）
def worker():
    logger.info("工作线程日志")  # 自动获得新 trace_id

# HTTP 请求场景（使用装饰器）
@with_trace
def handle_request(user_id):
    logger.info(f"处理用户请求: {user_id}")  # 每次调用新 trace_id
```

**11 个示例**：
1. 基础日志使用（5 个级别）
2. 指定日志级别
3. 日志输出到文件
4. 自动链路追踪（trace_id）
5. 多线程独立 trace_id
6. @with_trace 装饰器
7. 手动控制 trace_id（高级用法）
8. 从配置文件读取日志级别
9. 多个 logger 的使用
10. 异常日志记录
11. 结构化日志（extra 参数）

**日志格式**（固定）：
```
时间戳 - [trace_id] - 级别 - [线程名] - 文件:函数:行号 - 消息
```

**完整代码**: `python_practice/logging_usage_example.py`

### 3. 多线程日志追踪 (test_thread_logging.py)

专注演示多线程环境下的日志追踪：

```python
import threading
from ai_starter import get_logger, with_trace

logger = get_logger(__name__)

# 每个线程自动获得独立 trace_id
def worker(worker_id):
    logger.info(f"工作线程 {worker_id} 启动")
    logger.info(f"工作线程 {worker_id} 完成")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
    t.start()
```

**输出示例**：
```
[trace-id-abc] - INFO - [Worker-0] - 工作线程 0 启动
[trace-id-def] - INFO - [Worker-1] - 工作线程 1 启动
[trace-id-ghi] - INFO - [Worker-2] - 工作线程 2 启动
```

**观察要点**：
- 主线程显示为 `[MainThread]`
- 工作线程显示为 `[Worker-0]`, `[Worker-1]`, `[Worker-2]`
- 每个线程有独立的 trace_id
- 同一线程内的所有日志共享相同的 trace_id

**完整代码**: `python_practice/test_thread_logging.py`

### 4. @property 装饰器最佳实践 (property_example.py)

对应 **编码规范.md - 属性封装** 章节，提供完整可运行代码：

```python
# ❌ 错误示例 1：直接暴露内部变量
class BadPerson:
    def __init__(self, name, age):
        self.name = name
        self.age = age  # 无封装

# ❌ 错误示例 2：Java 风格 getter/setter
class JavaStylePerson:
    def get_name(self):
        return self._name
    def set_name(self, value):
        self._name = value

# ✅ 正确示例：使用 @property
class Person:
    def __init__(self, name, age):
        self._name = name  # 私有变量
        self._age = age

    @property
    def name(self):
        """获取姓名（只读）"""
        return self._name

    @property
    def age(self):
        """获取年龄"""
        return self._age

    @age.setter
    def age(self, value):
        """设置年龄（带验证）"""
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value

    @property
    def is_adult(self):
        """计算属性（每次计算）"""
        return self._age >= 18
```

**5 个示例**：
1. **基础用法**: getter、setter、只读属性、计算属性
2. **延迟加载** (Lazy Loading): 只在第一次访问时加载数据
3. **查询构建器**: 类似 `Text2SQLIndexBuilder` 的设计模式
4. **风格对比**: Java 风格 vs Python 风格
5. **总结**: @property 的 5 大优势

**使用 @property 的优势**：
1. **封装**: 变量私有化（`_variable`）
2. **简洁**: 访问时像普通属性（`obj.attr`）
3. **验证**: 在 setter 中添加验证逻辑
4. **灵活**: 可以从属性升级到 @property 而不改使用代码
5. **计算**: 可以实现计算属性（每次访问时重新计算）

**完整代码**: `python_practice/property_example.py`

## 📚 学习路径

### 入门（核心功能）

1. **say_hello.py** - 了解项目结构
2. **config_usage_example.py** - 学习配置管理
3. **logging_usage_example.py** - 掌握日志系统

### 进阶（编码规范）

4. **property_example.py** - 掌握 @property 最佳实践
5. **test_thread_logging.py** - 理解多线程日志追踪

### 实战（应用到项目）

参考其他项目的实现：
- **workflow-demo**: 完整的 @property 封装示例
- **llama-demo**: 配置管理和日志使用
- **rag-demo**: LangChain 集成

## 🔍 技术细节

### Config 自动加载原理

```
当前工作目录（cwd）
    ↓
[向上查找 pyproject.toml]
    ↓
找到项目根目录
    ↓
[加载 config.yaml]
    ↓
返回 Config 单例
```

**特点**：
- 从任意子目录运行都能找到配置文件
- 懒加载：首次 `get()` 时才加载
- 单例模式：全局共享一个 Config 实例

### Logging trace_id 机制

```
请求进入
    ↓
[trace_context 检查]
    ↓
有 trace_id? → 使用现有 trace_id
无 trace_id? → 生成新 UUID
    ↓
[存储到 threading.local]
    ↓
所有日志自动包含 trace_id
    ↓
请求结束 → 清除 trace_id
```

**适用场景**：
- HTTP 请求处理：每个请求独立 trace_id
- MQ 消息处理：每个消息独立 trace_id
- 多线程任务：每个线程独立 trace_id

## 📦 依赖

```toml
dependencies = [
    "ai-starter",  # 核心功能（Config、Logging、HTTP Client）
]
```

**ai-starter 提供**：
- `Config` - 配置管理（自动查找项目根目录）
- `get_logger()` - 日志工具（自动 trace_id）
- `with_trace` - 装饰器（为函数分配新 trace_id）
- `trace_context` - 上下文管理器（手动控制 trace_id）
- `ChromaDB` - 向量数据库客户端
- `GLMEmbedding` - 智谱AI Embedding

## 🛠️ 与编码规范的对应关系

| 示例文件 | 对应规范章节 | 说明 |
|---------|-------------|------|
| `property_example.py` | 属性封装 | @property vs getter/setter |
| `config_usage_example.py` | 配置管理 | 使用 Config 类而非环境变量 |
| `logging_usage_example.py` | 日志规范 | 使用 get_logger() 而非 print() |
| 所有文件 | 命名规范 | 遵循 PEP 8（snake_case） |

**编码规范.md 引用方式**：

```markdown
## 属性封装

**核心规则**: 使用 @property 而不是 Java 风格的 getter/setter

**完整示例**: `python-practice/python_practice/property_example.py`
```

## 📖 参考资源

### 项目内文档
- [ai-starter/README.md](../ai-starter/README.md) - 工具库详细说明
- [编码规范.md](../编码规范.md) - Python 编码标准
- [总结.md](../总结.md) - AI 框架对比分析

### 官方文档
- [Python Logging](https://docs.python.org/3/library/logging.html) - Python 标准库
- [PEP 8](https://peps.python.org/pep-0008/) - Python 代码风格指南

---

**最后更新**: 2026-03-04

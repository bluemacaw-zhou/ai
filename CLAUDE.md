# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python monorepo with AI-related projects. Uses **uv** for dependency management.

**Tech Stack:**
- Python 3.13
- uv (package manager)
- Private PyPI: `http://10.100.1.27:8688/repository/Pypi-group/simple`

## Project Structure

```
ai/
├── ai-starter/          # Shared utility library
├── workflow-demo/       # Text-to-SQL workflow (LlamaIndex)
├── rag-demo/           # RAG demo (LangChain + ChromaDB)
├── llama-demo/         # LlamaIndex demos
├── chromadb-demo/      # ChromaDB vector operations
├── fastmcp-demo/       # FastMCP server
└── python-practice/    # Python learning
```

**ai-starter** is the core shared library providing:
- `Config` - Configuration management (auto-finds project root via `pyproject.toml`)
- `get_logger()` - Standard Python logging with automatic trace IDs
- `HttpClientFactory` - HTTP client with proxy/SSL configuration
- `ChromaDB` - Vector database client
- `GLMEmbedding` / `LangChainGLMEmbedding` - ZhipuAI embeddings
- `CustomChatZhipuAI` - ZhipuAI LLM (supports both LangChain and Qwen-Agent)
- `PDFChunker` - PDF document processing
- Optional dependencies: `[chromadb]`, `[embedding]`, `[langchain]`, `[all]`

## Architecture Patterns

### Editable Installation
All projects reference `ai-starter` as editable dependency. Changes to `ai-starter` are immediately available without rebuilding:

```toml
# In application pyproject.toml
[tool.uv.sources]
ai-starter = { path = "../ai-starter", editable = true }
```

### Configuration Loading
`Config` class automatically finds project root by searching upward for `pyproject.toml`, then loads `config.yaml` from project root. This works regardless of working directory:

```python
from ai_starter import Config

# Automatically finds project root and loads config.yaml
config = Config()  # Lazy loading on first get()
api_key = config.get("api.zhipuai.key")
```

**Key behavior:**
- Searches from `cwd` upward for first `pyproject.toml`
- Loads `config.yaml` from that directory
- Works from any subdirectory (e.g., `project/subdir/file.py`)
- `config.yaml` must be in project root (gitignored)

### Logging with Trace IDs
Standard Python logging with automatic trace ID per thread/request:

```python
from ai_starter import get_logger, with_trace

logger = get_logger(__name__)

@with_trace  # Auto-assigns new trace_id for each call
def handle_request():
    logger.info("Processing request")  # [trace-id-123] - Processing request
```

### HTTP Client Factory
`HttpClientFactory` reads all config from `config.yaml` (no parameters needed):

```python
from ai_starter import HttpClientFactory

# Automatically reads api.zhipuai.{verify_ssl, use_proxy, timeout} from config
http_client = HttpClientFactory.create()  # prefix defaults to "api.zhipuai"
```

## Key Commands

```bash
# Sync dependencies for a project
uv sync --project <project-name>

# Add dependency to current project
cd <project-dir>
uv add <package-name>

# Add ai-starter with optional features
uv add "ai-starter[chromadb,embedding]"

# Build ai-starter (rarely needed due to editable install)
uv build ai-starter
```

## Configuration Pattern

**Structure:**
```yaml
api:
  zhipuai:
    key: "your_api_key"
    verify_ssl: false
    use_proxy: true
    timeout: 60

models:
  llm:
    model: "glm-4-flash"
  embedding:
    model: "embedding-3"

database:
  chromadb:
    host: "localhost"
    port: 8000
```

**Setup:**
1. Copy `config.example.yaml` → `config.yaml`
2. Fill in secrets (API keys, passwords)
3. `config.yaml` is gitignored

## Coding Standards

**CRITICAL:** Follow all rules in `编码规范.md`:

### Type Annotations (Required)
All public methods and `@property` must have return type annotations:

```python
@property
def table_infos(self) -> list[TableInfo]:  # ✓ Required
    return self._table_infos

def create_llm(self, model: str | None = None) -> OpenAILike:  # ✓ Required
    return OpenAILike(model=model)
```

### Property Encapsulation
Use `@property` instead of Java-style getters/setters:

```python
# ✓ Correct
@property
def config(self) -> Config:
    return self._config

# ✗ Wrong
def get_config(self) -> Config:
    return self._config
```

### Multiple Inheritance for Interfaces
When implementing multiple framework interfaces, use explicit multiple inheritance:

```python
class CustomChatZhipuAI(LangChainChatModel, QwenAgentLLM):
    """
    Implements:
    - LangChainChatModel: _generate(), _stream()
    - QwenAgentLLM: chat(), support_multimodal_input
    """
```

### Import Statements
**File-level imports:** Use relative imports for internal modules (preferred for package structure):

```python
# At top of file (as module import)
from ..model import TableInfo
from .utils import helper_function
```

**`__main__` block imports:** ALWAYS use absolute imports (relative imports will fail):

```python
if __name__ == "__main__":
    # ✓ Correct - absolute import
    from workflow_demo.model import TableInfo
    from workflow_demo.utils.sqlite_loader import SQLiteLoader

    # ✗ Wrong - will raise ImportError
    from ..model import TableInfo
    from .sqlite_loader import SQLiteLoader
```

**Reason:** When running a file directly (`python file.py`), Python doesn't know the package structure, so relative imports fail with `ImportError: attempted relative import with no known parent package`.

**Pattern for dual support:**
```python
# Top of file - supports both modes
try:
    from ..model import TableInfo  # Works when imported as module
except ImportError:
    from workflow_demo.model import TableInfo  # Works when run directly
```

### Project Structure
All projects must follow standard structure:

```
project-name/           # Repository name (can use hyphens)
├── project_name/       # Package name (must use underscores)
│   ├── __init__.py
│   └── module.py
├── pyproject.toml
├── config.yaml         # Gitignored
└── README.md
```

## pyproject.toml Patterns

**Library (ai-starter):**
```toml
[tool.uv]
package = true  # This is a library
```

**Application:**
```toml
[project]
dependencies = ["ai-starter[chromadb,embedding]"]

[tool.uv]
package = false  # This is an application

[tool.uv.sources]
ai-starter = { path = "../ai-starter", editable = true }
```

## CRITICAL Rules

1. **⚠️ Proxy Environment**: NEVER execute `pip`, `uv`, `curl` in Claude Code terminal. Tell user to run commands in their terminal due to proxy issues.

2. **⚠️ No Unsolicited Documentation**: DO NOT create or modify documentation files (.md) or test cases unless explicitly requested by the user. This includes:
   - README files
   - Technical documentation
   - Change logs or improvement summaries
   - API documentation
   - Architecture diagrams
   - Test cases or examples for updated code
   Focus on functional code only. Only create documentation or test cases when the user explicitly asks for it.

3. **⚠️ No print() Statements**: NEVER use `print()` for logging. ALWAYS use `get_logger()` from `ai_starter`:
   ```python
   # ✗ Wrong
   print("Processing data...")

   # ✓ Correct
   from ai_starter import get_logger
   logger = get_logger(__name__)
   logger.info("Processing data...")
   ```

4. **⚠️ Follow 编码规范.md**: ALL code must strictly follow standards in `编码规范.md`:
   - **Type annotations**: Required on all public methods and `@property`
   - **Property encapsulation**: Use `@property` instead of getters/setters
   - **Multiple inheritance**: Explicit interface declarations
   - **Project structure**: Package directory required (not flat structure)
   - **Naming conventions**: Follow Python PEP 8 (see `编码规范.md`)
   - **Configuration**: Use `Config` class, not environment variables
   - **Logging**: Use `get_logger()`, never `print()`

## Documentation Standards

### 编码规范.md Writing Rules

When updating `编码规范.md`:

1. **Keep code examples minimal** - Only short snippets (5-20 lines) to illustrate the rule
2. **Full examples go to python-practice** - Complete, runnable code belongs in `python-practice/python_practice/`
3. **Reference examples clearly** - End each section with: `完整示例：python-practice/python_practice/xxx_example.py`
4. **Focus on rules, not tutorials** - Explain the "what" and "why", not detailed "how-to"
5. **Use tables and summaries** - Prefer structured format over long prose

**Example structure:**
```markdown
## 规范名称

### 核心规则

**一句话说明规则**

```python
# ✓ 正确：5-10 行示例
...

# ✗ 错误：对比示例
...
```

**完整示例：** `python-practice/python_practice/xxx_example.py`
```

## Reference

- Full documentation: `README.md`
- Coding standards: `编码规范.md`
- uv documentation: https://docs.astral.sh/uv/

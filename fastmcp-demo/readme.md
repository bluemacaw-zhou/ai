# FastMCP Demo

**MCP**（Model Context Protocol）服务器演示项目，基于 FastMCP 框架。

## 技术栈

- **框架**: FastMCP
- **协议**: MCP (Model Context Protocol)
- **传输**: HTTP

## 项目特点

- ✅ MCP 服务器实现
- ✅ HTTP 传输支持
- ✅ Docker 容器化部署
- ✅ 与 Claude Desktop 集成

## 什么是 MCP？

**MCP（Model Context Protocol）** 是 Anthropic 推出的开放标准协议，用于让 LLM 应用（如 Claude Desktop）安全地访问外部数据源和工具。

### 核心概念

```
[Claude Desktop / LLM 应用]
    ↓ (MCP 协议)
[MCP 服务器]
    ↓
[外部数据源/工具]
  - 文件系统
  - 数据库
  - API 服务
  - ...
```

### MCP 的优势

1. **标准化**: 统一的协议，避免重复开发集成
2. **安全**: 受控的权限管理
3. **可扩展**: 支持自定义工具和数据源
4. **即插即用**: 一次开发，多处使用

## 目录结构

```
fastmcp-demo/
├── fastmcp_demo.py      # MCP 服务器实现
├── Dockerfile           # Docker 镜像
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd fastmcp-demo
uv sync
```

### 2. 本地运行

```bash
# HTTP 传输模式
fastmcp run fastmcp_demo.py:mcp --transport http --port 8000
```

### 3. Docker 部署

```bash
# 构建镜像
docker build -t fastmcp_demo:latest .

# 运行容器
docker run -p 8000:8000 fastmcp_demo:latest
```

## 集成到 Claude Desktop

在 Claude Desktop 配置文件中添加：

```json
{
  "mcpServers": {
    "my-server": {
      "command": "fastmcp",
      "args": [
        "run",
        "path/to/fastmcp_demo.py:mcp",
        "--transport",
        "http",
        "--port",
        "8000"
      ]
    }
  }
}
```

## MCP 服务器示例

```python
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

# 注册工具
@mcp.tool()
def get_data(query: str) -> str:
    """获取数据的工具"""
    return f"查询结果: {query}"

# 注册资源
@mcp.resource("file://data/example.txt")
def get_file_content() -> str:
    """读取文件内容"""
    with open("data/example.txt") as f:
        return f.read()
```

## 依赖

```toml
dependencies = [
    "fastmcp>=2.14.0",
    "mcp>=1.24.0",
]
```

## MCP 生态

### 官方 MCP 服务器

- **filesystem**: 文件系统访问
- **github**: GitHub API
- **sqlite**: SQLite 数据库
- **puppeteer**: Web 浏览器自动化
- **更多**: https://code.claude.com/docs/en/mcp#popular-mcp-servers

### 自定义 MCP 服务器

你可以创建自己的 MCP 服务器来：
- 连接私有数据源
- 集成内部 API
- 提供专用工具

## 参考

- [MCP 官方文档](https://code.claude.com/docs/en/mcp)
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP 服务器列表](https://code.claude.com/docs/en/mcp#popular-mcp-servers)
- [Claude Desktop 集成](https://code.claude.com/docs/en/mcp#claude-desktop-integration)

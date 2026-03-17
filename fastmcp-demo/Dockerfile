# Team Toolbox MCP 服务 - 生产环境 Dockerfile
# 构建上下文: ai/ (父目录), 命令: docker build -f fastmcp-demo/Dockerfile .

# FROM 10.100.6.129:8987/python:3.13
FROM python:3.13

# 安装 uv
RUN pip install uv

# 设置工作目录
WORKDIR /app

# 复制 ai-starter 源码（editable 本地依赖）
COPY ai-starter/ ./ai-starter/

# 复制项目依赖文件
COPY fastmcp-demo/pyproject.toml fastmcp-demo/uv.lock ./

# 安装依赖（使用锁定版本，不安装开发依赖）
RUN uv sync --frozen --no-dev

# 复制应用代码
COPY fastmcp-demo/fastmcp_demo/ ./fastmcp_demo/

# 暴露端口
EXPOSE 20000

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:20000/health')" || exit 1

# 运行服务 - 使用 FastMCP HTTP 传输模式
CMD ["uv", "run", "fastmcp", "run", "fastmcp_demo/server.py:mcp", "--transport", "http", "--port", "20000", "--host", "0.0.0.0"]

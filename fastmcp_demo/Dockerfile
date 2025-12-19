# Team Toolbox MCP 服务 - 生产环境 Dockerfile
# 使用完整镜像，包含所有编译工具，适用于内网私有 pip 源

# FROM 10.100.6.129:8987/python:3.13
FROM python:3.13

# 设置工作目录
WORKDIR /app

# 配置 pip 使用私有仓库
# 复制 pip 配置文件到镜像
# COPY pip.conf /root/.config/pip/pip.conf

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（从私有 pip 仓库）
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY server.py .

# 暴露端口
EXPOSE 20000

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:20000/health')" || exit 1

# 运行服务 - 使用 FastMCP HTTP 传输模式
CMD ["fastmcp", "run", "server.py:mcp", "--transport", "http", "--port", "20000", "--host", "0.0.0.0"]

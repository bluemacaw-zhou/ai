# Chat Agent

基于 LangGraph 的聊天代理，使用 FastAPI 提供 HTTP 接口。

## 环境与启动

Python 3.13，使用 uv 独立管理依赖和虚拟环境。在本目录执行：

```bash
uv sync --locked
uv run python src/main.py
```

服务监听 `0.0.0.0:8000`。从仓库根目录也可运行：

```bash
uv run --project chat-agent python chat-agent/src/main.py
```

本地配置 `config.yaml` 放在 `chat-agent/` 根目录，不提交到 Git。启动前需按所用服务配置模型与外部服务连接。

`pyproject.toml`、`uv.lock`、`.python-version` 和 `.venv/` 均位于项目根目录，源码保留在 `src/`。

## 验证

```bash
uv run python -m unittest discover -s test -p 'test_prompt_loader.py'
```

架构图见 [design/README.md](design/README.md)。

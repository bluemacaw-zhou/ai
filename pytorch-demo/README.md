# PyTorch Demo

基于工作区的 `uv` 管理方式创建的 PyTorch 空白项目。

## 环境

- Python：3.13（本机安装的 CPython 3.13.3）
- PyTorch：2.8.0（提供 Windows / CPython 3.13 wheel）
- 运行目标：CPU；当前环境未发现 `nvidia-smi`，因此不配置 CUDA 专用索引。

## 使用

```powershell
cd D:\workspace\ai\pytorch-demo
uv sync
uv run python main.py
```

首次 `uv sync` 会在项目目录创建隔离的 `.venv` 并生成 `uv.lock`。

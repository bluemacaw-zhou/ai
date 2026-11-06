# NN Lab

从线性回归开始，逐步学习神经网络、反向传播和 Transformer。

使用 uv workspace 管理所有章节，共享根目录的 `.venv` 和 `uv.lock`，各章节在自己的 `pyproject.toml` 中声明依赖。

## 开始使用

```bash
cd /Users/michael/workspace/ai/nn-lab
uv sync --all-packages
uv run jupyter lab
```

在 JupyterLab 中打开章节 Notebook，选择当前环境的 Python 内核。

## 章节

- [linear-demo](linear-demo/README.md)：单变量线性回归、MSE、梯度下降与拟合可视化。

## 添加章节

为新章节创建目录和 `pyproject.toml`，将目录加入根配置的 `tool.uv.workspace.members`，然后运行 `uv sync --all-packages`。

```bash
# 为指定章节添加依赖
uv add --package linear-demo numpy
# 按锁文件恢复所有章节环境
uv sync --all-packages --locked
```

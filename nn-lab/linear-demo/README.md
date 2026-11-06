# 01：单变量线性回归与梯度下降

从 y = wx 到 y = wx + b，理解参数、预测、均方误差（MSE）、梯度和学习率。本目录对应课程专题一；示例主体是一个输入特征，不是一个 Transformer 层。

本目录编号对应已核对的语雀课程专题；目录编号不是运行时配置，也不决定模型层数。

## 推荐阅读

1. [demo_01_y_wx.py](<demo_01_y_wx.py>)：先用一个参数 W 拟合 y = 3x，逐轮刷新预测直线观察拟合过程。
2. [demo_02_gradient_descent.py](<demo_02_gradient_descent.py>)：观察 epoch、参数 w 与 loss 的关系。
3. [demo_03_fitting_process.py](<demo_03_fitting_process.py>)：用静态图和动态绘图观察拟合。
4. [demo_04_y_wx_b.py](<demo_04_y_wx_b.py>)：理解偏置 b 为什么能让直线离开原点。
5. [demo_05_approximate_fit.py](<demo_05_approximate_fit.py>)：分别设置 w、b 的学习率观察收敛。
6. [demo_06_linear_sgd.py](<demo_06_linear_sgd.py>)：把手写公式换成 Linear 和 SGD 的调用。

## 运行

在本项目 `linear-demo` 目录执行（`ai-starter` 从当前项目读取 `config.yaml`）：

```bash
uv sync --all-packages
uv run python demo_01_y_wx.py
```

按推荐顺序阅读和运行上述脚本，替换命令中的文件名即可。说明和公式推导均写在代码注释中。每个脚本由一个 demo 类组织教学步骤，通过文件末尾的入口调用 `run()`。

静态图需要关闭窗口后继续；实时拟合图统一使用 `tool/fitting_plot.py` 中的 `FittingPlot`，固定真实点和坐标范围，每轮只更新预测直线。demo01 每轮停留 0.5 秒，其余动态示例默认 0.3 秒，可通过 `interval` 调整。训练结束后保留最终图形，关闭窗口后继续或退出；训练中关闭窗口则结束当前训练。

在 PyCharm 中运行实时拟合示例时，使用独立图形窗口观察动画。公共绘图工具会在 macOS 的 PyCharm 环境中切换到 `MacOSX` 后端。若仍在 Plots 页签显示，在 Settings 中搜索 **Show plots in tool window** 并取消勾选（部分新版位于 **Python → Plots**，其他版本位于 **Tools → Python Plots / Python Scientific**），停止程序后重新运行。不要使用旧的 Python Console 会话。

主要依赖为 NumPy、Matplotlib 和本地 `ai-starter`（通过 `get_logger(__name__)` 创建日志，使用 `log.info` 输出训练过程）；`demo_06_linear_sgd.py` 使用本目录 `tool/linear.py` 中的标量 `Linear` 和 `tool/sgd.py` 中的 `SGD`，无需原项目的 `pytorch/` 模块。

内容复制自 `ziya-chatgpt-3/chat-1`，保留原始示例和学习顺序，已转为可直接运行的 Python 脚本。本章实际训练的是只有一到两个参数的线性回归模型。

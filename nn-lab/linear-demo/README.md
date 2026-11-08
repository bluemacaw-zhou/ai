# 一元线性模型与梯度下降

用 NumPy 手写训练过程，从单个权重开始，逐步理解均方误差、梯度、偏置、学习率，以及两层模型之间的反向传播。每个样本只有一个输入特征；数组中的多个元素表示多个样本。

## 学习顺序

| 示例 | 学习内容 | 运行后的展示 |
| --- | --- | --- |
| [01：基础拟合](demo_01_y_wx.py) | 手写预测、MSE、求导和参数更新，学习 `y = 3x` | 三个同步窗口：拟合直线、epoch-loss、W-loss |
| [02：偏置对照](demo_02_y_wx_b.py) | 两个模型共用一套训练逻辑，理解 `y = wx` 与 `y = wx + b` 的表达能力 | 三个同步窗口：无偏置拟合、有偏置拟合、两条 loss 曲线 |
| [03：学习率对照](demo_03_learning_rate.py) | 相同数据和初始权重，只改变学习率，观察跨过最低点与收敛速度 | 两张 epoch-loss 图、一张完整 W-loss 曲线与移动轨迹图 |
| [04：两层反向传播](demo_04_two_layer.py) | 串联两个 Linear，先反向传播第二层，再将输入梯度传给第一层 | 日志：第一轮完整中间值、每 20 轮的参数和损失、最终组合模型 |

代码包含变量类型、数组形状和公式推导注释。建议先阅读 01 的完整训练步骤，再看后续示例如何复用 `Linear` 和 `SGD`。

### 01：基础拟合

模型为 `Y_pred = X * W`，权重和权重梯度都是标量；预测值、误差以及 loss 对每个预测值的导数是一维数组。每轮按“预测 → 损失 → 展示 → 梯度 → 更新参数”的顺序执行，循环结束后统一展示最终状态。

三个窗口共享同一轮的权重、预测值和损失，绘图不重复计算训练结果。W-loss 图记录实际经过的参数点。

### 02：偏置对照

目标为 `y = 3x + 5`，使用关于 0 对称、包含 `x = 0` 的样本。两个模型使用相同数据、初值和学习率；无偏置模型固定 `b = 0`，有偏置模型同时更新 `w`、`b`。拟合图中的橙色虚线表示预测误差。

对称样本下，两者都能学到斜率约为 3，但无偏置模型无法补上截距，MSE 最终停在 25；有偏置模型的 MSE 接近 0。可修改 `target_b = 0.0`，观察两者都能拟合；也可把 `sample_count` 从 5 改为 101，在相同区间内对比样本数量。决定能否准确拟合的是模型和数据关系是否匹配，不能简单归因于样本数量。

### 03：学习率对照

两个模型学习 `y = 3x`，使用相同初始权重 `W = 0.1`，分别取学习率 `0.04` 和 `0.089`。本例的损失为 `11(W - 3)²`，更新后满足：

```text
W_new - 3 = (1 - 22 × lr) × (W - 3)
```

`lr = 0.04` 时快速靠近最低点；`lr = 0.089` 时反复跨过最低点，收敛反而更慢。参数来回跳跃时，loss 仍可能持续下降，因此需要结合第三个窗口的轨迹、当前大圆点和最新一步箭头观察。前两张图使用相同坐标范围，方便直接比较。

本例中 `lr > 1/11` 会发散；这个阈值取决于数据和损失函数，不能直接套用到其他任务。修改学习率实验时，若参数超出第三张图的范围，可同步调整 `weight_limits`。

### 04：两层反向传播

前向顺序为 `X → layer1 → h → layer2 → Y_pred → loss`，反向顺序相反：

```python
d_loss_d_h = layer2.backward(d_loss_d_y_pred)
layer1.backward(d_loss_d_h)
optimizer1.step()
optimizer2.step()
```

`backward()` 将本层参数梯度保存在模型属性中，返回 loss 对本层输入的导数，供前一层继续求导。所有层的梯度计算完成后再更新参数。原始输入 X 固定，不使用第一层返回的输入梯度来更新数据。

两层都包含偏置，没有激活函数，组合后仍是一条直线：

```text
y = w2 × (w1 × x + b1) + b2
  = (w2 × w1) × x + (w2 × b1 + b2)
```

训练目标仍是 `3x + 5`；各层参数不必单独等于 3 或 5，只需组合后的斜率、截距接近目标。本例重点是梯度传递，而非增加非线性表达能力。

## 运行

在 `linear-demo` 目录执行，让 `ai-starter` 读取当前项目的 `config.yaml`：

```bash
uv sync --all-packages
uv run python demo_01_y_wx.py
uv run python demo_02_y_wx_b.py
uv run python demo_03_learning_rate.py
uv run python demo_04_two_layer.py
```

每次选择一个示例运行，也可以直接在 PyCharm 中运行对应文件。主要依赖为 NumPy、Matplotlib 和本地 `ai-starter`；不依赖 PyTorch。

01–03 的三个窗口在 macOS 原生后端中按屏幕可用区域从左到右排列。每轮统一暂停一次处理窗口事件，默认间隔分别为 0.5、0.1、0.3 秒，可修改对应 `plt.pause()`。训练结束后保留图窗；训练中关闭任意一个窗口停止训练，关闭剩余窗口后退出。04 只输出日志，不打开窗口。

在 macOS 的 PyCharm 中，公共绘图基类会切换到 `MacOSX` 后端，以独立窗口显示动画。若图仍被 Plots 面板接管，可在设置中搜索 **Show plots in tool window** 并取消勾选，然后重新运行脚本。显式设置 `MPLBACKEND=Agg` 时保留无界面模式。

## 公共代码

- [Linear](tool/linear.py)：单输入、单输出线性层，保存权重、偏置及其梯度；`use_bias=False` 时禁用偏置。
- [SGD](tool/sgd.py)：梯度清零和参数更新，反向传播计算与参数更新分开执行。
- [FittingPlot](tool/fitting_plot.py)：抽象绘图基类，管理窗口、后端和刷新。各子类独立实现 `update()`，demo 直接调用，不额外封装协调类。
- [LinearFitPlot](tool/linear_fit_plot.py)、[EpochLossPlot](tool/epoch_loss_plot.py)、[WeightLossPlot](tool/weight_loss_plot.py)：拟合直线及两种训练曲线。
- [LossComparisonPlot](tool/loss_comparison_plot.py)、[LearningRatePathPlot](tool/learning_rate_path_plot.py)：多模型损失对比、完整损失曲线上的参数轨迹。
- [窗口排列](tool/window_layout.py)：在 macOS 原生后端中横向排列当前示例的窗口。

`Scalar` 表示一个 Python 或 NumPy 数值标量；`NDArray[np.float64]` 表示浮点数组，具体形状写在变量旁的注释中。导数命名采用 `d_loss_d_w`、`d_loss_d_y_pred` 等形式，明确“谁对谁求导”。

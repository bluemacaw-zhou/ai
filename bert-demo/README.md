# bert-demo

基于中文 BERT 的评论情感二分类示例。使用本地 `bert-base-chinese` 提取 `[CLS]` 特征，冻结 BERT 参数，只训练一层 `Linear(768, 2)` 分类层，在 ChnSentiCorp 数据集上判断评论是**负面（0）**还是**正面（1）**。

- 数据集：ChnSentiCorp（train / validation / test，test 1200 条）
- 选模型依据：验证集损失最低的一轮保存为 `best_classifier.pth`
- 早停规则：先观察到验证损失下降，之后连续 `patience` 轮出现“训练损失下降、验证损失上升”才停止
- 当前 `best_classifier.pth` 测试集结果：loss 0.2305，accuracy 0.9142，macro-F1 0.9142

## 准备

```bash
# 安装依赖（项目根目录 ai/ 下执行）
uv sync --project bert-demo

# 进入项目目录，下载数据集和模型（只需一次）
cd bert-demo
uv run python dataset_download.py          # → data/ChnSentiCorp/
uv run python model_tokenizer_download.py  # → model/bert-base-chinese/
```

`data/`、`model/`、`params/` 均已加入 `.gitignore`，不会提交到仓库。

### `uv run` 卡住时排查

`uv run` 会先自动同步环境，再执行脚本。如果停在下面的提示不动：

```
   Building ai-starter @ file:///Users/michael/workspace/ai/ai-starter
⠧ Preparing packages... (0/1)
```

说明 uv 正在重新构建 editable 依赖 `ai-starter`，需要从私有源 `10.100.1.27` 下载构建工具 hatchling。不在内网时连不上该地址，就会一直等待。

用 `-v` 查看详细日志，确认卡在哪一步：

```bash
uv run -v python predict.py
```

如果日志停在请求 `10.100.1.27`（如下载 hatchling），说明是网络问题。处理方式：

```bash
# 依赖已装好时：跳过同步直接运行
uv run --no-sync python predict.py
# 或在已激活的虚拟环境中直接运行
python predict.py

# 确实需要同步时：连上内网/VPN 后执行 uv sync，或尝试使用本地缓存
uv sync --offline
```

## 入口层用法

项目有三个入口，都在项目根目录运行。

### 1. 训练：`train.py`

```bash
uv run python train.py
uv run python train.py --epochs 50 --batch-size 32 --patience 3 --minimum-loss-change 0.001
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 30000 | 最大轮数，实际通常由早停提前结束 |
| `--batch-size` | 50 | 每批评论数 |
| `--patience` | 3 | 连续多少轮满足过拟合趋势才停止 |
| `--minimum-loss-change` | 0.0 | 相邻轮次损失变化超过此值才算升/降 |

学习率（1e-3）、最大长度（512）、日志间隔（5 批）在 `training_config.py` 中修改。

训练产出（`params/` 目录）：

| 文件 | 内容 |
|---|---|
| `best_classifier.pth` | 验证损失最低一轮的分类层参数 |
| `last_classifier.pth` | 最后完成一轮的分类层参数 |

> 检查点只包含分类层权重和偏置（约 8 KB），不含 BERT 和优化器状态，因此不能断点续训；重新运行会从头训练并覆盖这两个文件。

### 2. 测试集评估：`evaluate.py`

```bash
# 评估 params/ 下全部 best_classifier*.pth（当前为 CPU 版和 GPU 版两个）
uv run python evaluate.py

# 评估指定的一个或多个检查点
uv run python evaluate.py --checkpoint params/best_classifier_gpu.pth
uv run python evaluate.py --checkpoint params/best_classifier.pth params/last_classifier.pth --batch-size 16
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | `params/best_classifier*.pth` 全部 | 一个或多个要评估的分类层参数 |
| `--batch-size` | 8 | 评估批大小，只影响速度和内存 |

- 评估固定使用 CPU，CPU 与 GPU 训练出的参数在同一环境下比较。
- BERT 只加载一次，之后每个检查点只替换分类层。
- 评估多个检查点时，最后会在日志中输出对比表：

```
checkpoint                           loss  accuracy  macro_f1  confusion_matrix
best_classifier.pth                0.2305    0.9142    0.9142  [[543, 49], [54, 554]]
best_classifier_gpu.pth            0.2283    0.9092    0.9091  [[534, 58], [51, 557]]
```

**评测结果位置**：与检查点同目录、同名，扩展名换为 `.evaluation.json`：

| 评估的检查点 | 结果文件 |
|---|---|
| `params/best_classifier.pth` | `params/best_classifier.evaluation.json` |
| `params/best_classifier_gpu.pth` | `params/best_classifier_gpu.evaluation.json` |

重复评估同一检查点会覆盖原结果文件。结果字段：

| 字段 | 说明 |
|---|---|
| `checkpoint` / `split` / `sample_count` | 检查点绝对路径、数据划分（固定 `test`）、样本数 |
| `loss` / `accuracy` | 样本平均交叉熵损失、准确率 |
| `per_class` | 负面评价、正面评价各自的 precision / recall / f1 / support |
| `macro` | 两个类别的宏平均 precision / recall / f1 |
| `confusion_matrix` | 混淆矩阵，行为真实类别，列为预测类别，顺序为 [负面, 正面] |

日志中同样会输出上述汇总指标。

### 3. 单条语句预测：`predict.py`

```bash
# 判断一条后退出
uv run python predict.py --text "房间很干净，服务也热情"

# 交互模式：逐行输入，输入 q 退出
uv run python predict.py

# 使用其他检查点
uv run python predict.py --checkpoint params/best_classifier_gpu.pth --text "物流很慢"
```

输出示例（日志）：

```
模型判定：正面评价（置信度 0.9715）文本：房间很干净，服务也热情，下次还会再来
```

置信度是预测类别的 softmax 概率。预测自动选择设备（有 CUDA 用 GPU，否则 CPU），不写入任何文件。

## 文件说明

### 入口层

| 文件 | 作用 |
|---|---|
| `train.py` | 训练入口：解析参数，执行 `Trainer(config).run()` |
| `evaluate.py` | 测试集评估入口：加载模型 → `TestSetEvaluator` 评估 → 保存 JSON |
| `predict.py` | 单条语句预测入口，支持 `--text` 和交互模式 |

### 核心类

| 文件 | 类 | 作用 |
|---|---|---|
| `trainer.py` | `Trainer` | 组织完整训练流程：准备数据/模型/优化器 → 逐轮训练与验证 → 保存 best/last → 判断早停 → 恢复最佳参数 |
| `evaluator.py` | `TestSetEvaluator` | 遍历一次测试集，计算 loss、accuracy、各类别及宏平均 P/R/F1、混淆矩阵，并保存为 JSON |
| `model_loader.py` | `TrainedModelLoader`、`Prediction` | 加载本地 BERT、分词器和训练好的分类层（加载后为 eval 模式）；`predict()` 返回类别、类别名和置信度 |
| `tool.py` | `TrainingTool` | 公共静态方法：选择设备、创建优化器、批数据迁移到设备、训练一轮、验证、保存/加载检查点 |
| `post_processing_model.py` | `PostProcessingModel` | 模型结构：冻结的 BERT + 分类层；提供加载分词器和创建模型的静态方法 |
| `review_dataset.py` | `ReviewDataset` | 读取本地 ChnSentiCorp 指定划分，批量编码，创建 DataLoader |
| `training_config.py` | `TrainingConfig` | 训练配置（路径、轮数、批大小、学习率、早停阈值）及命令行参数解析 |
| `overfitting_monitor.py` | `OverfittingMonitor` | 跟踪相邻轮次的训练/验证损失，判断是否出现持续过拟合趋势 |
| `epoch_metrics.py` | `EpochMetrics` | 一轮的样本平均损失和准确率 |

### 准备与练习脚本

| 文件 | 作用 |
|---|---|
| `dataset_download.py` | 从 Hugging Face 下载 ChnSentiCorp 并保存到 `data/ChnSentiCorp/` |
| `model_tokenizer_download.py` | 下载 `bert-base-chinese` 模型和分词器到 `model/bert-base-chinese/`；文件内注释说明了浏览器手动下载方式 |
| `tokenizer_test.py` | 演示分词器如何编码和解码中文文本 |

### 配置与数据目录

| 路径 | 作用 |
|---|---|
| `pyproject.toml` / `uv.lock` | 依赖声明与锁定（torch 2.8.0、transformers、datasets、ai-starter） |
| `config.yaml` | ai-starter 日志配置（级别、日志文件） |
| `data/ChnSentiCorp/` | 本地数据集（train / validation / test） |
| `model/bert-base-chinese/` | 本地预训练模型与词表 |
| `params/` | 训练检查点和评测结果（见下） |
| `runpod/` | Runpod GPU 训练部署文件，自制镜像和官方镜像两种方案，见 [runpod/README.md](runpod/README.md) |

### `params/` 目录内容

| 文件 | 来源 |
|---|---|
| `best_classifier.pth` / `last_classifier.pth` | 本地 `train.py` 训练产出 |
| `best_classifier_gpu.pth` / `last_classifier_gpu.pth` | Runpod RTX 3080 Ti 训练产出，取回方式见 [runpod/README.md](runpod/README.md) 第 6 节 |
| `*.evaluation.json` | `evaluate.py` 对同名检查点的测试集评测结果 |
| `*.log` | 早期训练和阶段评估的运行日志 |

## 模块依赖关系

```
train.py    ──► Trainer ──► TrainingTool, ReviewDataset, PostProcessingModel,
                            OverfittingMonitor, TrainingConfig
evaluate.py ──► TrainedModelLoader + TestSetEvaluator
predict.py  ──► TrainedModelLoader.predict()
TrainedModelLoader ──► PostProcessingModel, TrainingTool.load_checkpoint
```

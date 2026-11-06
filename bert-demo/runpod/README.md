# bert-demo：Runpod GPU 训练

通用流程（工作站配置、方案 A/B 步骤、租用规则、tmux、结束与费用、常见故障）见共享仓库 [`~/workspace/runpod/README.md`](/Users/michael/workspace/runpod/README.md)。本文只记录 bert-demo 特有的内容。命令在项目根目录 `/Users/michael/workspace/ai/bert-demo` 执行，注明“Pod 内”的除外。

## 1. 项目要点

- **训练内容**：BERT 冻结，只训练 `Linear(768, 2)` 分类层；训练入口为 `Trainer`（`trainer.py`）。
- **默认参数**：30000 轮上限、batch size 50、最大长度 512，按过拟合规则提前停止。
- **检查点**：只含分类层（约 8 KB），推理仍需原始 BERT。
- **运行资源**（两种方案都需要带上）：
  - `model/bert-base-chinese/`：约 392 MiB 权重和词表
  - `data/ChnSentiCorp/`：train 9600 / validation 1200 / test 1200
  - `config.yaml`：日志配置，必须放在项目根目录
  - 相邻共享包 `../ai-starter`：只用到 `get_logger`；它的 `config.yaml` 不能上传
- **不上传**：`params/` 中的历史权重。云端每次都从原始 BERT 开始训练，不支持续训。

## 2. 目录文件

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| `build.sh`、`rent.sh`、`rent.py`、`check_cuda.py`、`wheels.sh` | 通用 | 来自共享仓库 `templates/`，用 `install-project.sh --update` 更新，不在本项目内修改 |
| `rent.json` | 配置 | Community Cloud，RTX 30/40，≥ 8 GB 显存，GPU 报价 ≤ $0.25/h，CUDA ≥ 12.8，40 GB 容器盘 + 10 GB `/workspace` 卷，排除 3070 / 3070 Ti |
| `image.conf` | 配置 | `version=0.1.2-cu128` |
| `build-contexts.conf` | 配置 | `ai_starter=../ai-starter`，由 `wheels.sh` 打成 wheel |
| `Dockerfile` / `Dockerfile.dockerignore` | 项目 | 方案 A 镜像配方和构建上下文白名单 |
| `requirements.txt` | 项目 | transformers 4.57.6、datasets 3.6.0、httpx 0.28.1、PyYAML 6.0.3；torch 由官方镜像提供，不在此固定 |
| `check_resources.py` | 项目 | 加载真实 BERT 和三份数据，完成一次前向、反向和分类层更新；默认要求 CUDA，`--device cpu` 用于本机检查 |
| `train_cuda.py` | 项目 | 强制 CUDA，调用 `Trainer`，检查点写入 `OUTPUT_DIR`（目录已有检查点时拒绝启动） |
| `train_official.sh` | 项目 | 方案 B 启动脚本：设置 HF 离线变量，默认输出到 `/workspace/bert-demo/runs/<UTC 时间戳>`，用 `.venv` 运行 `train_cuda.py`，参数直接透传 |

## 3. 方案 A：自制镜像

### 镜像内容

`Dockerfile` 依次执行：

1. `FROM runpod/pytorch:1.5.1-cu1281-torch280-ubuntu2404`（官方镜像，与方案 B 相同）
2. 设置 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`
3. 断网从 `runpod/wheels/` 安装 `requirements.txt` 和 `ai-starter` 的 wheel（bind mount 挂载，wheel 本身不写进镜像层）
4. 把 `*.py`、`pyproject.toml`、`config.yaml`、模型、数据和 3 个 runpod 脚本复制到 `/opt/bert-demo`
5. 断网运行 `check_resources.py --device cpu`
6. 沿用官方入口 `/start.sh`，由它根据 `PUBLIC_KEY` 配置 SSH

### 命令

```bash
./runpod/wheels.sh                   # 联网准备依赖，只在 requirements.txt / ai-starter 变化时运行
./runpod/build.sh                    # 断网构建 → bert-demo:0.1.2-cu128 / ghcr.io/bluemacaw-zhou/bert-demo:0.1.2-cu128
./runpod/build.sh --tag-only --push  # 上传
./runpod/rent.sh --create            # 租用
```

```bash
# Pod 内
cd /opt/bert-demo
python runpod/check_resources.py     # 必须输出 PASS 且 device=cuda
tmux new -s train
export OUTPUT_DIR="/workspace/bert-demo/runs/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR" && python -m pip freeze > "$OUTPUT_DIR/pip-freeze.txt"
python runpod/train_cuda.py 2>&1 | tee "$OUTPUT_DIR/train.log"
# 试跑：python runpod/train_cuda.py --epochs 1 --batch-size 8
```

## 4. 方案 B：官方镜像 + 上传

```bash
./runpod/rent.sh --create --image runpod/pytorch:1.5.1-cu1281-torch280-ubuntu2404 --public-image
runpodctl ssh info POD_ID
```

打包上传（在 `/Users/michael/workspace/ai` 下执行，压缩包约 400 MB）：

```bash
cd /Users/michael/workspace/ai
COPYFILE_DISABLE=1 tar -czf /tmp/bert-demo-upload.tar.gz \
  --exclude='.venv' --exclude='__pycache__' --exclude='.DS_Store' --exclude='.git' \
  --exclude='bert-demo/params' \
  --exclude='bert-demo/runpod/.rental*' --exclude='bert-demo/runpod/.rent.lock' --exclude='bert-demo/runpod/.image-ref' \
  --exclude='ai-starter/config.yaml' --exclude='ai-starter/dist' \
  bert-demo ai-starter
scp -i ~/.ssh/runpod -P POD_PORT /tmp/bert-demo-upload.tar.gz root@POD_IP:/workspace/
```

```bash
# Pod 内
cd /workspace && tar -xzf bert-demo-upload.tar.gz && rm bert-demo-upload.tar.gz
cd /workspace/bert-demo
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r runpod/requirements.txt 2>&1 | tee runpod-setup.log
.venv/bin/pip install /workspace/ai-starter 2>&1 | tee -a runpod-setup.log
.venv/bin/python runpod/check_resources.py

tmux new -s train
bash runpod/train_official.sh 2>&1 | tee train.log
# 试跑：bash runpod/train_official.sh --epochs 3 --batch-size 16
```

官方镜像使用 Python 3.12，而 `pyproject.toml` 要求 3.13。所以只安装依赖，不在 Pod 内执行 `uv sync`。项目代码在 3.12 上已实测可用，方案 A 镜像使用的也是同一个 Python。

## 5. 取回结果与评测

```bash
scp -i ~/.ssh/runpod -P POD_PORT -r root@POD_IP:/workspace/bert-demo/runs ~/Downloads/runpod-results/POD_ID/
cp ~/Downloads/runpod-results/POD_ID/runs/<时间戳>/best_classifier.pth params/best_classifier_gpu.pth
cp ~/Downloads/runpod-results/POD_ID/runs/<时间戳>/last_classifier.pth params/last_classifier_gpu.pth
python evaluate.py      # 同时评估 params/ 下全部 best_classifier*.pth，并输出 CPU / GPU 对比表
```

GPU 权重统一使用 `_gpu` 后缀，评测结果写到 `params/best_classifier_gpu.evaluation.json`。详见项目根目录 [README](../README.md)。

## 6. 运行记录

| 日期 | 方案 | GPU | 结果 |
| --- | --- | --- | --- |
| 2026-11-04 | B | RTX 3080 Ti，Community，$0.18/h | 第 148 轮早停，验证损失最低在第 27 轮（0.2628）；`params/best_classifier_gpu.pth` 测试集 accuracy 0.9092 / loss 0.2283 |

- 该次运行只保存了 tmux 回滚缓冲，没有完整训练日志，取回的结果目录已清理。
- 方案 A（2026-11-06）：基础镜像已从自建的 `runpod-base`（Python 3.13）改为官方镜像，依赖改为由 `wheels.sh` 预先准备、断网构建。在此之前构建时现场下载依赖，曾因取不到 pyarrow 的版本列表（`from versions: none`）而失败。`0.1.2-cu128` 已断网构建成功（39 个 wheel，约 107 MB；镜像 20.4 GB，比官方镜像大约 0.9 GB），构建内 CPU 自检 PASS，尚未推送。
- 删除本地旧镜像时发现 `ghcr.io/bluemacaw-zhou/bert-demo:0.1.1-cu128` 带有 repo digest，GHCR 上可能还留有这个旧版本 package。

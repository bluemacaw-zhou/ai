# Fully offline build: every input is local.
#   base image  - Runpod official PyTorch image in the local OrbStack engine
#   wheels      - runpod/wheels/, prepared once by ./runpod/wheels.sh (linux/amd64, cp312)
#   code/data   - this project
# Base provides Python 3.12, torch 2.8.0+cu128, sshd/tmux and /start.sh (SSH via PUBLIC_KEY).
ARG BASE_IMAGE=runpod/pytorch:1.5.1-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

# Resources are baked into the image; never download models/datasets at runtime.
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Install project dependencies and ai-starter from the local wheelhouse only.
# Bind mounts keep the wheels out of image layers; a missing wheel fails here instead of downloading.
RUN --network=none \
    --mount=type=bind,source=runpod/wheels,target=/tmp/wheels \
    --mount=type=bind,source=runpod/requirements.txt,target=/tmp/requirements.txt \
    python -m pip install --no-cache-dir --no-index --find-links /tmp/wheels -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --no-index --find-links /tmp/wheels --no-deps ai-starter

# Inputs live under /opt; /workspace is reserved for the mounted output volume.
WORKDIR /opt/bert-demo
COPY *.py pyproject.toml config.yaml ./
COPY runpod/check_cuda.py runpod/check_resources.py runpod/train_cuda.py ./runpod/
COPY model/bert-base-chinese/ ./model/bert-base-chinese/
COPY data/ChnSentiCorp/ ./data/ChnSentiCorp/
# Validate actual weights/tokenizer/data on CPU while building, without network.
RUN --network=none python runpod/check_resources.py --device cpu

# Keep the official ENTRYPOINT/CMD: /start.sh configures SSH from PUBLIC_KEY (rent.sh passes it).

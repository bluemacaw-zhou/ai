#!/bin/bash
# 从脚本所在仓库递归查找虚拟环境，兼容 macOS 和 Linux。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_root"

venv_dirs=()
while IFS= read -r -d '' venv_dir; do
    venv_dirs+=("$venv_dir")
done < <(find . -type d \( -name .git -o -name node_modules \) -prune -o -type d -name .venv -print0 -prune)

if [ "${#venv_dirs[@]}" -eq 0 ]; then
    echo "没有找到 .venv 虚拟环境目录"
    exit 0
fi

printf '找到以下虚拟环境目录：\n'
printf '  %s\n' "${venv_dirs[@]}"
printf '\n共找到 %s 个虚拟环境\n' "${#venv_dirs[@]}"
read -r -p "确认删除以上所有虚拟环境？(y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消操作"
    exit 0
fi

for venv_dir in "${venv_dirs[@]}"; do
    printf '正在删除: %s\n' "$venv_dir"
    rm -rf -- "$venv_dir"
done

echo "清理完成。在各独立项目目录运行 uv sync 可重新创建环境；nn-lab 使用 uv sync --all-packages。"

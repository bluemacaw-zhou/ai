#!/bin/bash
# 清理所有项目的虚拟环境目录（Git Bash）

echo "========================================"
echo "清理所有 .venv 虚拟环境目录"
echo "========================================"
echo

# 列出所有 .venv 目录
echo "找到以下虚拟环境目录："
echo
find . -type d -name ".venv" -maxdepth 2 2>/dev/null
echo

# 统计数量
count=$(find . -type d -name ".venv" -maxdepth 2 2>/dev/null | wc -l)
if [ $count -eq 0 ]; then
    echo "没有找到任何虚拟环境目录"
    exit 0
fi

echo "共找到 $count 个虚拟环境"
echo

# 确认删除
read -p "确认删除以上所有虚拟环境？(y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消操作"
    exit 0
fi

echo
echo "开始删除虚拟环境..."
echo

# 删除所有 .venv 目录
find . -type d -name ".venv" -maxdepth 2 2>/dev/null | while read dir; do
    echo "正在删除: $dir"
    rm -rf "$dir"
done

echo
echo "========================================"
echo "清理完成！"
echo "========================================"
echo
echo "提示：在各项目目录运行 'uv sync' 可重新创建虚拟环境"
echo

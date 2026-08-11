#!/bin/bash
# 将本项目分发到两只鲸鱼（blue-whale / sperm-whale）的相同路径
# 用法: bash scripts/sync_to_whales.sh
set -euo pipefail

SRC="/home/lynchpin/Projects/Drug-The-Whole-Genome/"
DEST="/home/lynchpin/Projects/Drug-The-Whole-Genome/"
WHALES=("blue-whale.local" "sperm-whale.local")

for w in "${WHALES[@]}"; do
    echo "=== [$w] 确保目标目录存在 ==="
    ssh "$w" "mkdir -p '$DEST'"

    echo "=== [$w] rsync 分发中 ... ==="
    rsync -a --info=progress2 \
        --exclude='.git' \
        --exclude='logs/' \
        --exclude='__pycache__/' \
        --exclude='resources/' \
        --exclude='output/' \
        "$SRC" "$w:$DEST"
    echo "=== [$w] 完成 ==="
done

echo "全部完成"

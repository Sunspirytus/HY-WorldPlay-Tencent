#!/bin/bash

# HY-WorldPlay Gradio UI 启动脚本
# 使用 micromamba -n hunyuan-worldplay 环境

ENV_NAME="hunyuan-worldplay"

echo "🎮 启动 HY-WorldPlay 世界模型视频生成界面..."
echo ""

# 检查micromamba
if ! command -v micromamba &> /dev/null; then
    echo "❌ 错误：未找到 micromamba"
    exit 1
fi

# 检查环境是否存在
if ! micromamba env list | grep -q "$ENV_NAME"; then
    echo "❌ 错误：环境 $ENV_NAME 不存在"
    echo "可用环境:"
    micromamba env list
    exit 1
fi

echo "✅ 使用环境: $ENV_NAME"
echo ""
echo "🌐 启动 Gradio 服务..."
echo "   访问地址: http://localhost:7860"
echo ""

# 启动Gradio应用
micromamba run -n "$ENV_NAME" python3 gradio_app.py

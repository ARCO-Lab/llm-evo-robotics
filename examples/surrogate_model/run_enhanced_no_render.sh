#!/bin/bash
# 强制无渲染模式运行脚本

echo "🚫 强制无渲染模式启动"
echo "设置环境变量..."

# 设置强制无渲染环境变量
export FORCE_NO_RENDER=1
export SDL_VIDEODRIVER=dummy
export DISPLAY=""

echo "✅ 环境变量已设置"
echo "   FORCE_NO_RENDER=1"
echo "   SDL_VIDEODRIVER=dummy"
echo "   DISPLAY=\"\""

# 激活虚拟环境
source /home/xli149/Documents/repos/RoboGrammar/venv/bin/activate

# 运行训练
echo "🚀 启动无渲染训练..."
python enhanced_multi_network_extractor_backup.py \
    --experiment-name "$1" \
    --mode basic \
    --training-steps "${2:-500}" \
    --num-generations "${3:-1}" \
    --individuals-per-generation "${4:-2}" \
    --silent-mode

echo "🎉 无渲染训练完成！"

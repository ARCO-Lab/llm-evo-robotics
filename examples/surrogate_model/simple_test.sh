#!/bin/bash
echo "🧪 测试新的attention网络分离功能"
echo "=============================="

cd /home/xli149/Documents/repos/test_robo2/examples/surrogate_model

echo "🚀 运行测试..."
timeout 45 python enhanced_multi_network_extractor.py \
    --experiment-name test_attention_separation \
    --mode basic \
    --training-steps 600 \
    --num-generations 1 \
    --individuals-per-generation 1

echo "✅ 测试完成"

# 检查结果
LOG_DIR="enhanced_multi_network_logs/test_attention_separation_multi_network_loss"
if [ -d "$LOG_DIR" ]; then
    echo ""
    echo "📁 生成的文件:"
    ls -la "$LOG_DIR"
    
    echo ""
    echo "📊 attention_losses.csv 字段:"
    if [ -f "$LOG_DIR/attention_losses.csv" ]; then
        head -1 "$LOG_DIR/attention_losses.csv"
        
        echo ""
        echo "📋 检查新字段:"
        header=$(head -1 "$LOG_DIR/attention_losses.csv")
        
        for field in "attention_actor_param_mean" "attention_critic_param_mean" "robot_num_joints" "J0_activity" "L0_length"; do
            if echo "$header" | grep -q "$field"; then
                echo "   ✅ 包含: $field"
            else
                echo "   ❌ 缺少: $field"
            fi
        done
    else
        echo "❌ 未找到 attention_losses.csv"
    fi
else
    echo "❌ 未找到日志目录: $LOG_DIR"
fi


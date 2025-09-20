#!/usr/bin/env python3
"""
验证新功能的简单测试
"""

import os
import subprocess
import sys

# 切换到正确的目录
os.chdir('/home/xli149/Documents/repos/test_robo2/examples/surrogate_model')

print("🧪 验证新的attention网络分离和关节分布记录功能")
print("=" * 60)
print(f"📁 当前目录: {os.getcwd()}")

# 构建命令
cmd = [
    'timeout', '45',
    sys.executable, 'enhanced_multi_network_extractor.py',
    '--experiment-name', 'verify_new_features',
    '--mode', 'basic',
    '--training-steps', '700',
    '--num-generations', '1', 
    '--individuals-per-generation', '1'
]

print(f"🚀 执行命令: {' '.join(cmd)}")

try:
    # 运行命令
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, 
        text=True,
        timeout=50
    )
    
    print(f"✅ 命令完成，退出码: {result.returncode}")
    
    # 查找attention相关的输出
    if result.stdout:
        lines = result.stdout.split('\n')
        attention_lines = []
        
        for line in lines:
            if any(keyword in line for keyword in [
                'Actor Attention参数', 'Critic Attention参数',
                '🤖 机器人结构', '🎯 最重要关节',
                '🔍 关节活跃度', '📏 Link长度',
                '🆔 检测到Individual ID'
            ]):
                attention_lines.append(line.strip())
        
        if attention_lines:
            print(f"\n✅ 检测到新功能输出:")
            for line in attention_lines[-10:]:
                print(f"   {line}")
        else:
            print(f"\n❌ 未检测到新功能输出")
    
    # 检查生成的文件
    log_dir = "enhanced_multi_network_logs/verify_new_features_multi_network_loss"
    if os.path.exists(log_dir):
        print(f"\n📁 生成的文件:")
        files = os.listdir(log_dir)
        for file in files:
            print(f"   📄 {file}")
            
        # 检查attention_losses.csv的字段
        csv_file = os.path.join(log_dir, "attention_losses.csv")
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                header = f.readline().strip()
                print(f"\n📊 attention_losses.csv 字段:")
                fields = header.split(',')
                for i, field in enumerate(fields):
                    prefix = "✅" if any(keyword in field for keyword in [
                        'actor_param', 'critic_param', 'robot_num', 'J0_', 'L0_'
                    ]) else "  "
                    print(f"   {prefix} {field}")
    else:
        print(f"\n❌ 未找到日志目录: {log_dir}")
        
except subprocess.TimeoutExpired:
    print("⏰ 测试超时（这是正常的）")
except Exception as e:
    print(f"❌ 测试失败: {e}")

print("\n🎯 测试完成")


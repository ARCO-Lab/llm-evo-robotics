#!/usr/bin/env python3

import os
import sys

# 设置工作目录
os.chdir('/home/xli149/Documents/repos/test_robo2/examples/surrogate_model')

print("🧪 快速验证新功能")
print("当前目录:", os.getcwd())

# 直接运行
exit_code = os.system('timeout 40 python enhanced_multi_network_extractor.py --experiment-name quick_verify --mode basic --training-steps 500 --num-generations 1 --individuals-per-generation 1')

print(f"退出码: {exit_code}")

# 检查结果
log_dir = "enhanced_multi_network_logs/quick_verify_multi_network_loss"
if os.path.exists(log_dir):
    print("\n生成的文件:")
    files = os.listdir(log_dir)
    for f in files:
        print(f"  {f}")
    
    # 检查CSV
    csv_path = os.path.join(log_dir, "attention_losses.csv") 
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as file:
            header = file.readline().strip()
            print(f"\nCSV字段: {header}")
            
            # 检查关键字段
            if 'attention_actor_param_mean' in header:
                print("✅ 包含Actor参数字段")
            if 'attention_critic_param_mean' in header:
                print("✅ 包含Critic参数字段")
            if 'robot_num_joints' in header:
                print("✅ 包含机器人结构字段")
else:
    print("❌ 未找到日志目录")


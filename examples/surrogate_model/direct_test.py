#!/usr/bin/env python3
import os
import sys

# 直接运行测试
if __name__ == "__main__":
    print("🧪 直接测试新功能")
    
    # 运行命令
    cmd = "timeout 45 python enhanced_multi_network_extractor.py --experiment-name test_new_features --mode basic --training-steps 600 --num-generations 1 --individuals-per-generation 1"
    
    print(f"🚀 执行: {cmd}")
    exit_code = os.system(cmd)
    
    print(f"✅ 完成，退出码: {exit_code}")
    
    # 检查结果
    log_dir = "enhanced_multi_network_logs/test_new_features_multi_network_loss"
    if os.path.exists(log_dir):
        print(f"\n📁 生成的文件:")
        for file in os.listdir(log_dir):
            print(f"   {file}")
            
        # 检查attention_losses.csv
        csv_file = os.path.join(log_dir, "attention_losses.csv")
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                header = f.readline().strip()
                print(f"\n📊 attention_losses.csv 字段:")
                print(f"   {header}")
                
                # 检查新字段
                new_fields = ['attention_actor_param_mean', 'attention_critic_param_mean', 'robot_num_joints']
                for field in new_fields:
                    if field in header:
                        print(f"   ✅ 包含: {field}")
                    else:
                        print(f"   ❌ 缺少: {field}")
    else:
        print(f"❌ 未找到日志目录")


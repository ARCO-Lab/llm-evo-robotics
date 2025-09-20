#!/usr/bin/env python3
"""
测试新的分离attention网络和关节分布记录功能
"""

import subprocess
import sys
import os
import time

def run_test():
    print("🧪 测试新的分离attention网络记录功能")
    print("=" * 60)
    
    # 运行测试
    cmd = [
        'timeout', '45',
        'python', 'enhanced_multi_network_extractor.py',
        '--experiment-name', 'test_new_attention_features',
        '--mode', 'basic', 
        '--training-steps', '800',
        '--num-generations', '1',
        '--individuals-per-generation', '1'
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    print("⏳ 等待训练完成...")
    
    try:
        result = subprocess.run(cmd, cwd='.', shell=False)
        print(f"✅ 训练完成，退出码: {result.returncode}")
        
        # 检查生成的文件
        log_dir = "enhanced_multi_network_logs/test_new_attention_features_multi_network_loss"
        
        if os.path.exists(log_dir):
            print(f"\n📁 检查生成的文件:")
            
            # 检查attention_losses.csv
            attention_csv = os.path.join(log_dir, "attention_losses.csv")
            if os.path.exists(attention_csv):
                print(f"📊 attention_losses.csv:")
                with open(attention_csv, 'r') as f:
                    header = f.readline().strip()
                    print(f"   字段: {header}")
                    
                    # 检查是否包含新字段
                    new_fields = [
                        'attention_actor_param_mean', 'attention_critic_param_mean',
                        'robot_num_joints', 'J0_activity', 'L0_length'
                    ]
                    
                    found_new_fields = []
                    for field in new_fields:
                        if field in header:
                            found_new_fields.append(field)
                    
                    if found_new_fields:
                        print(f"   ✅ 包含新字段: {found_new_fields}")
                    else:
                        print(f"   ❌ 缺少新字段")
                    
                    # 显示第一行数据
                    first_line = f.readline().strip()
                    if first_line:
                        print(f"   示例数据: {first_line[:100]}...")
            else:
                print(f"❌ 未找到attention_losses.csv")
        else:
            print(f"❌ 未找到日志目录: {log_dir}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    run_test()


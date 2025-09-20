#!/usr/bin/env python3
"""
最终验证测试 - 检查所有新功能
"""

import os
import sys
import subprocess
import time

def main():
    print("🎯 最终验证测试 - 分离attention网络和关节分布记录")
    print("=" * 70)
    
    # 确保在正确目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 运行测试
    experiment_name = "final_verification"
    cmd = [
        'timeout', '40',
        sys.executable, 'enhanced_multi_network_extractor.py',
        '--experiment-name', experiment_name,
        '--mode', 'basic',
        '--training-steps', '600',
        '--num-generations', '1',
        '--individuals-per-generation', '1'
    ]
    
    print(f"🚀 运行: {' '.join(cmd)}")
    
    try:
        # 运行命令并捕获输出
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45
        )
        
        print(f"✅ 测试完成，退出码: {result.returncode}")
        
        # 分析输出
        output = result.stdout
        if output:
            lines = output.split('\n')
            
            # 查找关键信息
            key_info = []
            for line in lines:
                if any(keyword in line for keyword in [
                    '🆔 检测到Individual ID',
                    'Actor Attention参数', 'Critic Attention参数',
                    '🤖 机器人结构', '🎯 最重要关节',
                    '🔍 关节活跃度', '📏 Link长度',
                    '🏆 关节使用排名'
                ]):
                    key_info.append(line.strip())
            
            if key_info:
                print(f"\n✅ 检测到的关键信息:")
                for info in key_info[-15:]:  # 显示最后15行
                    print(f"   {info}")
            else:
                print(f"\n❌ 未检测到关键信息")
        
        # 检查生成的文件
        log_dir = f"enhanced_multi_network_logs/{experiment_name}_multi_network_loss"
        
        if os.path.exists(log_dir):
            print(f"\n📁 生成的文件:")
            files = os.listdir(log_dir)
            for file in sorted(files):
                print(f"   📄 {file}")
            
            # 详细检查attention_losses.csv
            csv_file = os.path.join(log_dir, "attention_losses.csv")
            if os.path.exists(csv_file):
                print(f"\n📊 attention_losses.csv 分析:")
                
                with open(csv_file, 'r') as f:
                    header = f.readline().strip()
                    fields = header.split(',')
                    
                    print(f"   总字段数: {len(fields)}")
                    
                    # 检查新字段
                    new_fields = {
                        'attention_actor_param_mean': '✅' if 'attention_actor_param_mean' in header else '❌',
                        'attention_critic_param_mean': '✅' if 'attention_critic_param_mean' in header else '❌',
                        'robot_num_joints': '✅' if 'robot_num_joints' in header else '❌',
                        'most_important_joint': '✅' if 'most_important_joint' in header else '❌',
                        'J0_activity': '✅' if 'J0_activity' in header else '❌',
                        'L0_length': '✅' if 'L0_length' in header else '❌',
                    }
                    
                    print(f"   新功能字段检查:")
                    for field, status in new_fields.items():
                        print(f"     {status} {field}")
                    
                    # 显示第一行数据（如果有）
                    first_line = f.readline().strip()
                    if first_line:
                        print(f"\n   📋 示例数据:")
                        values = first_line.split(',')
                        for i, (field, value) in enumerate(zip(fields[:10], values[:10])):
                            print(f"     {field}: {value}")
                        if len(fields) > 10:
                            print(f"     ... 还有 {len(fields)-10} 个字段")
            else:
                print(f"\n❌ 未找到 attention_losses.csv")
        else:
            print(f"\n❌ 未找到日志目录: {log_dir}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时（正常现象）")
        print("📊 检查是否生成了部分数据...")
        
        # 即使超时也检查文件
        log_dir = f"enhanced_multi_network_logs/{experiment_name}_multi_network_loss"
        if os.path.exists(log_dir):
            files = os.listdir(log_dir)
            print(f"   📁 生成了 {len(files)} 个文件")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()


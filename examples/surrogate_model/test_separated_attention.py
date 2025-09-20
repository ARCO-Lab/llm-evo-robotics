#!/usr/bin/env python3
"""
测试分离的Actor和Critic attention网络记录
"""

import subprocess
import sys
import os
import time

def test_separated_attention():
    """测试分离的attention网络记录"""
    print("🧪 测试分离的Actor和Critic attention网络记录")
    print("=" * 60)
    
    # 运行一个短时间的训练测试
    cmd = [
        sys.executable, 'enhanced_multi_network_extractor.py',
        '--experiment-name', 'test_separated_attention',
        '--mode', 'basic',
        '--training-steps', '800',
        '--num-generations', '1',
        '--individuals-per-generation', '1'
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    
    try:
        # 运行测试
        result = subprocess.run(
            cmd,
            timeout=45,  # 45秒超时
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print(f"✅ 测试完成，退出码: {result.returncode}")
        
        # 检查生成的文件
        log_dir = "enhanced_multi_network_logs/test_separated_attention_multi_network_loss"
        if os.path.exists(log_dir):
            print(f"\n📁 生成的文件:")
            for file in os.listdir(log_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(log_dir, file)
                    print(f"   📊 {file}")
                    
                    # 查看CSV文件的header
                    with open(file_path, 'r') as f:
                        header = f.readline().strip()
                        print(f"      字段: {header}")
                        
                        # 查看第一行数据
                        first_line = f.readline().strip()
                        if first_line:
                            print(f"      示例: {first_line[:100]}...")
                    print()
        else:
            print(f"❌ 未找到日志目录: {log_dir}")
            
        # 显示部分输出
        if result.stdout:
            print(f"\n📋 部分训练输出:")
            lines = result.stdout.split('\n')
            attention_lines = [line for line in lines if 'Attention' in line or '🤖 机器人' in line or '🎯 最重要' in line]
            for line in attention_lines[-10:]:  # 显示最后10行attention相关信息
                print(f"   {line}")
                
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时，但这是正常的")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_separated_attention()


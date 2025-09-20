#!/usr/bin/env python3
"""
快速测试分离的attention网络记录
"""

import os
import subprocess
import sys

def main():
    print("🧪 快速测试分离的attention网络记录")
    print("=" * 50)
    
    # 运行短时间测试
    cmd = [
        'timeout', '30',
        sys.executable, 'enhanced_multi_network_extractor.py',
        '--experiment-name', 'quick_separated_test',
        '--mode', 'basic',
        '--training-steps', '500',
        '--num-generations', '1',
        '--individuals-per-generation', '1'
    ]
    
    print(f"🚀 运行: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"✅ 测试完成 (退出码: {result.returncode})")
        
        # 检查输出中是否包含分离的attention信息
        output_lines = result.stdout.split('\n')
        attention_lines = []
        
        for line in output_lines:
            if any(keyword in line for keyword in [
                'Actor Attention参数', 'Critic Attention参数', 
                '🤖 机器人结构', '🎯 最重要关节',
                '🔍 关节活跃度', '📏 Link长度'
            ]):
                attention_lines.append(line.strip())
        
        if attention_lines:
            print(f"\n✅ 检测到分离的attention信息:")
            for line in attention_lines[-8:]:  # 显示最后8行
                print(f"   {line}")
        else:
            print(f"\n❌ 未检测到分离的attention信息")
            
        # 检查生成的文件
        log_dir = "enhanced_multi_network_logs/quick_separated_test_multi_network_loss"
        if os.path.exists(log_dir):
            print(f"\n📁 生成的文件:")
            for file in os.listdir(log_dir):
                if file.endswith('.csv'):
                    print(f"   📊 {file}")
                    # 查看文件头
                    with open(os.path.join(log_dir, file), 'r') as f:
                        header = f.readline().strip()
                        if 'attention_actor' in header or 'attention_critic' in header:
                            print(f"      ✅ 包含分离的attention字段")
                        else:
                            print(f"      ❌ 缺少分离的attention字段")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()


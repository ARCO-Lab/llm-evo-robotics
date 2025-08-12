#!/usr/bin/env python3
"""
简化模型测试脚本 - 直接使用现有的test_best_model.py
"""

import os
import sys
import subprocess

def main():
    # 设置模型路径
    model_path = "../../trained_models/reacher2d/test/08-10-2025-18-54-38/best_models/final_model_step_19999.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请检查路径是否正确")
        return
    
    print(f"✅ 找到模型文件: {model_path}")
    print(f"📊 开始评估模型...")
    
    # 使用现有的test_best_model.py脚本
    try:
        cmd = [
            sys.executable,  # 使用当前Python解释器
            "test_best_model.py", 
            "--model-path", model_path,
            "--episodes", "5"
        ]
        
        print(f"🚀 运行命令: {' '.join(cmd)}")
        
        # 运行测试脚本
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("📝 测试输出:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ 错误信息:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ 测试完成")
        else:
            print(f"❌ 测试失败，退出码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 运行测试时发生错误: {e}")

if __name__ == "__main__":
    main() 
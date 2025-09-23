#!/usr/bin/env python3
"""
渲染控制测试脚本
验证不同的渲染控制方法
"""

import os
import sys
import subprocess
import time

def test_render_control():
    """测试渲染控制的不同方法"""
    
    print("🎨 渲染控制测试")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        {
            "name": "无渲染模式（环境变量）",
            "env_vars": {"FORCE_NO_RENDER": "1"},
            "args": ["--experiment-name", "test_no_render_env"]
        },
        {
            "name": "启用渲染模式（参数）",
            "env_vars": {},
            "args": ["--experiment-name", "test_with_render_param", "--enable-rendering"]
        },
        {
            "name": "强制渲染模式（环境变量）",
            "env_vars": {"FORCE_RENDER": "1"},
            "args": ["--experiment-name", "test_force_render_env"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 测试 {i}: {test_case['name']}")
        print("-" * 30)
        
        # 设置环境变量
        env = os.environ.copy()
        for key, value in test_case['env_vars'].items():
            env[key] = value
            print(f"   设置环境变量: {key}={value}")
        
        # 构建命令
        cmd = [
            sys.executable,
            "enhanced_multi_network_extractor_backup.py",
            "--mode", "basic",
            "--training-steps", "100",  # 短时间测试
            "--num-generations", "1",
            "--individuals-per-generation", "1"
        ] + test_case['args']
        
        print(f"   命令: {' '.join(cmd)}")
        
        # 启动进程
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            
            print(f"   ✅ 进程已启动 (PID: {process.pid})")
            
            # 等待5秒
            time.sleep(5)
            
            # 检查子进程
            child_processes = []
            try:
                result = subprocess.run(['pgrep', '-f', 'enhanced_train'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    child_pids = result.stdout.strip().split('\n')
                    for pid in child_pids:
                        try:
                            proc_info = subprocess.run(['ps', '-p', pid, '-o', 'args='], 
                                                     capture_output=True, text=True)
                            if proc_info.stdout.strip():
                                child_processes.append(proc_info.stdout.strip())
                        except:
                            pass
            except:
                pass
            
            # 分析结果
            if child_processes:
                print(f"   🔍 检测到子进程:")
                for proc in child_processes:
                    if '--render' in proc:
                        print(f"     ✅ 启用渲染: {proc[:100]}...")
                    elif '--no-render' in proc:
                        print(f"     🚫 禁用渲染: {proc[:100]}...")
                    else:
                        print(f"     ❓ 未知状态: {proc[:100]}...")
            else:
                print(f"   ⚠️ 未检测到子进程")
            
            # 终止进程
            process.terminate()
            process.wait(timeout=5)
            print(f"   🛑 进程已终止")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
        
        print()
    
    print("🎉 渲染控制测试完成！")

if __name__ == "__main__":
    test_render_control()

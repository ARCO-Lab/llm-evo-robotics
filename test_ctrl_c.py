#!/usr/bin/env python3
"""
测试Ctrl+C是否能正常中断MAP-Elites训练
"""
import subprocess
import sys
import time
import signal

def test_ctrl_c():
    print("🧪 开始测试Ctrl+C中断功能...")
    print("⏰ 将在5秒后启动训练，然后发送SIGINT信号...")
    
    # 激活环境并启动训练
    cmd = [
        "bash", "-c", 
        "source /home/xli149/Documents/repos/RoboGrammar/venv/bin/activate && python examples/surrogate_model/map_elites/map_elites_trainer.py --train"
    ]
    
    try:
        # 启动进程
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"📋 训练进程启动，PID: {process.pid}")
        
        # 等待5秒让训练开始
        time.sleep(5)
        
        # 发送SIGINT信号（相当于Ctrl+C）
        print("🔥 发送SIGINT信号（模拟Ctrl+C）...")
        process.send_signal(signal.SIGINT)
        
        # 等待进程结束，最多等待10秒
        try:
            stdout, stderr = process.communicate(timeout=10)
            print(f"✅ 进程已结束，退出码: {process.returncode}")
            print("📝 输出的最后几行:")
            lines = stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        except subprocess.TimeoutExpired:
            print("⚠️ 进程在10秒内未结束，强制终止...")
            process.kill()
            stdout, stderr = process.communicate()
            print("❌ 需要强制终止，说明Ctrl+C处理仍有问题")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_ctrl_c()
    if success:
        print("🎉 Ctrl+C测试通过！")
    else:
        print("💥 Ctrl+C测试失败！")
    sys.exit(0 if success else 1)

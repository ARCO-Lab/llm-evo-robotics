#!/usr/bin/env python3
"""
可靠的MAP-Elites训练 + 损失记录启动器
使用简化的、不依赖复杂环境的损失记录系统
"""

import os
import sys
import subprocess
import time
import signal
import threading
from datetime import datetime

def start_simple_loss_monitor(experiment_name):
    """启动简化损失监控器"""
    try:
        from simple_loss_monitor import start_simple_loss_monitor as start_monitor
        monitor = start_monitor(experiment_name)
        print(f"✅ 简化损失监控器已启动: {experiment_name}")
        return monitor
    except Exception as e:
        print(f"❌ 启动简化损失监控器失败: {e}")
        return None

def run_training_with_loss_logging(experiment_name, mode='basic', extra_args=None):
    """运行训练并启动损失记录"""
    print(f"🚀 启动MAP-Elites训练 + 可靠损失记录")
    print(f"   实验名称: {experiment_name}")
    print(f"   训练模式: {mode}")
    print("=" * 60)
    
    # 1. 启动简化损失监控器
    print("📊 启动损失监控器...")
    monitor = start_simple_loss_monitor(experiment_name)
    
    # 2. 设置环境变量
    os.environ['LOSS_EXPERIMENT_NAME'] = experiment_name
    print(f"🔗 设置实验名称环境变量: {experiment_name}")
    
    # 3. 构建训练命令（使用相对路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, 'map_elites_with_loss_logger.py')
    
    cmd = [
        sys.executable, 
        training_script,
        '--mode', mode,
        '--experiment-name', experiment_name
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"🎯 启动训练命令: {' '.join(cmd)}")
    
    # 4. 启动训练进程
    training_process = None
    try:
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ 训练进程已启动 (PID: {training_process.pid})")
        
        # 5. 实时显示训练输出并监控损失
        monitor_thread = threading.Thread(
            target=monitor_loss_files,
            args=(experiment_name,),
            daemon=True
        )
        monitor_thread.start()
        
        # 6. 读取训练输出
        for line in training_process.stdout:
            print(f"[训练] {line.rstrip()}")
            
        # 等待训练完成
        training_process.wait()
        
        if training_process.returncode == 0:
            print("🎉 训练完成！")
        else:
            print(f"⚠️ 训练结束，返回码: {training_process.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 接收到中断信号...")
        if training_process:
            training_process.terminate()
            training_process.wait()
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
    finally:
        # 7. 停止损失监控器
        if monitor:
            print("🛑 停止损失监控器...")
            from simple_loss_monitor import stop_simple_loss_monitor
            stop_simple_loss_monitor()
        
        print("🧹 清理完成")

def monitor_loss_files(experiment_name):
    """监控损失文件生成"""
    print(f"🔍 开始监控损失文件: {experiment_name}")
    
    simple_log_dir = f"simple_loss_logs/{experiment_name}_loss_log"
    
    while True:
        try:
            if os.path.exists(simple_log_dir):
                files = os.listdir(simple_log_dir)
                csv_files = [f for f in files if f.endswith('.csv')]
                
                if csv_files:
                    total_size = 0
                    for csv_file in csv_files:
                        file_path = os.path.join(simple_log_dir, csv_file)
                        if os.path.exists(file_path):
                            size = os.path.getsize(file_path)
                            total_size += size
                    
                    if total_size > 0:
                        print(f"📊 损失数据更新: {len(csv_files)} 个文件, 总大小: {total_size} 字节")
                
            time.sleep(10)  # 每10秒检查一次
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(5)
    
    print("🔍 损失文件监控结束")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可靠的MAP-Elites训练 + 损失记录')
    parser.add_argument('--experiment-name', type=str, required=True, help='实验名称')
    parser.add_argument('--mode', type=str, default='basic', 
                       choices=['basic', 'advanced', 'multiprocess', 'shared-ppo', 'custom'],
                       help='训练模式')
    parser.add_argument('--training-steps-per-individual', type=int, help='每个个体训练步数')
    parser.add_argument('--num-generations', type=int, help='进化代数')
    parser.add_argument('--enable-rendering', action='store_true', help='启用渲染')
    
    args = parser.parse_args()
    
    # 构建额外参数
    extra_args = []
    if args.training_steps_per_individual:
        extra_args.extend(['--training-steps-per-individual', str(args.training_steps_per_individual)])
    if args.num_generations:
        extra_args.extend(['--num-generations', str(args.num_generations)])
    if args.enable_rendering:
        extra_args.append('--enable-rendering')
    
    # 运行训练
    run_training_with_loss_logging(args.experiment_name, args.mode, extra_args)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
MAP-Elites训练器 + 损失记录器启动脚本
这是一个简单的Python启动脚本，用于同时运行MAP-Elites训练和损失记录器
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from datetime import datetime

def print_colored(message, color='blue'):
    """打印带颜色的消息"""
    colors = {
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'blue': '\033[0;34m',
        'purple': '\033[0;35m',
        'cyan': '\033[0;36m',
        'white': '\033[0;37m',
        'reset': '\033[0m'
    }
    
    color_code = colors.get(color, colors['blue'])
    reset_code = colors['reset']
    print(f"{color_code}{message}{reset_code}")

def check_dependencies():
    """检查Python依赖"""
    print_colored("🔍 检查Python依赖...", 'blue')
    
    required_packages = ['torch', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_colored(f"  ✅ {package}", 'green')
        except ImportError:
            missing_packages.append(package)
            print_colored(f"  ❌ {package}", 'red')
    
    if missing_packages:
        print_colored(f"缺少必要包: {', '.join(missing_packages)}", 'red')
        print_colored("请安装缺少的包后重试", 'red')
        return False
    
    print_colored("依赖检查通过", 'green')
    return True

def show_menu():
    """显示训练模式选择菜单"""
    print_colored("\n🎯 MAP-Elites训练器 + 损失记录器", 'purple')
    print_colored("=" * 50, 'purple')
    print_colored("请选择训练模式:", 'blue')
    print_colored("1. 基础训练 (basic)", 'white')
    print_colored("2. 高级训练 (advanced)", 'white')
    print_colored("3. 多进程训练 (multiprocess)", 'white')
    print_colored("4. 共享PPO训练 (shared-ppo)", 'white')
    print_colored("5. 自定义训练 (custom)", 'white')
    print_colored("6. 测试损失记录器", 'white')
    print_colored("0. 退出", 'white')

def get_user_choice():
    """获取用户选择"""
    while True:
        try:
            choice = input("\n请输入选择 (0-6): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print_colored("无效选择，请输入0-6之间的数字", 'red')
        except KeyboardInterrupt:
            print_colored("\n用户取消", 'yellow')
            return '0'

def run_training_mode(mode, extra_args=None):
    """运行指定的训练模式"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_script = os.path.join(script_dir, 'map_elites_with_loss_logger.py')
    
    if not os.path.exists(python_script):
        print_colored(f"找不到训练脚本: {python_script}", 'red')
        return False
    
    # 构建命令
    cmd = [sys.executable, python_script, '--mode', mode]
    
    # 添加实验名称
    experiment_name = f"map_elites_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd.extend(['--experiment-name', experiment_name])
    
    # 添加额外参数
    if extra_args:
        cmd.extend(extra_args)
    
    print_colored(f"🚀 启动{mode}训练模式...", 'green')
    print_colored(f"实验名称: {experiment_name}", 'blue')
    print_colored(f"执行命令: {' '.join(cmd)}", 'cyan')
    
    try:
        # 运行训练
        process = subprocess.run(cmd, check=True)
        print_colored("✅ 训练完成", 'green')
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ 训练失败: {e}", 'red')
        return False
    except KeyboardInterrupt:
        print_colored("⚠️ 训练被用户中断", 'yellow')
        return False

def test_loss_logger():
    """测试损失记录器"""
    print_colored("🧪 测试损失记录器...", 'blue')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(script_dir, 'network_loss_logger.py')
    
    if not os.path.exists(test_script):
        print_colored(f"找不到测试脚本: {test_script}", 'red')
        return False
    
    try:
        subprocess.run([sys.executable, test_script], check=True, timeout=30)
        print_colored("✅ 损失记录器测试完成", 'green')
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ 测试失败: {e}", 'red')
        return False
    except subprocess.TimeoutExpired:
        print_colored("⏰ 测试超时", 'yellow')
        return False
    except KeyboardInterrupt:
        print_colored("⚠️ 测试被用户中断", 'yellow')
        return False

def get_custom_training_params():
    """获取自定义训练参数"""
    print_colored("\n⚙️ 自定义训练参数配置", 'blue')
    
    params = []
    
    # 基本参数
    try:
        num_generations = input("进化代数 (默认20): ").strip()
        if num_generations:
            params.extend(['--num-generations', num_generations])
            
        training_steps = input("每个个体训练步数 (默认2000): ").strip()
        if training_steps:
            params.extend(['--training-steps-per-individual', training_steps])
            
        initial_pop = input("初始种群大小 (默认10): ").strip()
        if initial_pop:
            params.extend(['--num-initial-random', initial_pop])
            
        # 是否启用渲染
        enable_render = input("启用环境渲染? (y/n, 默认n): ").strip().lower()
        if enable_render == 'y':
            params.append('--enable-rendering')
            
        # 是否使用遗传算法fitness
        use_genetic = input("使用遗传算法fitness? (y/n, 默认n): ").strip().lower()
        if use_genetic == 'y':
            params.append('--use-genetic-fitness')
            
        return params
        
    except KeyboardInterrupt:
        print_colored("\n用户取消配置", 'yellow')
        return None

def main():
    """主函数"""
    print_colored("🎯 MAP-Elites训练器 + 损失记录器启动器", 'purple')
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 主循环
    while True:
        show_menu()
        choice = get_user_choice()
        
        if choice == '0':
            print_colored("👋 再见!", 'green')
            break
        elif choice == '1':
            run_training_mode('basic')
        elif choice == '2':
            run_training_mode('advanced')
        elif choice == '3':
            run_training_mode('multiprocess')
        elif choice == '4':
            run_training_mode('shared-ppo')
        elif choice == '5':
            # 自定义训练
            custom_params = get_custom_training_params()
            if custom_params is not None:
                run_training_mode('custom', custom_params)
        elif choice == '6':
            test_loss_logger()
        
        # 询问是否继续
        if choice != '0':
            continue_choice = input("\n是否继续? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print_colored("👋 再见!", 'green')
                break
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_colored("\n👋 程序被用户中断", 'yellow')
        sys.exit(130)

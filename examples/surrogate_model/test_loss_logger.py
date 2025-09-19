#!/usr/bin/env python3
"""
损失记录器测试脚本
用于验证损失记录系统是否正常工作
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试1: 基本功能测试")
    print("=" * 40)
    
    try:
        from loss_logger_interface import start_loss_logging, log_network_loss, stop_loss_logging
        
        # 启动损失记录器
        print("📊 启动损失记录器...")
        logger = start_loss_logging(
            experiment_name="test_basic_functionality",
            networks=['attention', 'ppo', 'gnn'],
            update_interval=3.0
        )
        
        if logger is None:
            print("❌ 损失记录器启动失败")
            return False
            
        print(f"✅ 损失记录器已启动")
        print(f"   日志目录: {logger.get_log_dir()}")
        
        # 生成测试数据
        print("📈 生成测试损失数据...")
        for step in range(100):
            # Attention网络损失
            attention_loss = {
                'attention_loss': max(0.1, 2.0 - step*0.01 + np.random.normal(0, 0.1)),
                'attention_accuracy': min(1.0, 0.3 + step * 0.005 + np.random.normal(0, 0.02))
            }
            log_network_loss('attention', step, attention_loss)
            
            # PPO网络损失
            ppo_loss = {
                'actor_loss': max(0.01, 1.5 - step*0.008 + np.random.normal(0, 0.08)),
                'critic_loss': max(0.01, 1.2 - step*0.006 + np.random.normal(0, 0.06)),
                'entropy': max(0.001, 0.8 - step*0.003 + np.random.normal(0, 0.02))
            }
            log_network_loss('ppo', step, ppo_loss)
            
            # GNN网络损失
            gnn_loss = {
                'gnn_loss': max(0.1, 3.0 - step*0.012 + np.random.normal(0, 0.15)),
                'node_accuracy': min(1.0, 0.25 + step * 0.007 + np.random.normal(0, 0.01))
            }
            log_network_loss('gnn', step, gnn_loss)
            
            if step % 20 == 0:
                print(f"   步数: {step}")
            
            time.sleep(0.02)
        
        print("✅ 测试数据生成完成")
        
        # 等待图表生成
        print("⏳ 等待图表生成...")
        time.sleep(5)
        
        # 停止记录器
        print("🛑 停止损失记录器...")
        stop_loss_logging()
        
        print("✅ 基本功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_methods():
    """测试接口方法"""
    print("\n🧪 测试2: 接口方法测试")
    print("=" * 40)
    
    try:
        from loss_logger_interface import (
            LossLoggerInterface, 
            log_attention_loss, 
            log_ppo_loss, 
            log_gnn_loss,
            get_loss_log_directory,
            is_loss_logger_alive
        )
        
        # 创建接口实例
        print("🎯 创建损失记录器接口...")
        interface = LossLoggerInterface(
            experiment_name="test_interface_methods",
            update_interval=3.0
        )
        
        print(f"✅ 接口已创建")
        print(f"   日志目录: {interface.get_log_dir()}")
        print(f"   记录器状态: {'运行中' if is_loss_logger_alive() else '未运行'}")
        
        # 测试便捷函数
        print("📊 测试便捷函数...")
        for step in range(50):
            log_attention_loss(step, {'attention_loss': 1.8 - step*0.01})
            log_ppo_loss(step, {
                'actor_loss': 1.3 - step*0.008, 
                'critic_loss': 1.1 - step*0.006
            })
            log_gnn_loss(step, {'gnn_loss': 2.5 - step*0.015})
            
            if step % 10 == 0:
                print(f"   步数: {step}, 日志目录: {get_loss_log_directory()}")
        
        print("✅ 便捷函数测试完成")
        
        # 停止接口
        print("🛑 停止接口...")
        interface.stop()
        
        print("✅ 接口方法测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 接口方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_manager():
    """测试上下文管理器"""
    print("\n🧪 测试3: 上下文管理器测试")
    print("=" * 40)
    
    try:
        from loss_logger_interface import LossLoggingContext, log_network_loss
        
        print("🎯 使用上下文管理器...")
        with LossLoggingContext(experiment_name="test_context_manager") as logger:
            print(f"✅ 上下文已创建")
            print(f"   日志目录: {logger.get_log_dir()}")
            
            # 生成一些测试数据
            for step in range(30):
                log_network_loss('attention', step, {'attention_loss': 1.5 - step*0.01})
                log_network_loss('ppo', step, {'actor_loss': 1.2 - step*0.008})
                
                if step % 10 == 0:
                    print(f"   步数: {step}")
                    
                time.sleep(0.05)
        
        print("✅ 上下文管理器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 上下文管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_decorator():
    """测试装饰器"""
    print("\n🧪 测试4: 装饰器测试")
    print("=" * 40)
    
    try:
        from loss_logger_interface import auto_log_loss, start_loss_logging, stop_loss_logging
        
        # 启动记录器
        print("📊 启动损失记录器...")
        start_loss_logging(experiment_name="test_decorator")
        
        # 定义带装饰器的函数
        @auto_log_loss('ppo')
        def mock_ppo_training(step):
            """模拟PPO训练"""
            return {
                'actor_loss': max(0.01, 1.5 - step*0.01),
                'critic_loss': max(0.01, 1.2 - step*0.008)
            }
        
        @auto_log_loss('attention', 'training_step')
        def mock_attention_training(training_step, other_param):
            """模拟Attention训练"""
            return {
                'attention_loss': max(0.05, 2.0 - training_step*0.015),
                'attention_accuracy': min(1.0, 0.3 + training_step*0.01)
            }
        
        print("🎯 测试装饰器...")
        for step in range(20):
            mock_ppo_training(step)
            mock_attention_training(training_step=step, other_param="test")
            
            if step % 5 == 0:
                print(f"   步数: {step}")
                
            time.sleep(0.1)
        
        print("✅ 装饰器测试完成")
        
        # 停止记录器
        stop_loss_logging()
        
        return True
        
    except Exception as e:
        print(f"❌ 装饰器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_output_files():
    """检查输出文件"""
    print("\n🧪 测试5: 输出文件检查")
    print("=" * 40)
    
    try:
        log_dir = "network_loss_logs"
        
        if not os.path.exists(log_dir):
            print(f"⚠️ 日志目录不存在: {log_dir}")
            return False
        
        print(f"📁 检查日志目录: {log_dir}")
        
        # 列出所有实验目录
        experiments = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        
        if not experiments:
            print("⚠️ 没有找到实验目录")
            return False
        
        print(f"📊 找到 {len(experiments)} 个实验:")
        for exp in experiments:
            print(f"   - {exp}")
        
        # 检查最新的实验目录
        latest_exp = max(experiments, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
        exp_dir = os.path.join(log_dir, latest_exp)
        
        print(f"🔍 检查最新实验: {latest_exp}")
        
        # 检查文件
        expected_files = [
            'config.json',
            'network_loss_curves_realtime.png'
        ]
        
        expected_csv_files = [
            'attention_losses.csv',
            'ppo_losses.csv', 
            'gnn_losses.csv'
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(exp_dir, file_name)
            if os.path.exists(file_path):
                print(f"   ✅ {file_name}")
            else:
                print(f"   ⚠️ {file_name} (未找到)")
        
        csv_found = 0
        for file_name in expected_csv_files:
            file_path = os.path.join(exp_dir, file_name)
            if os.path.exists(file_path):
                print(f"   ✅ {file_name}")
                csv_found += 1
            else:
                print(f"   ⚠️ {file_name} (未找到)")
        
        if csv_found > 0:
            print("✅ 输出文件检查完成")
            return True
        else:
            print("⚠️ 没有找到CSV文件")
            return False
        
    except Exception as e:
        print(f"❌ 输出文件检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 损失记录器系统测试")
    print("=" * 50)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有测试
    tests = [
        ("基本功能", test_basic_functionality),
        ("接口方法", test_interface_methods),
        ("上下文管理器", test_context_manager),
        ("装饰器", test_decorator),
        ("输出文件", check_output_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}测试通过")
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
    
    # 测试结果
    print(f"\n{'='*50}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！损失记录器系统工作正常")
        return 0
    else:
        print(f"⚠️ {total-passed} 个测试失败，请检查系统配置")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 测试程序出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



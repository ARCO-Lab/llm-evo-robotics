#!/usr/bin/env python3
"""
专门测试2关节机器人的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import argparse
import time
from enhanced_train_interface import MAPElitesTrainingInterface

def test_2_joint_robot():
    """测试2关节机器人"""
    print("🤖 测试2关节机器人")
    print("=" * 40)
    
    # 创建2关节机器人的参数
    training_args = argparse.Namespace()
    training_args.seed = 42
    training_args.num_joints = 2  # 🎯 2个关节
    training_args.link_lengths = [50.0, 60.0]  # 🎯 2节的长度
    training_args.lr = 1e-4
    training_args.alpha = 0.2
    training_args.tau = 0.005
    training_args.gamma = 0.99
    training_args.batch_size = 32
    training_args.buffer_capacity = 5000
    training_args.warmup_steps = 50  # 很少的热身步数
    training_args.target_entropy_factor = 0.8
    training_args.total_steps = 300  # 很短的训练用于测试
    training_args.update_frequency = 1
    training_args.save_dir = './test_2_joint_robot'
    
    print(f"🎯 机器人配置:")
    print(f"   关节数: {training_args.num_joints}")
    print(f"   链节长度: {training_args.link_lengths}")
    print(f"   训练步数: {training_args.total_steps}")
    
    # 创建训练接口 - 开启渲染以便观察
    interface = MAPElitesTrainingInterface(
        silent_mode=False,      # 显示输出
        enable_rendering=True   # 开启可视化
    )
    
    print(f"\n🚀 开始训练2关节机器人...")
    print(f"💡 请观察渲染窗口中的机器人是否为2个关节")
    
    try:
        start_time = time.time()
        result = interface.train_individual(training_args)
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成:")
        print(f"   耗时: {training_time:.1f}秒")
        print(f"   平均奖励: {result['avg_reward']:.2f}")
        print(f"   成功率: {result['success_rate']:.2f}")
        
        print(f"\n❓ 观察结果:")
        print(f"   1. 渲染窗口中的机器人是否有2个关节?")
        print(f"   2. 机器人的运动是否符合2关节的预期?")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_joint_counts():
    """测试不同关节数的机器人"""
    print("\n🧪 测试不同关节数的机器人")
    print("=" * 40)
    
    test_configs = [
        {"joints": 2, "lengths": [40.0, 50.0]},
        {"joints": 3, "lengths": [35.0, 45.0, 40.0]},
        {"joints": 4, "lengths": [30.0, 40.0, 35.0, 25.0]}
    ]
    
    interface = MAPElitesTrainingInterface(
        silent_mode=False,
        enable_rendering=True
    )
    
    for i, config in enumerate(test_configs):
        print(f"\n🔬 测试配置 {i+1}: {config['joints']}关节")
        
        # 创建参数
        training_args = argparse.Namespace()
        training_args.seed = 42
        training_args.num_joints = config['joints']
        training_args.link_lengths = config['lengths'].copy()
        training_args.lr = 1e-4
        training_args.alpha = 0.2
        training_args.tau = 0.005
        training_args.gamma = 0.99
        training_args.batch_size = 32
        training_args.buffer_capacity = 5000
        training_args.warmup_steps = 30
        training_args.target_entropy_factor = 0.8
        training_args.total_steps = 200  # 很短的训练
        training_args.update_frequency = 1
        training_args.save_dir = f'./test_{config["joints"]}_joint_robot'
        
        print(f"   配置: {config['joints']}关节, 长度={config['lengths']}")
        
        try:
            result = interface.train_individual(training_args)
            print(f"   ✅ 成功: avg_reward={result['avg_reward']:.2f}")
            
            # 等待用户确认
            user_input = input(f"   ❓ 机器人是否显示为{config['joints']}关节? (y/n/s=跳过): ")
            if user_input.lower() == 'n':
                print(f"   ❌ 配置{i+1}未正确显示")
                return False
            elif user_input.lower() == 's':
                print(f"   ⏭️  跳过配置{i+1}")
                break
            else:
                print(f"   ✅ 配置{i+1}正确")
                
        except Exception as e:
            print(f"   ❌ 配置{i+1}失败: {e}")
            return False
    
    return True

def quick_debug_parameter_passing():
    """快速调试参数传递"""
    print("\n🔍 快速调试参数传递")
    print("=" * 40)
    
    # 创建接口
    interface = MAPElitesTrainingInterface(silent_mode=False, enable_rendering=False)
    
    # 创建测试参数
    args = argparse.Namespace()
    args.num_joints = 2
    args.link_lengths = [30.0, 40.0]
    args.lr = 1e-4
    args.alpha = 0.2
    args.tau = 0.005
    args.gamma = 0.99
    args.batch_size = 32
    args.buffer_capacity = 5000
    args.warmup_steps = 100
    args.target_entropy_factor = 0.8
    args.total_steps = 100
    args.update_frequency = 1
    args.save_dir = './debug_test'
    
    print(f"🎯 输入参数:")
    print(f"   args.num_joints = {args.num_joints}")
    print(f"   args.link_lengths = {args.link_lengths}")
    
    # 调用参数转换
    enhanced_args = interface._convert_to_enhanced_args(args)
    
    print(f"\n📊 转换后参数:")
    print(f"   enhanced_args.num_joints = {enhanced_args.num_joints}")
    print(f"   enhanced_args.link_lengths = {enhanced_args.link_lengths}")
    
    # 验证
    if enhanced_args.num_joints == 2 and enhanced_args.link_lengths == [30.0, 40.0]:
        print(f"✅ 参数转换正确")
        return True
    else:
        print(f"❌ 参数转换有问题")
        return False

def main():
    """主函数"""
    print("🤖 2关节机器人测试套件")
    print("=" * 50)
    
    # 选择测试模式
    print("\n选择测试模式:")
    print("1. 快速调试参数传递 (无渲染)")
    print("2. 测试单个2关节机器人 (有渲染)")
    print("3. 测试多种关节配置 (有渲染)")
    
    try:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == '1':
            success = quick_debug_parameter_passing()
            if success:
                print("🎉 参数传递正常!")
            else:
                print("🔧 参数传递有问题，需要检查enhanced_train_interface.py")
                
        elif choice == '2':
            success = test_2_joint_robot()
            if success:
                print("🎉 2关节机器人测试完成!")
            else:
                print("🔧 需要检查机器人配置传递")
                
        elif choice == '3':
            success = test_different_joint_counts()
            if success:
                print("🎉 所有关节配置测试通过!")
            else:
                print("🔧 某些配置有问题")
                
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")

if __name__ == "__main__":
    main()
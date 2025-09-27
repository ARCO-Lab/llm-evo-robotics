#!/usr/bin/env python3
"""
Reacher 环境使用示例
展示如何在现有代码中使用新的环境工厂
"""

import sys
import os
import numpy as np

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

def example_basic_usage():
    """基本使用示例"""
    print("🎯 基本使用示例")
    print("=" * 40)
    
    # 方法1: 使用环境工厂（推荐）
    from envs.reacher_env_factory import create_reacher_env
    
    # 自动选择最佳环境
    env = create_reacher_env(version='auto', render_mode=None)
    
    # 运行几个步骤
    obs = env.reset()
    print(f"初始观察: {obs.shape}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"步骤 {step+1}: 奖励={reward:.3f}, 完成={done}, 距离={info['distance']:.1f}")
        
        if done:
            obs = env.reset()
            print("环境重置")
    
    env.close()
    print("✅ 基本使用示例完成\n")

def example_version_comparison():
    """版本对比示例"""
    print("🔍 版本对比示例")
    print("=" * 40)
    
    from envs.reacher_env_factory import create_reacher_env
    
    versions = ['original', 'mujoco']
    results = {}
    
    for version in versions:
        try:
            print(f"测试 {version} 版本...")
            env = create_reacher_env(version=version, render_mode=None)
            
            # 运行性能测试
            import time
            start_time = time.time()
            
            obs = env.reset()
            total_reward = 0
            steps = 0
            
            for _ in range(50):  # 运行50步
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    obs = env.reset()
            
            end_time = time.time()
            
            results[version] = {
                'total_reward': total_reward,
                'steps': steps,
                'time': end_time - start_time,
                'steps_per_second': steps / (end_time - start_time)
            }
            
            env.close()
            print(f"   ✅ {version} 版本测试完成")
            
        except Exception as e:
            print(f"   ❌ {version} 版本测试失败: {e}")
            results[version] = None
    
    # 显示结果
    print("\n📊 性能对比结果:")
    for version, result in results.items():
        if result:
            print(f"   {version}:")
            print(f"     总奖励: {result['total_reward']:.2f}")
            print(f"     步数: {result['steps']}")
            print(f"     耗时: {result['time']:.3f}秒")
            print(f"     性能: {result['steps_per_second']:.1f} 步/秒")
        else:
            print(f"   {version}: 不可用")
    
    print("✅ 版本对比示例完成\n")

def example_backward_compatibility():
    """向后兼容性示例"""
    print("🔄 向后兼容性示例")
    print("=" * 40)
    
    # 旧的导入方式仍然有效
    try:
        from envs.reacher_env_factory import Reacher2DEnv
        
        # 使用旧的构造方式
        env = Reacher2DEnv(num_links=2, render_mode=None)
        
        # 旧的使用方式
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"旧接口工作正常: obs={obs.shape}, reward={reward:.3f}")
        
        env.close()
        print("✅ 向后兼容性测试通过")
        
    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
    
    print("✅ 向后兼容性示例完成\n")

def example_configuration():
    """配置示例"""
    print("⚙️ 配置示例")
    print("=" * 40)
    
    from envs.reacher_env_factory import create_reacher_env
    
    # 使用自定义配置
    config_path = "examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    
    try:
        # 创建带配置的环境
        env = create_reacher_env(
            version='mujoco',  # 明确指定使用 MuJoCo 版本
            render_mode=None,
            config_path=config_path,
            curriculum_stage=1
        )
        
        print(f"环境配置:")
        print(f"   观察空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        print(f"   关节数量: {env.num_links}")
        
        # 测试运行
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"运行测试: 奖励={reward:.3f}, 目标距离={info['distance']:.1f}")
        
        env.close()
        print("✅ 配置示例完成")
        
    except Exception as e:
        print(f"❌ 配置示例失败: {e}")
    
    print("✅ 配置示例完成\n")

def main():
    """主函数"""
    print("🎯 Reacher 环境使用示例")
    print("=" * 60)
    
    # 运行所有示例
    example_basic_usage()
    example_version_comparison()
    example_backward_compatibility()
    example_configuration()
    
    print("🎉 所有示例完成!")
    print("\n💡 使用建议:")
    print("   1. 使用 create_reacher_env(version='auto') 自动选择最佳环境")
    print("   2. 如果需要特定版本，使用 version='mujoco' 或 version='original'")
    print("   3. 旧代码可以继续使用 Reacher2DEnv() 构造函数")
    print("   4. MuJoCo 版本提供更好的物理仿真，推荐在可用时使用")

if __name__ == "__main__":
    main()

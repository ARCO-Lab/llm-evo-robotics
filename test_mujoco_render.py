#!/usr/bin/env python3
"""
测试MuJoCo渲染是否正常工作
"""

import gymnasium as gym
import time
import numpy as np

def test_standard_reacher_render():
    """测试标准Reacher渲染"""
    print("🎮 测试标准 MuJoCo Reacher-v5 渲染")
    
    try:
        # 创建带渲染的环境
        env = gym.make('Reacher-v5', render_mode='human')
        print("✅ 环境创建成功")
        
        # 重置环境
        obs, info = env.reset()
        print("✅ 环境重置成功")
        
        print("🎯 开始渲染测试 (10步)...")
        print("   如果看到MuJoCo窗口，说明渲染正常工作")
        
        for step in range(10):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: 奖励={reward:.3f}")
            
            # 暂停一下让您看清楚
            time.sleep(0.5)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print("✅ 渲染测试完成")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 渲染测试失败: {e}")
        return False

def test_real_multi_joint_render():
    """测试真实多关节环境渲染"""
    print("\n🎮 测试真实多关节环境渲染")
    
    try:
        from real_multi_joint_reacher import RealMultiJointWrapper
        
        # 创建2关节环境
        env = RealMultiJointWrapper(
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode='human'  # 明确启用渲染
        )
        print("✅ 多关节环境创建成功")
        
        # 重置环境
        obs, info = env.reset()
        print("✅ 多关节环境重置成功")
        
        print("🎯 开始多关节渲染测试 (10步)...")
        
        for step in range(10):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: 奖励={reward:.3f}, 距离={info.get('distance_to_target', 'N/A'):.3f}")
            
            # 暂停一下
            time.sleep(0.5)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print("✅ 多关节渲染测试完成")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 多关节渲染测试失败: {e}")
        return False

def test_trained_model_render():
    """测试训练好的模型渲染"""
    print("\n🎮 测试训练好的模型渲染")
    
    try:
        from stable_baselines3 import SAC
        from real_multi_joint_reacher import RealMultiJointWrapper
        
        # 创建环境
        env = RealMultiJointWrapper(
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode='human'
        )
        
        # 加载模型
        model_path = "models/multi_joint_2j_sac"
        model = SAC.load(model_path, env=env)
        print("✅ 模型加载成功")
        
        # 测试模型
        obs, info = env.reset()
        print("🎯 开始训练模型渲染测试 (20步)...")
        print("   使用训练好的策略控制机器人")
        
        for step in range(20):
            # 使用训练好的模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            distance = info.get('distance_to_target', float('inf'))
            print(f"   Step {step+1}: 奖励={reward:.3f}, 距离={distance:.3f}")
            
            # 暂停让您观察
            time.sleep(0.3)
            
            if terminated or truncated:
                obs, info = env.reset()
                print("   环境重置")
        
        print("✅ 训练模型渲染测试完成")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 训练模型渲染测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🌟 MuJoCo 渲染测试套件")
    print("💡 请注意观察是否有MuJoCo窗口弹出")
    print()
    
    results = []
    
    # 测试1: 标准Reacher
    print("=" * 60)
    result1 = test_standard_reacher_render()
    results.append(("标准Reacher", result1))
    
    # 等待一下
    time.sleep(2)
    
    # 测试2: 多关节环境
    print("=" * 60)
    result2 = test_real_multi_joint_render()
    results.append(("多关节环境", result2))
    
    # 等待一下
    time.sleep(2)
    
    # 测试3: 训练模型
    print("=" * 60)
    result3 = test_trained_model_render()
    results.append(("训练模型", result3))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 渲染测试总结")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    all_success = all(result for _, result in results)
    if all_success:
        print("\n🎉 所有渲染测试都成功！")
        print("   如果您看到了MuJoCo窗口，说明渲染正常工作")
    else:
        print("\n⚠️ 部分渲染测试失败")
        print("   可能需要检查MuJoCo或图形驱动配置")

if __name__ == "__main__":
    main()



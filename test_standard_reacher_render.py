#!/usr/bin/env python3
"""
测试标准MuJoCo Reacher的渲染和训练
确保基础功能正常工作
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import time
import numpy as np

def test_standard_reacher_manual():
    """手动测试标准Reacher的渲染"""
    print("🎮 手动测试标准Reacher渲染")
    
    # 创建环境
    env = gym.make('Reacher-v5', render_mode='human')
    
    print("✅ 标准Reacher环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    obs, info = env.reset()
    print(f"\n📊 初始状态:")
    print(f"   观察维度: {obs.shape}")
    print(f"   cos角度: {obs[0:2]}")  # cos0, cos1
    print(f"   sin角度: {obs[2:4]}")  # sin0, sin1  
    print(f"   关节速度: {obs[4:6]}")  # vel0, vel1
    print(f"   末端位置: {obs[6:8]}")  # ee_x, ee_y
    print(f"   目标位置: {obs[8:10]}")  # target_x, target_y
    
    print(f"\n🎯 手动控制测试 (10步，每步1秒)")
    
    # 测试动作
    test_actions = [
        [1.0, 0.0],   # 只动第1个关节
        [0.0, 1.0],   # 只动第2个关节
        [-1.0, 0.0],  # 反向第1个关节
        [0.0, -1.0],  # 反向第2个关节
        [1.0, 1.0],   # 两关节同向
        [-1.0, -1.0], # 两关节反向
        [1.0, -1.0],  # 关节相对
        [-1.0, 1.0],  # 关节相对
        [0.5, 0.5],   # 小幅动作
        [0.0, 0.0]    # 无动作
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n🔧 Step {i+1}: 动作 {action}")
        
        # 记录执行前状态
        prev_obs = obs.copy()
        prev_angles = np.arctan2(prev_obs[2:4], prev_obs[0:2])
        
        # 执行动作并计时
        start_time = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - start_time
        
        # 计算变化
        new_angles = np.arctan2(obs[2:4], obs[0:2])
        angle_changes = new_angles - prev_angles
        
        print(f"   执行时间: {step_time*1000:.1f}ms")
        print(f"   角度变化: {np.degrees(angle_changes)}度")
        print(f"   关节速度: {obs[4:6]}")
        print(f"   末端位置: {obs[6:8]}")
        print(f"   距离目标: {np.linalg.norm(obs[6:8] - obs[8:10]):.3f}")
        print(f"   奖励: {reward:.3f}")
        
        # 暂停观察
        print("   (暂停1秒观察渲染...)")
        time.sleep(1.0)
        
        if terminated or truncated:
            obs, info = env.reset()
            print("   环境已重置")
    
    env.close()
    print("✅ 手动测试完成")

def test_standard_reacher_training():
    """测试标准Reacher的SAC训练"""
    print("\n🚀 测试标准Reacher SAC训练")
    
    # 创建环境
    env = gym.make('Reacher-v5', render_mode='human')
    env = Monitor(env)
    
    print("✅ 训练环境创建完成")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        env,
        verbose=2,
        learning_starts=100,
        log_interval=4,
        device='cpu'  # 使用CPU避免GPU问题
    )
    
    print("✅ SAC模型创建完成")
    print("🎯 开始训练 (5000步)...")
    print("💡 请观察MuJoCo窗口中的机器人训练过程")
    print("📊 注意FPS和关节运动情况")
    
    try:
        start_time = time.time()
        model.learn(total_timesteps=5000)
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        
        # 保存模型
        model.save("models/standard_reacher_sac_test")
        print("💾 模型已保存: models/standard_reacher_sac_test")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        # 保存中断的模型
        model.save("models/standard_reacher_sac_interrupted")
        print("💾 中断模型已保存: models/standard_reacher_sac_interrupted")
    
    env.close()

def test_fps_measurement():
    """精确测量FPS"""
    print("\n📊 精确FPS测量")
    
    env = gym.make('Reacher-v5', render_mode='human')
    obs, info = env.reset()
    
    print("🎯 测量100步的真实FPS...")
    
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每20步报告一次
        if step % 20 == 0 and step > 0:
            elapsed = time.time() - start_time
            current_fps = step / elapsed
            print(f"   Step {step}: FPS = {current_fps:.1f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    total_time = time.time() - start_time
    final_fps = num_steps / total_time
    
    print(f"\n📈 FPS测量结果:")
    print(f"   总步数: {num_steps}")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   平均FPS: {final_fps:.1f}")
    print(f"   每步时间: {total_time/num_steps*1000:.1f}ms")
    
    if final_fps > 200:
        print("⚠️ FPS异常高，可能渲染有问题")
    elif final_fps < 20:
        print("⚠️ FPS异常低，可能性能有问题")
    else:
        print("✅ FPS正常范围")
    
    env.close()

def main():
    """主函数"""
    print("🌟 标准MuJoCo Reacher渲染和训练测试")
    print("💡 确保基础功能正常工作")
    print()
    
    try:
        # 1. 手动测试渲染
        test_standard_reacher_manual()
        
        # 2. FPS测量
        test_fps_measurement()
        
        # 3. 训练测试
        print("\n" + "="*60)
        print("准备开始训练测试...")
        print("如果前面的手动测试和FPS正常，按Enter继续训练测试")
        print("如果有问题，按Ctrl+C退出")
        print("="*60)
        input("按Enter继续...")
        
        test_standard_reacher_training()
        
        print(f"\n🎉 所有测试完成！")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

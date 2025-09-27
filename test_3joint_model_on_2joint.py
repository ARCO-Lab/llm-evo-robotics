#!/usr/bin/env python3
"""
测试3关节训练的模型能否直接控制2关节Reacher
这是一个有趣的泛化能力测试
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

def test_cross_joint_compatibility():
    """测试3关节模型在2关节环境上的表现"""
    print("🧪 跨关节兼容性测试")
    print("🎯 用3关节训练的模型控制2关节Reacher")
    print("💡 这将测试模型的泛化能力")
    print()
    
    try:
        # 1. 加载3关节训练的模型
        print("📂 加载3关节训练的模型...")
        
        # 尝试加载不同的模型文件
        model_paths = [
            "models/perfect_3joint_reacher_sac_interrupted.zip",
            "models/perfect_3joint_reacher_sac.zip"
        ]
        
        model = None
        loaded_model_path = None
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = SAC.load(model_path)
                    loaded_model_path = model_path
                    print(f"✅ 成功加载: {model_path}")
                    break
                except Exception as e:
                    print(f"❌ 加载失败 {model_path}: {e}")
                    continue
        
        if model is None:
            print("❌ 没有找到可用的3关节模型")
            print("💡 请先训练3关节模型")
            return
        
        # 2. 创建2关节环境
        print("\n🌍 创建2关节Reacher环境...")
        env_2joint = gym.make('Reacher-v5', render_mode='human')
        env_2joint = Monitor(env_2joint)
        
        print("✅ 2关节环境创建完成")
        print(f"   观察空间: {env_2joint.observation_space}")
        print(f"   动作空间: {env_2joint.action_space}")
        
        # 3. 分析维度兼容性
        print(f"\n🔍 维度兼容性分析:")
        
        # 获取模型的观察和动作空间
        model_obs_space = model.observation_space
        model_action_space = model.action_space
        
        print(f"   3关节模型观察空间: {model_obs_space}")
        print(f"   3关节模型动作空间: {model_action_space}")
        print(f"   2关节环境观察空间: {env_2joint.observation_space}")
        print(f"   2关节环境动作空间: {env_2joint.action_space}")
        
        # 检查维度差异
        obs_dim_diff = model_obs_space.shape[0] - env_2joint.observation_space.shape[0]
        action_dim_diff = model_action_space.shape[0] - env_2joint.action_space.shape[0]
        
        print(f"\n📏 维度差异:")
        print(f"   观察维度差异: {obs_dim_diff} (3关节: {model_obs_space.shape[0]}, 2关节: {env_2joint.observation_space.shape[0]})")
        print(f"   动作维度差异: {action_dim_diff} (3关节: {model_action_space.shape[0]}, 2关节: {env_2joint.action_space.shape[0]})")
        
        if obs_dim_diff != 3 or action_dim_diff != 1:
            print("⚠️ 维度差异不符合预期，可能存在兼容性问题")
        
        # 4. 开始测试
        print(f"\n🎮 开始跨关节测试 (10个episode)...")
        print("💡 观察3关节模型如何控制2关节机械臂")
        
        all_episode_rewards = []
        all_episode_lengths = []
        all_episode_successes = []
        all_episode_final_distances = []
        
        for episode in range(10):
            print(f"\n📍 Episode {episode + 1}/10:")
            
            obs, info = env_2joint.reset()
            episode_reward = 0
            episode_length = 0
            episode_success = False
            
            for step in range(100):  # 每个episode最多100步
                # 关键：处理观察维度差异
                # 2关节观察: [cos1, cos2, sin1, sin2, vel1, vel2, ee_x, ee_y, target_x, target_y] (10维)
                # 3关节观察: [cos1, cos2, cos3, sin1, sin2, sin3, vel1, vel2, vel3, ee_x, ee_y, target_x, target_y] (13维)
                
                # 方法1: 用零填充缺失的第3关节信息
                padded_obs = np.zeros(model_obs_space.shape[0])
                
                # 复制2关节的cos, sin值
                padded_obs[0] = obs[0]  # cos1
                padded_obs[1] = obs[1]  # cos2
                padded_obs[2] = 0.0     # cos3 (假设第3关节为0)
                padded_obs[3] = obs[2]  # sin1
                padded_obs[4] = obs[3]  # sin2
                padded_obs[5] = 0.0     # sin3 (假设第3关节为0)
                padded_obs[6] = obs[4]  # vel1
                padded_obs[7] = obs[5]  # vel2
                padded_obs[8] = 0.0     # vel3 (假设第3关节速度为0)
                padded_obs[9] = obs[6]  # ee_x
                padded_obs[10] = obs[7] # ee_y
                padded_obs[11] = obs[8] # target_x
                padded_obs[12] = obs[9] # target_y
                
                # 使用3关节模型预测动作
                action_3joint, _states = model.predict(padded_obs, deterministic=True)
                
                # 关键：处理动作维度差异
                # 只使用前2个关节的动作，忽略第3关节
                action_2joint = action_3joint[:2]
                
                # 执行动作
                obs, reward, terminated, truncated, info = env_2joint.step(action_2joint)
                
                episode_reward += reward
                episode_length += 1
                distance = np.linalg.norm(obs[6:8] - obs[8:10])  # 计算距离
                
                # 每20步打印一次状态
                if step % 20 == 0:
                    print(f"   Step {step}: 距离={distance:.3f}m, 奖励={reward:.3f}")
                    print(f"     3关节动作: [{action_3joint[0]:.3f}, {action_3joint[1]:.3f}, {action_3joint[2]:.3f}]")
                    print(f"     使用动作: [{action_2joint[0]:.3f}, {action_2joint[1]:.3f}]")
                
                # 检查是否成功
                if distance < 0.02:
                    episode_success = True
                    print(f"   ✅ 成功! 在第{step+1}步到达目标，距离={distance:.3f}m")
                    break
                
                # 检查是否结束
                if terminated or truncated:
                    final_distance = distance
                    if terminated and not episode_success:
                        print(f"   ⚠️ Episode结束，最终距离={final_distance:.3f}m")
                    break
            else:
                # 如果循环正常结束（没有break），说明达到了100步
                final_distance = distance
                print(f"   ⏰ 达到最大步数(100)，最终距离={final_distance:.3f}m")
            
            # 记录episode统计
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_episode_successes.append(episode_success)
            all_episode_final_distances.append(final_distance)
            
            print(f"   📊 Episode {episode + 1} 总结: 奖励={episode_reward:.2f}, 长度={episode_length}, 成功={'是' if episode_success else '否'}")
        
        # 5. 分析结果
        avg_reward = np.mean(all_episode_rewards)
        avg_length = np.mean(all_episode_lengths)
        success_rate = np.mean(all_episode_successes) * 100
        avg_final_distance = np.mean(all_episode_final_distances)
        
        print(f"\n📊 跨关节兼容性测试结果:")
        print(f"   平均episode奖励: {avg_reward:.3f}")
        print(f"   平均episode长度: {avg_length:.1f}步")
        print(f"   平均最终距离: {avg_final_distance:.3f}m")
        print(f"   成功率: {success_rate:.1f}% ({int(success_rate/10)}/10 episodes)")
        
        # 6. 结论分析
        print(f"\n🔬 兼容性分析:")
        
        if success_rate >= 20:
            print("   ✅ 良好的跨关节兼容性!")
            print("   💡 3关节模型能够有效控制2关节机械臂")
            if success_rate >= 50:
                print("   🎉 优秀的泛化能力!")
        elif success_rate >= 10:
            print("   🔶 部分兼容性")
            print("   💡 3关节模型在2关节上有一定效果，但性能下降")
        else:
            print("   ⚠️ 兼容性较差")
            print("   💡 3关节模型难以有效控制2关节机械臂")
        
        print(f"\n🧠 可能的原因:")
        print(f"   • 观察空间差异: 3关节模型期望13维输入，2关节只有10维")
        print(f"   • 动作空间差异: 3关节输出3维动作，2关节只需2维")
        print(f"   • 动力学差异: 3关节和2关节的运动模式不同")
        print(f"   • 训练数据偏差: 3关节模型没有见过2关节的状态分布")
        
        # 详细统计
        successful_episodes = [i+1 for i, success in enumerate(all_episode_successes) if success]
        if successful_episodes:
            print(f"   🎯 成功的episode: {successful_episodes}")
        
        env_2joint.close()
        
        print(f"\n🎉 跨关节兼容性测试完成!")
        print(f"💡 这个实验展示了模型在不同关节数间的泛化能力")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🧪 3关节模型 → 2关节环境 兼容性测试")
    print("🎯 验证模型的跨关节泛化能力")
    print("="*60)
    
    test_cross_joint_compatibility()

if __name__ == "__main__":
    main()



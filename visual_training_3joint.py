#!/usr/bin/env python3
"""
可视化3关节Reacher训练过程
实时显示机械臂学习过程
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from complete_sequential_training_with_evaluation import create_env
import time

class VisualTrainingCallback(BaseCallback):
    """可视化训练回调"""
    
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # 每隔一定步数进行可视化评估
        if self.n_calls % self.eval_freq == 0:
            print(f"\n🎮 训练步数: {self.n_calls} - 开始可视化评估")
            self._visual_evaluation()
        return True
    
    def _visual_evaluation(self):
        """可视化评估当前模型"""
        obs, info = self.eval_env.reset()
        
        target_pos = info.get('target_pos', [0, 0])
        initial_distance = info.get('distance_to_target', 0)
        
        print(f"   目标位置: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")
        print(f"   初始距离: {initial_distance:.4f}")
        
        episode_reward = 0
        min_distance = initial_distance
        success_achieved = False
        
        for step in range(100):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            episode_reward += reward
            
            distance = info.get('distance_to_target', 0)
            min_distance = min(min_distance, distance)
            
            if info.get('is_success', False) and not success_achieved:
                print(f"   ✅ Step {step}: 到达目标! 距离={distance:.4f}")
                success_achieved = True
            
            # 每20步显示状态
            if step % 20 == 0:
                fingertip_pos = info.get('fingertip_pos', [0, 0])
                print(f"   Step {step:2d}: pos=({fingertip_pos[0]:.3f},{fingertip_pos[1]:.3f}), dist={distance:.4f}")
            
            # 控制速度以便观察
            time.sleep(0.02)
            
            if terminated or truncated:
                break
        
        improvement = initial_distance - min_distance
        print(f"   结果: 最小距离={min_distance:.4f}, 改善={improvement:.4f}, 奖励={episode_reward:.1f}, 成功={'✅' if success_achieved else '❌'}")

def visual_training():
    """可视化训练过程"""
    print("🎮 开始可视化3关节Reacher训练")
    print("📋 你将看到:")
    print("  - 训练过程中的实时机械臂运动")
    print("  - 每1000步的性能评估")
    print("  - 学习进度的可视化展示")
    print("  - 红色目标球和绿色末端执行器")
    
    # 创建训练环境（无渲染）
    train_env = create_env(3, render_mode=None)
    
    # 创建可视化评估环境（有渲染）
    eval_env = create_env(3, render_mode='human')
    
    print("✅ 环境创建完成")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,  # 减小buffer以加快训练
        batch_size=128,     # 减小batch size
    )
    
    print("✅ SAC模型创建完成")
    
    # 创建可视化回调
    visual_callback = VisualTrainingCallback(
        eval_env=eval_env,
        eval_freq=1000,  # 每1000步评估一次
        verbose=1
    )
    
    print("\n🎯 开始可视化训练...")
    print("   按Ctrl+C可以停止训练")
    
    try:
        # 开始训练
        model.learn(
            total_timesteps=10000,
            callback=visual_callback,
            progress_bar=True
        )
        
        print("\n✅ 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    
    # 保存模型
    model.save('models/visual_trained_3joint_sac')
    print("💾 模型已保存: models/visual_trained_3joint_sac.zip")
    
    # 最终测试
    print("\n🏁 最终性能测试...")
    final_test(model, eval_env)
    
    train_env.close()
    eval_env.close()

def final_test(model, env):
    """最终性能测试"""
    print("🧪 进行5个episodes的最终测试...")
    
    success_count = 0
    rewards = []
    
    for i in range(5):
        print(f"\n--- 最终测试 Episode {i+1} ---")
        obs, info = env.reset()
        
        target_pos = info.get('target_pos', [0, 0])
        initial_distance = info.get('distance_to_target', 0)
        
        print(f"目标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}), 初始距离: {initial_distance:.4f}")
        
        episode_reward = 0
        min_distance = initial_distance
        success_achieved = False
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            distance = info.get('distance_to_target', 0)
            min_distance = min(min_distance, distance)
            
            if info.get('is_success', False):
                success_achieved = True
            
            # 控制速度
            time.sleep(0.03)
            
            if terminated or truncated:
                break
        
        if success_achieved:
            success_count += 1
        
        rewards.append(episode_reward)
        improvement = initial_distance - min_distance
        
        print(f"结果: 最小距离={min_distance:.4f}, 改善={improvement:.4f}, 奖励={episode_reward:.1f}, 成功={'✅' if success_achieved else '❌'}")
    
    print(f"\n📊 最终测试结果:")
    print(f"成功率: {success_count/5:.1%}")
    print(f"平均奖励: {np.mean(rewards):.1f}")
    print(f"奖励范围: [{min(rewards):.1f}, {max(rewards):.1f}]")

def quick_demo():
    """快速演示现有模型"""
    print("🎮 快速演示现有3关节模型")
    
    try:
        # 尝试加载现有模型
        model = SAC.load('models/complete_sequential_3joint_reacher.zip')
        print("✅ 加载现有模型成功")
    except:
        print("❌ 无法加载现有模型，将使用随机动作演示")
        model = None
    
    # 创建可视化环境
    env = create_env(3, render_mode='human')
    
    print("\n🎯 开始演示...")
    print("   你将看到3关节机械臂的运动")
    
    try:
        for episode in range(3):
            print(f"\n--- Episode {episode+1} ---")
            obs, info = env.reset()
            
            target_pos = info.get('target_pos', [0, 0])
            initial_distance = info.get('distance_to_target', 0)
            
            print(f"目标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}), 初始距离: {initial_distance:.4f}")
            
            for step in range(100):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()  # 随机动作
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                distance = info.get('distance_to_target', 0)
                
                if step % 20 == 0:
                    fingertip_pos = info.get('fingertip_pos', [0, 0])
                    print(f"  Step {step:2d}: pos=({fingertip_pos[0]:.3f},{fingertip_pos[1]:.3f}), dist={distance:.4f}")
                
                # 控制速度
                time.sleep(0.05)
                
                if terminated or truncated:
                    break
            
            time.sleep(1.0)  # Episode间暂停
    
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    
    env.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # 快速演示模式
        quick_demo()
    else:
        # 完整可视化训练模式
        visual_training()


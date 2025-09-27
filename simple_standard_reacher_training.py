#!/usr/bin/env python3
"""
简单的标准MuJoCo Reacher SAC训练
确保基础训练功能正常
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import time

def train_standard_reacher():
    """训练标准Reacher"""
    print("🚀 标准MuJoCo Reacher SAC训练")
    
    # 创建环境
    env = gym.make('Reacher-v5', render_mode='human')
    env = Monitor(env)
    
    print("✅ 训练环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        env,
        verbose=2,
        learning_starts=100,
        device='cpu'  # 使用CPU避免GPU问题
    )
    
    print("✅ SAC模型创建完成")
    print("🎯 开始训练 (5000步)...")
    print("💡 请观察MuJoCo窗口中的机器人训练过程")
    print("📊 注意FPS和关节运动情况")
    print()
    
    try:
        start_time = time.time()
        
        # 训练模型，每1000步输出一次
        model.learn(
            total_timesteps=5000,
            log_interval=4  # 这个参数应该在learn()中
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {5000/training_time:.1f}")
        
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
    
    finally:
        env.close()

def main():
    """主函数"""
    print("🌟 标准MuJoCo Reacher训练测试")
    print("💡 确保基础训练功能正常工作")
    print()
    
    try:
        train_standard_reacher()
        print(f"\n🎉 训练测试完成！")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



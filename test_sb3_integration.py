#!/usr/bin/env python3
"""
测试 SB3 SAC 与 MuJoCo Reacher 环境的集成
"""

import sys
import os
import numpy as np

# 添加路径
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/sac'))

def test_sb3_with_mujoco_reacher():
    """测试 SB3 SAC 与 MuJoCo Reacher 的集成"""
    print("🧪 测试 SB3 SAC 与 MuJoCo Reacher 集成")
    print("=" * 60)
    
    try:
        # 导入环境工厂
        os.chdir(os.path.join(base_dir, 'examples/2d_reacher'))
        from envs.reacher_env_factory import create_reacher_env
        
        # 导入 SB3 适配器
        sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/sac'))
        from sb3_sac_adapter import SB3SACFactory
        
        # 创建 MuJoCo 环境
        print("🎯 创建 MuJoCo Reacher 环境")
        env = create_reacher_env(version='mujoco', render_mode=None)
        
        print(f"   观察空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        print(f"   动作维度: {env.action_space.shape[0]}")
        
        # 创建 SB3 SAC 适配器
        print(f"\n🤖 创建 SB3 SAC 适配器")
        sac = SB3SACFactory.create_reacher_sac(
            action_dim=env.action_space.shape[0],
            buffer_capacity=10000,
            batch_size=64,
            lr=3e-4,
            device='cpu'
        )
        
        # 设置环境
        sac.set_env(env)
        
        # 测试基本交互
        print(f"\n🎮 测试环境交互")
        obs, info = env.reset()
        print(f"   初始观察: {obs.shape}")
        
        # 测试动作生成
        for i in range(5):
            action = sac.get_action(obs, deterministic=False)
            print(f"   步骤 {i+1}: 动作 {action} (范围: [{action.min():.3f}, {action.max():.3f}])")
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"            奖励: {reward:.6f}, 完成: {done}")
            
            if done:
                obs, info = env.reset()
        
        # 测试兼容性接口
        print(f"\n�� 测试兼容性接口")
        print(f"   can_sample(64): {sac.can_sample(64)}")
        print(f"   buffer大小: {len(sac)}")
        print(f"   熵系数: {sac.alpha}")
        
        # 测试更新接口
        update_result = sac.update()
        print(f"   更新结果: {update_result}")
        
        env.close()
        
        print(f"\n✅ SB3 SAC 与 MuJoCo Reacher 集成测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sb3_training_interface():
    """测试 SB3 SAC 的训练接口"""
    print(f"\n🏋️ 测试 SB3 SAC 训练接口")
    print("-" * 60)
    
    try:
        # 导入环境工厂
        os.chdir(os.path.join(base_dir, 'examples/2d_reacher'))
        from envs.reacher_env_factory import create_reacher_env
        from sb3_sac_adapter import SB3SACFactory
        
        # 创建环境
        env = create_reacher_env(version='mujoco', render_mode=None)
        
        # 创建SAC
        sac = SB3SACFactory.create_reacher_sac(
            action_dim=env.action_space.shape[0],
            buffer_capacity=5000,
            batch_size=32,
            lr=3e-4,
            device='cpu'
        )
        sac.set_env(env)
        
        # 短期训练测试
        print("🚀 开始短期训练测试 (1000 steps)")
        sac.learn(total_timesteps=1000)
        
        # 测试训练后的性能
        print("🎯 测试训练后的动作生成")
        obs, info = env.reset()
        for i in range(3):
            action = sac.get_action(obs, deterministic=True)  # 确定性动作
            print(f"   确定性动作 {i+1}: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                obs, info = env.reset()
        
        # 测试保存和加载
        print("💾 测试模型保存和加载")
        save_path = "test_sb3_sac_model"
        sac.save(save_path)
        
        # 创建新的SAC并加载
        sac_loaded = SB3SACFactory.create_reacher_sac(
            action_dim=env.action_space.shape[0],
            device='cpu'
        )
        sac_loaded.load(save_path, env=env)
        
        # 测试加载后的动作
        obs, info = env.reset()
        action_original = sac.get_action(obs, deterministic=True)
        action_loaded = sac_loaded.get_action(obs, deterministic=True)
        
        print(f"   原始模型动作: {action_original}")
        print(f"   加载模型动作: {action_loaded}")
        print(f"   动作差异: {np.abs(action_original - action_loaded).max():.6f}")
        
        # 清理
        if os.path.exists(save_path + ".zip"):
            os.remove(save_path + ".zip")
        
        env.close()
        
        print(f"✅ SB3 SAC 训练接口测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 训练接口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎯 SB3 SAC 集成测试套件")
    
    success1 = test_sb3_with_mujoco_reacher()
    success2 = test_sb3_training_interface()
    
    print(f"\n📋 测试结果总结:")
    print(f"   基本集成测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   训练接口测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print(f"\n🎉 所有测试通过！SB3 SAC 可以替换现有实现！")
        print(f"\n💡 下一步:")
        print(f"   1. 修改 enhanced_train_backup.py 使用 SB3SACAdapter")
        print(f"   2. 运行完整训练测试")
        print(f"   3. 对比性能差异")
    else:
        print(f"\n⚠️ 部分测试失败，需要进一步调试")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

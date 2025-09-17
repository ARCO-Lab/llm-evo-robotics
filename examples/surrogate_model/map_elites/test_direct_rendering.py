#!/usr/bin/env python3
"""
直接测试MAP-Elites中的reacher2d环境渲染
"""

import sys
import os
import time
import argparse

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '2d_reacher', 'envs'))
sys.path.append(os.path.dirname(__file__))

def test_direct_reacher_rendering():
    """直接测试reacher2d环境渲染"""
    print("🎨 测试直接reacher2d环境渲染")
    print("=" * 50)
    
    try:
        from reacher2d_env import Reacher2DEnv
        
        # 创建环境 - 强制启用渲染
        print("🤖 创建reacher2d环境 (启用渲染)...")
        env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 40, 30], 
            render_mode='human'  # 强制人类可视化模式
        )
        
        print("✅ 环境创建成功")
        print("🎨 应该看到一个渲染窗口...")
        
        # 重置环境
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        print("✅ 环境重置成功")
        
        # 运行一些步骤并渲染
        print("🏃 运行训练步骤并渲染...")
        for step in range(50):
            # 随机动作
            action = env.action_space.sample()
            
            # 执行动作
            result = env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result
                truncated = False
            
            # 渲染
            env.render()
            
            # 打印一些信息
            if step % 10 == 0:
                print(f"步骤 {step}: reward={reward:.3f}")
            
            # 短暂暂停
            time.sleep(0.1)
            
            # 如果episode结束，重置
            if done or truncated:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                print(f"Episode结束，重置环境")
        
        print("✅ 渲染测试完成")
        env.close()
        
    except Exception as e:
        print(f"❌ 直接渲染测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_map_elites_with_rendering():
    """测试MAP-Elites训练器中的渲染"""
    print("\n🧬 测试MAP-Elites训练器中的渲染")
    print("=" * 50)
    
    try:
        from training_adapter import MAPElitesTrainingAdapter
        from map_elites_core import Individual, RobotGenotype, RobotPhenotype
        
        # 创建基础参数
        base_args = argparse.Namespace()
        base_args.env_type = 'reacher2d'
        base_args.num_processes = 1
        base_args.seed = 42
        base_args.save_dir = './test_direct_rendering'
        base_args.lr = 3e-4
        base_args.alpha = 0.2
        base_args.tau = 0.005
        base_args.gamma = 0.99
        base_args.use_real_training = True
        
        print("🔧 创建训练适配器 (启用渲染)...")
        adapter = MAPElitesTrainingAdapter(
            base_args=base_args,
            enable_rendering=True,  # 🎨 启用渲染
            silent_mode=False,      # 🔊 显示详细输出
            use_genetic_fitness=False  # 简化测试
        )
        
        print("✅ 训练适配器创建成功")
        
        # 创建一个测试个体
        genotype = RobotGenotype(
            num_links=3,
            link_lengths=[60.0, 40.0, 30.0],
            lr=3e-4,
            alpha=0.2
        )
        
        phenotype = RobotPhenotype(
            avg_reward=0.0,
            success_rate=0.0,
            min_distance=200.0
        )
        
        individual = Individual(
            individual_id="test_render_robot",
            genotype=genotype,
            phenotype=phenotype,
            generation=0
        )
        
        print("🤖 开始评估个体 (应该显示渲染窗口)...")
        print("⚠️ 如果enhanced_train.py有语法错误，这可能会失败")
        
        # 评估个体 - 这应该显示渲染
        evaluated_individual = adapter.evaluate_individual(
            individual, 
            training_steps=100  # 短训练
        )
        
        print(f"✅ 个体评估完成: fitness={evaluated_individual.fitness:.3f}")
        
    except Exception as e:
        print(f"❌ MAP-Elites渲染测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 开始渲染测试")
    
    # 1. 直接环境渲染测试
    test_direct_reacher_rendering()
    
    # 2. MAP-Elites集成渲染测试
    test_map_elites_with_rendering()
    
    print("\n🎉 渲染测试完成")



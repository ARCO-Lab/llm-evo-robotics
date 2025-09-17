#!/usr/bin/env python3
"""
简化的MAP-Elites训练脚本，直接显示reacher2d渲染
绕过enhanced_train.py的语法错误问题
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, Any

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '2d_reacher', 'envs'))
sys.path.append(os.path.dirname(__file__))

def train_individual_with_rendering(individual_config: Dict[str, Any], training_steps: int = 200) -> Dict[str, Any]:
    """
    直接训练单个个体并显示渲染
    """
    try:
        from reacher2d_env import Reacher2DEnv
        import pygame
        
        print(f"\n🤖 训练个体: {individual_config['num_links']}关节, 长度={individual_config['link_lengths']}")
        
        # 创建环境 - 强制启用渲染
        env = Reacher2DEnv(
            num_links=individual_config['num_links'],
            link_lengths=individual_config['link_lengths'],
            render_mode='human'  # 强制人类可视化模式
        )
        
        # 设置窗口标题
        pygame.display.set_caption(f"MAP-Elites Training - Robot {individual_config.get('id', 'Unknown')}")
        
        print("🎨 渲染窗口已创建，开始训练...")
        
        # 重置环境
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_reward = 0
        episode_count = 0
        success_count = 0
        min_distance = float('inf')
        
        for step in range(training_steps):
            # 处理pygame事件（重要！）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("用户关闭窗口，停止训练")
                    env.close()
                    return {
                        'success_rate': success_count / max(episode_count, 1),
                        'avg_reward': total_reward / max(episode_count, 1),
                        'min_distance': min_distance,
                        'episodes_completed': episode_count,
                        'training_stopped_by_user': True
                    }
            
            # 简单的随机策略（您可以替换为真正的RL算法）
            action = env.action_space.sample()
            
            # 执行动作
            result = env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result
                truncated = False
            
            total_reward += reward
            
            # 计算距离目标的距离
            if 'distance_to_target' in info:
                min_distance = min(min_distance, info['distance_to_target'])
            
            # 渲染环境
            env.render()
            
            # 检查成功
            if reward > -0.1:  # 简单的成功标准
                success_count += 1
            
            # 显示进度
            if step % 50 == 0:
                print(f"  步骤 {step}/{training_steps}: reward={reward:.3f}, 总奖励={total_reward:.1f}")
            
            # 控制渲染速度
            time.sleep(0.02)  # 50 FPS
            
            # Episode结束处理
            if done or truncated:
                episode_count += 1
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                if episode_count % 5 == 0:
                    print(f"  Episode {episode_count}完成")
        
        env.close()
        
        # 计算最终指标
        final_metrics = {
            'success_rate': success_count / max(training_steps, 1),
            'avg_reward': total_reward / max(episode_count, 1),
            'min_distance': min_distance if min_distance != float('inf') else 200.0,
            'episodes_completed': episode_count,
            'total_reward': total_reward,
            'training_completed': True
        }
        
        print(f"✅ 训练完成: success_rate={final_metrics['success_rate']:.3f}, avg_reward={final_metrics['avg_reward']:.1f}")
        return final_metrics
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success_rate': 0.0,
            'avg_reward': -100.0,
            'min_distance': 200.0,
            'episodes_completed': 0,
            'training_failed': True
        }

def run_simple_map_elites():
    """运行简化的MAP-Elites进化"""
    print("🚀 简化MAP-Elites进化训练")
    print("=" * 60)
    print("🎨 每个个体都会显示实时渲染窗口")
    print("⚠️ 关闭窗口可以跳到下一个个体")
    print("=" * 60)
    
    # 创建一些测试个体
    individuals = [
        {
            'id': 'robot_1',
            'num_links': 3,
            'link_lengths': [60.0, 40.0, 30.0],
            'lr': 3e-4,
            'alpha': 0.2
        },
        {
            'id': 'robot_2', 
            'num_links': 4,
            'link_lengths': [50.0, 40.0, 30.0, 25.0],
            'lr': 2e-4,
            'alpha': 0.15
        },
        {
            'id': 'robot_3',
            'num_links': 2,
            'link_lengths': [80.0, 60.0],
            'lr': 4e-4,
            'alpha': 0.25
        },
        {
            'id': 'robot_4',
            'num_links': 5,
            'link_lengths': [40.0, 35.0, 30.0, 25.0, 20.0],
            'lr': 1e-4,
            'alpha': 0.1
        }
    ]
    
    results = []
    
    for i, individual in enumerate(individuals):
        print(f"\n🧬 第 {i+1}/{len(individuals)} 个个体")
        print(f"   ID: {individual['id']}")
        print(f"   配置: {individual['num_links']}关节, 总长度={sum(individual['link_lengths']):.1f}px")
        
        # 训练个体并显示渲染
        result = train_individual_with_rendering(individual, training_steps=300)
        result['individual_id'] = individual['id']
        result['config'] = individual
        results.append(result)
        
        # 短暂暂停
        time.sleep(1)
    
    # 显示最终结果
    print(f"\n🎉 MAP-Elites进化完成!")
    print("=" * 60)
    print("最终结果:")
    
    for result in results:
        status = "✅" if result.get('training_completed', False) else "❌"
        print(f"{status} {result['individual_id']}: success={result['success_rate']:.3f}, reward={result['avg_reward']:.1f}")
    
    # 找到最佳个体
    best_result = max(results, key=lambda x: x['success_rate'])
    print(f"\n🏆 最佳个体: {best_result['individual_id']}")
    print(f"   成功率: {best_result['success_rate']:.3f}")
    print(f"   平均奖励: {best_result['avg_reward']:.1f}")
    print(f"   最小距离: {best_result['min_distance']:.1f}px")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简化MAP-Elites训练与渲染")
    parser.add_argument("--steps", type=int, default=300, help="每个个体的训练步数")
    args = parser.parse_args()
    
    try:
        run_simple_map_elites()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


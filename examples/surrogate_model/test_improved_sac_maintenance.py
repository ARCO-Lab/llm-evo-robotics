#!/usr/bin/env python3
"""
测试改进版SAC维持任务性能
验证熵权重调度、课程学习和改进奖励的效果
"""

import sys
import os
import numpy as np
import torch
import time
from datetime import datetime

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')
sys.path.extend([
    base_dir,
    os.path.join(base_dir, 'sac'),
    os.path.join(base_dir, 'attn_model'),
    os.path.join(base_dir, '../2d_reacher/envs')
])

# 导入模块
from sac.sac_model import AttentionSACWithBuffer
from attn_model import AttnModel
from reacher2d_env_improved import ImprovedReacher2DEnv

class ImprovedSACTester:
    """改进版SAC测试器"""
    
    def __init__(self, num_joints=5, curriculum_stage=0):
        self.num_joints = num_joints
        self.curriculum_stage = curriculum_stage
        
        # 创建改进的环境
        self.env = ImprovedReacher2DEnv(
            num_links=num_joints,
            link_lengths=[90.0] * num_joints,
            curriculum_stage=curriculum_stage,
            debug_level='INFO'
        )
        
        # 创建SAC模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {self.device}")
        
        # 创建attention模型
        attn_model = AttnModel(128, 130, 130, 4)
        
        # 创建SAC模型 - 启用熵权重调度
        self.sac = AttentionSACWithBuffer(
            attn_model=attn_model,
            action_dim=num_joints,
            joint_embed_dim=128,
            buffer_capacity=5000,
            batch_size=64,  # 较小的batch size用于快速测试
            lr=1e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,  # 初始alpha值，会自动调度
            device=self.device,
            env_type='reacher2d'
        )
        
        # 确保熵权重调度启用
        self.sac.entropy_schedule_enabled = True
        print(f"🔄 熵权重调度: 启用 ({self.sac.alpha_start:.3f} → {self.sac.alpha_end:.3f})")
        
        self.stats = {
            'episodes': 0,
            'total_steps': 0,
            'successful_episodes': 0,
            'maintain_times': [],
            'alpha_history': [],
            'reward_history': [],
            'curriculum_upgrades': 0
        }
    
    def run_test(self, total_steps=2000, max_episodes=50):
        """运行改进版SAC测试"""
        print(f"\n🚀 开始改进版SAC维持任务测试")
        print(f"📊 参数: 总步数={total_steps}, 最大episodes={max_episodes}")
        print(f"🎓 初始课程阶段: {self.curriculum_stage}")
        print("="*60)
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        step = 0
        
        start_time = time.time()
        
        while step < total_steps and self.stats['episodes'] < max_episodes:
            # 获取动作
            if step < 200:  # 初始随机探索
                action = np.random.uniform(-0.5, 0.5, self.num_joints)
            else:
                # SAC策略，根据阶段调整确定性
                is_stable_phase = step > total_steps * self.sac.exploration_phase_ratio
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                gnn_embeds_tensor = torch.zeros(1, 128).to(self.device)
                action = self.sac.get_action(
                    obs_tensor, gnn_embeds_tensor, self.num_joints, 
                    deterministic=is_stable_phase
                )[0].cpu().numpy()
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 存储经验
            if step >= 200:
                self.sac.store_experience(
                    state=obs,
                    gnn_embeds=np.zeros(128),
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    next_gnn_embeds=np.zeros(128),
                    done=terminated or truncated,
                    num_joints=self.num_joints
                )
            
            # 更新网络
            if step >= 500 and step % 4 == 0 and self.sac.memory.can_sample(self.sac.batch_size):
                loss_info = self.sac.update()
                
                # 更新熵权重调度
                if step % 50 == 0:  # 更频繁的调度更新
                    self.sac.update_alpha_schedule(step, total_steps)
                    self.stats['alpha_history'].append(self.sac.alpha)
            
            episode_reward += reward
            episode_steps += 1
            step += 1
            
            # Episode结束处理
            if terminated or truncated or episode_steps >= 500:
                self.stats['episodes'] += 1
                self.stats['total_steps'] += episode_steps
                self.stats['reward_history'].append(episode_reward)
                
                # 记录维持时间
                if 'maintain_progress' in info:
                    maintain_time = int(info['maintain_progress'] * self.env.maintain_target_steps)
                    self.stats['maintain_times'].append(maintain_time)
                
                # 检查是否成功
                if info.get('maintain_completed', False):
                    self.stats['successful_episodes'] += 1
                    print(f"🎉 Episode {self.stats['episodes']}: 维持任务成功! "
                          f"奖励: {episode_reward:.1f}, 步数: {episode_steps}")
                else:
                    max_maintain = max(self.stats['maintain_times'][-10:]) if self.stats['maintain_times'] else 0
                    print(f"📊 Episode {self.stats['episodes']}: 未完成维持 "
                          f"(最大: {max_maintain}步/{self.env.maintain_target_steps}步) "
                          f"奖励: {episode_reward:.1f}")
                
                # 🆕 课程学习升级检查
                if self.stats['episodes'] % 10 == 0:
                    self.check_curriculum_upgrade()
                
                # 重置episode
                obs, info = self.env.reset()
                episode_reward = 0
                episode_steps = 0
            else:
                obs = next_obs
        
        # 测试结束，输出统计
        self.print_final_stats(time.time() - start_time)
    
    def check_curriculum_upgrade(self):
        """检查是否可以升级课程"""
        if self.stats['episodes'] < 10:
            return
            
        # 检查最近10个episode的成功率
        recent_episodes = min(10, len(self.stats['maintain_times']))
        if recent_episodes < 5:
            return
        
        recent_maintains = self.stats['maintain_times'][-recent_episodes:]
        recent_success_rate = sum(1 for m in recent_maintains 
                                 if m >= self.env.maintain_target_steps) / recent_episodes
        
        # 如果成功率 >= 60% 且当前不是最高阶段，则升级
        if recent_success_rate >= 0.6 and self.curriculum_stage < 2:
            old_stage = self.curriculum_stage
            self.curriculum_stage += 1
            self.env.set_curriculum_stage(self.curriculum_stage)
            self.stats['curriculum_upgrades'] += 1
            
            print(f"\n🎓 课程升级! {old_stage} → {self.curriculum_stage}")
            print(f"   最近成功率: {recent_success_rate:.1%}")
            print(f"   新要求: {self.env.maintain_threshold}px, {self.env.maintain_target_steps}步\n")
    
    def print_final_stats(self, elapsed_time):
        """输出最终统计结果"""
        print("\n" + "="*60)
        print("🏆 改进版SAC维持任务测试完成!")
        print("="*60)
        
        print(f"⏱️  总用时: {elapsed_time:.1f}秒")
        print(f"📊 Episodes: {self.stats['episodes']}")
        print(f"🎯 成功Episodes: {self.stats['successful_episodes']}")
        print(f"📈 成功率: {self.stats['successful_episodes']/max(self.stats['episodes'],1):.1%}")
        
        if self.stats['maintain_times']:
            print(f"⏳ 平均维持时间: {np.mean(self.stats['maintain_times']):.1f}步")
            print(f"🏆 最长维持时间: {max(self.stats['maintain_times'])}步")
            print(f"🎯 目标维持时间: {self.env.maintain_target_steps}步")
        
        if self.stats['alpha_history']:
            print(f"🔄 Alpha变化: {self.stats['alpha_history'][0]:.3f} → {self.stats['alpha_history'][-1]:.3f}")
        
        if self.stats['reward_history']:
            recent_rewards = self.stats['reward_history'][-10:]
            print(f"💰 最近10次平均奖励: {np.mean(recent_rewards):.1f}")
        
        print(f"🎓 课程升级次数: {self.stats['curriculum_upgrades']}")
        print(f"📚 最终课程阶段: {self.curriculum_stage}")
        
        # 分析改进效果
        print("\n📋 改进效果分析:")
        if self.stats['successful_episodes'] > 0:
            print("✅ 成功学会维持任务!")
            print("✅ 熵权重调度有效 - 从探索转向稳定")
            if self.stats['curriculum_upgrades'] > 0:
                print("✅ 课程学习有效 - 逐步提高难度")
        else:
            print("⚠️ 未能完全掌握维持任务")
            if self.stats['maintain_times']:
                best_maintain = max(self.stats['maintain_times'])
                target = self.env.maintain_target_steps
                progress = best_maintain / target * 100
                print(f"📊 最佳进度: {progress:.1f}% ({best_maintain}/{target}步)")
        
        print("="*60)

def main():
    """主函数"""
    print("🎯 改进版SAC维持任务测试")
    print("实现功能:")
    print("  🔄 熵权重自适应调度 (0.2 → 0.05)")
    print("  🎓 课程学习 (3个难度阶段)")
    print("  🏆 改进奖励设计 (里程碑+温和惩罚)")
    print()
    
    # 创建测试器
    tester = ImprovedSACTester(
        num_joints=5,
        curriculum_stage=0  # 从最简单的阶段开始
    )
    
    # 运行测试
    tester.run_test(
        total_steps=2000,  # 总训练步数
        max_episodes=30    # 最大episode数
    )

if __name__ == "__main__":
    main()

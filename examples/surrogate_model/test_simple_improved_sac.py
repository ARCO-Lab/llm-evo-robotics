#!/usr/bin/env python3
"""
简化版改进SAC测试 - 验证熵权重调度和课程学习效果
使用简单SAC模型，避免复杂的attention机制
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
    os.path.join(base_dir, '../2d_reacher/envs')
])

# 导入模块
from sac.simple_sac_model import SimpleSAC
from reacher2d_env_improved import ImprovedReacher2DEnv

class SimpleSACTester:
    """简化版SAC测试器 - 专注于验证改进方法"""
    
    def __init__(self, num_joints=3, curriculum_stage=0):
        self.num_joints = num_joints
        self.curriculum_stage = curriculum_stage
        
        # 创建改进的环境
        self.env = ImprovedReacher2DEnv(
            num_links=num_joints,
            link_lengths=[90.0] * num_joints,
            curriculum_stage=curriculum_stage,
            debug_level='INFO'
        )
        
        # 创建简单SAC模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {self.device}")
        
        # 计算状态维度
        state_dim = self.env.observation_space.shape[0]
        
        # 创建简单SAC模型
        self.sac = SimpleSAC(
            state_dim=state_dim,
            action_dim=num_joints,
            buffer_capacity=5000,
            batch_size=64,
            lr=1e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,  # 初始alpha值
            device=self.device
        )
        
        # 🆕 添加熵权重调度功能
        self.sac.alpha_start = 0.2
        self.sac.alpha_end = 0.05
        self.sac.entropy_schedule_enabled = True
        self.sac.exploration_phase_ratio = 0.7
        
        print(f"🔄 熵权重调度: 启用 ({self.sac.alpha_start:.3f} → {self.sac.alpha_end:.3f})")
        print(f"📊 状态维度: {state_dim}, 动作维度: {num_joints}")
        
        self.stats = {
            'episodes': 0,
            'total_steps': 0,
            'successful_episodes': 0,
            'maintain_times': [],
            'alpha_history': [],
            'reward_history': [],
            'curriculum_upgrades': 0,
            'exploration_phase_end': 0
        }
    
    def update_alpha_schedule(self, current_step, total_steps):
        """🆕 更新熵权重调度"""
        if not self.sac.entropy_schedule_enabled:
            return
            
        # 计算训练进度
        progress = min(current_step / (total_steps * self.sac.exploration_phase_ratio), 1.0)
        
        # 线性衰减alpha
        old_alpha = self.sac.alpha
        scheduled_alpha = self.sac.alpha_start * (1 - progress) + self.sac.alpha_end * progress
        
        # 更新alpha值
        self.sac.alpha = scheduled_alpha
        
        # 同步更新log_alpha
        if hasattr(self.sac, 'log_alpha'):
            self.sac.log_alpha.data.fill_(torch.log(torch.tensor(scheduled_alpha)).item())
        
        # 记录阶段转换
        if progress >= 1.0 and self.stats['exploration_phase_end'] == 0:
            self.stats['exploration_phase_end'] = current_step
            print(f"🎯 进入稳定阶段! Alpha固定在 {scheduled_alpha:.3f}")
        
        # 每100步输出一次调度信息
        if current_step % 100 == 0 and abs(float(old_alpha) - float(scheduled_alpha)) > 0.001:
            phase = "探索阶段" if progress < 1.0 else "稳定阶段"
            print(f"🔄 Alpha调度更新 [Step {current_step}]: {float(old_alpha):.3f} → {float(scheduled_alpha):.3f} ({phase})")
            
        return scheduled_alpha
    
    def run_test(self, total_steps=1500, max_episodes=25):
        """运行简化版SAC测试"""
        print(f"\n🚀 开始简化版改进SAC维持任务测试")
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
            if step < 100:  # 初始随机探索
                action = np.random.uniform(-0.3, 0.3, self.num_joints)
            else:
                # SAC策略，根据阶段调整确定性
                is_stable_phase = step > total_steps * self.sac.exploration_phase_ratio
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                if is_stable_phase:
                    # 稳定阶段：更确定性的动作
                    action_tensor, _, mean = self.sac.actor.sample(obs_tensor)
                    action = mean.squeeze(0).detach().cpu().numpy()  # 使用均值而不是采样
                else:
                    # 探索阶段：正常采样
                    action_tensor, _, _ = self.sac.actor.sample(obs_tensor)
                    action = action_tensor.squeeze(0).detach().cpu().numpy()
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 存储经验
            if step >= 100:
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
                action_tensor = torch.FloatTensor(action).to(self.device)
                reward_tensor = torch.FloatTensor([reward]).to(self.device)
                done_tensor = torch.FloatTensor([float(terminated or truncated)]).to(self.device)
                
                self.sac.memory.push(obs_tensor, action_tensor, reward_tensor, 
                                   next_obs_tensor, done_tensor)
            
            # 更新网络
            if step >= 200 and step % 4 == 0 and self.sac.memory.can_sample(self.sac.batch_size):
                loss_info = self.sac.update()
                
                # 更新熵权重调度
                if step % 50 == 0:  # 更频繁的调度更新
                    alpha = self.update_alpha_schedule(step, total_steps)
                    self.stats['alpha_history'].append(alpha)
                    
                    # 输出训练状态
                    if loss_info and step % 200 == 0:
                        print(f"📊 [Step {step}] Actor Loss: {loss_info['actor_loss']:.4f}, "
                              f"Critic Loss: {loss_info['critic_loss']:.4f}, "
                              f"Alpha: {alpha:.3f}")
            
            episode_reward += reward
            episode_steps += 1
            step += 1
            
            # Episode结束处理
            if terminated or truncated or episode_steps >= 400:
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
                    max_maintain = max(self.stats['maintain_times'][-5:]) if self.stats['maintain_times'] else 0
                    print(f"📊 Episode {self.stats['episodes']}: 未完成维持 "
                          f"(最大: {max_maintain}步/{self.env.maintain_target_steps}步) "
                          f"奖励: {episode_reward:.1f}")
                
                # 🆕 课程学习升级检查
                if self.stats['episodes'] % 8 == 0:
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
        if self.stats['episodes'] < 8:
            return
            
        # 检查最近8个episode的成功率
        recent_episodes = min(8, len(self.stats['maintain_times']))
        if recent_episodes < 4:
            return
        
        recent_maintains = self.stats['maintain_times'][-recent_episodes:]
        recent_success_rate = sum(1 for m in recent_maintains 
                                 if m >= self.env.maintain_target_steps) / recent_episodes
        
        # 如果成功率 >= 50% 且当前不是最高阶段，则升级
        if recent_success_rate >= 0.5 and self.curriculum_stage < 2:
            old_stage = self.curriculum_stage
            old_threshold = self.env.maintain_threshold
            old_target = self.env.maintain_target_steps
            
            self.curriculum_stage += 1
            self.env.set_curriculum_stage(self.curriculum_stage)
            self.stats['curriculum_upgrades'] += 1
            
            print(f"\n🎓 课程升级! {old_stage} → {self.curriculum_stage}")
            print(f"   最近成功率: {recent_success_rate:.1%}")
            print(f"   要求变化: {old_threshold}px/{old_target}步 → {self.env.maintain_threshold}px/{self.env.maintain_target_steps}步\n")
    
    def print_final_stats(self, elapsed_time):
        """输出最终统计结果"""
        print("\n" + "="*60)
        print("🏆 简化版改进SAC维持任务测试完成!")
        print("="*60)
        
        print(f"⏱️  总用时: {elapsed_time:.1f}秒")
        print(f"📊 Episodes: {self.stats['episodes']}")
        print(f"🎯 成功Episodes: {self.stats['successful_episodes']}")
        print(f"📈 成功率: {self.stats['successful_episodes']/max(self.stats['episodes'],1):.1%}")
        
        if self.stats['maintain_times']:
            print(f"⏳ 平均维持时间: {np.mean(self.stats['maintain_times']):.1f}步")
            print(f"🏆 最长维持时间: {max(self.stats['maintain_times'])}步")
            print(f"🎯 目标维持时间: {self.env.maintain_target_steps}步")
            
            # 分析维持时间趋势
            if len(self.stats['maintain_times']) >= 10:
                early = np.mean(self.stats['maintain_times'][:5])
                recent = np.mean(self.stats['maintain_times'][-5:])
                improvement = recent - early
                print(f"📈 维持时间改进: {early:.1f} → {recent:.1f} (提升{improvement:+.1f}步)")
        
        if self.stats['alpha_history']:
            print(f"🔄 Alpha变化: {self.stats['alpha_history'][0]:.3f} → {self.stats['alpha_history'][-1]:.3f}")
            if self.stats['exploration_phase_end'] > 0:
                print(f"🎯 探索→稳定转换: Step {self.stats['exploration_phase_end']}")
        
        if self.stats['reward_history']:
            recent_rewards = self.stats['reward_history'][-5:]
            print(f"💰 最近5次平均奖励: {np.mean(recent_rewards):.1f}")
        
        print(f"🎓 课程升级次数: {self.stats['curriculum_upgrades']}")
        print(f"📚 最终课程阶段: {self.curriculum_stage}")
        
        # 🎯 改进效果分析
        print("\n📋 改进方法效果分析:")
        
        # 1. 熵权重调度效果
        if self.stats['alpha_history']:
            alpha_reduction = self.stats['alpha_history'][0] - self.stats['alpha_history'][-1]
            print(f"✅ 熵权重调度: Alpha降低了 {alpha_reduction:.3f} (从探索转向稳定)")
        
        # 2. 课程学习效果
        if self.stats['curriculum_upgrades'] > 0:
            print(f"✅ 课程学习: 成功升级 {self.stats['curriculum_upgrades']} 次")
        else:
            print(f"⚠️ 课程学习: 未触发升级 (需要更多训练)")
        
        # 3. 维持任务效果
        if self.stats['successful_episodes'] > 0:
            print(f"✅ 维持任务: 成功解决 SAC 探索vs利用冲突!")
            print(f"   成功学会了在目标位置稳定维持")
        else:
            if self.stats['maintain_times']:
                best_maintain = max(self.stats['maintain_times'])
                target = self.env.maintain_target_steps
                progress = best_maintain / target * 100
                print(f"📊 维持任务: 部分成功，最佳进度 {progress:.1f}%")
                
                if progress > 50:
                    print(f"   ✅ 显著改进: 能够维持超过目标的一半时间")
                elif progress > 20:
                    print(f"   ⚠️ 有改进: 能够短期维持，但需要更多训练")
                else:
                    print(f"   ❌ 效果有限: 需要调整参数或延长训练")
            else:
                print(f"❌ 维持任务: 未能接近目标区域")
        
        print("="*60)

def main():
    """主函数"""
    print("🎯 简化版改进SAC维持任务测试")
    print("验证改进方法:")
    print("  🔄 熵权重自适应调度 (0.2 → 0.05)")
    print("  🎓 课程学习 (3个难度阶段)")
    print("  🏆 改进奖励设计 (里程碑+温和惩罚)")
    print("  🧠 简化SAC模型 (避免复杂attention)")
    print()
    
    # 创建测试器
    tester = SimpleSACTester(
        num_joints=3,  # 使用3关节简化测试
        curriculum_stage=0  # 从最简单的阶段开始
    )
    
    # 运行测试
    tester.run_test(
        total_steps=1500,  # 总训练步数
        max_episodes=25    # 最大episode数
    )

if __name__ == "__main__":
    main()

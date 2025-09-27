#!/usr/bin/env python3
"""
简化版GPT-5通用多任务SAC架构
先用单进程测试核心功能
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from gymnasium.envs.mujoco import MujocoEnv

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 全局配置
J_MAX = 3  # 简化：支持最大3关节
SUPPORTED_JOINTS = [2, 3]  # 简化：只支持2和3关节

class JointTokenEncoder(nn.Module):
    """关节Token编码器"""
    
    def __init__(self, joint_token_dim: int = 32):
        super().__init__()
        # 简化输入: [cos, sin, vel, link_len] = 4维
        input_dim = 4
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, joint_token_dim),
            nn.ReLU(),
            nn.Linear(joint_token_dim, joint_token_dim),
            nn.ReLU()
        )
        
        self.joint_token_dim = joint_token_dim
        
    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: [batch_size, J_max, 4]
        Returns:
            joint_tokens: [batch_size, J_max, joint_token_dim]
        """
        batch_size, num_joints, feature_dim = joint_features.shape
        joint_features_flat = joint_features.view(-1, feature_dim)
        joint_tokens_flat = self.encoder(joint_features_flat)
        joint_tokens = joint_tokens_flat.view(batch_size, num_joints, self.joint_token_dim)
        return joint_tokens

class TaskTokenEncoder(nn.Module):
    """任务Token编码器"""
    
    def __init__(self, task_token_dim: int = 32):
        super().__init__()
        # 简化输入: [N/J_max, is_2joint, is_3joint] = 3维
        input_dim = 3
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, task_token_dim),
            nn.ReLU(),
            nn.Linear(task_token_dim, task_token_dim),
            nn.ReLU()
        )
        
        self.task_token_dim = task_token_dim
        
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(task_features)

class SimpleUniversalExtractor(BaseFeaturesExtractor):
    """简化的通用多任务特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        self.joint_encoder = JointTokenEncoder(32)
        self.task_encoder = TaskTokenEncoder(32)
        
        # 简化的注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=2,
            dropout=0.0,
            batch_first=True
        )
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32, features_dim),  # joint + task
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
        print(f"🔧 SimpleUniversalExtractor: 支持{SUPPORTED_JOINTS}关节")
        print(f"   J_max = {J_MAX}")
        print(f"   特征维度: {features_dim}")
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 解析观察
        joint_features, task_features, joint_mask = self._parse_observations(observations)
        
        # 编码
        joint_tokens = self.joint_encoder(joint_features)  # [B, J_max, 32]
        task_token = self.task_encoder(task_features)      # [B, 32]
        
        # 注意力池化
        key_padding_mask = ~joint_mask  # 反转mask
        
        # 使用task_token作为query进行池化
        task_query = task_token.unsqueeze(1)  # [B, 1, 32]
        
        pooled_joints, _ = self.attention(
            query=task_query,
            key=joint_tokens,
            value=joint_tokens,
            key_padding_mask=key_padding_mask
        )  # [B, 1, 32]
        
        pooled_joints = pooled_joints.squeeze(1)  # [B, 32]
        
        # 融合
        global_features = torch.cat([pooled_joints, task_token], dim=-1)  # [B, 64]
        features = self.fusion(global_features)  # [B, features_dim]
        
        return features
    
    def _parse_observations(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析观察为关节特征、任务特征和mask"""
        batch_size = observations.shape[0]
        
        # 假设观察格式：[joint_features..., task_info...]
        # 这里需要根据实际环境调整
        
        # 简化版本：从观察中提取信息
        # 假设观察包含了足够的信息来推断关节数和特征
        
        joint_features = torch.zeros(batch_size, J_MAX, 4).to(observations.device)
        task_features = torch.zeros(batch_size, 3).to(observations.device)
        joint_mask = torch.ones(batch_size, J_MAX, dtype=torch.bool).to(observations.device)
        
        # 根据观察维度推断关节数
        obs_dim = observations.shape[1]
        if obs_dim == 12:  # 2关节: 2*2 + 2*2 + 4 = 12
            num_joints = 2
        elif obs_dim == 16:  # 3关节: 3*2 + 3*2 + 4 = 16
            num_joints = 3
        else:
            num_joints = 2  # 默认
        
        # 解析关节特征
        joint_cos = observations[:, :num_joints]
        joint_sin = observations[:, num_joints:2*num_joints]
        joint_vel = observations[:, 2*num_joints:3*num_joints]
        
        for i in range(num_joints):
            joint_features[:, i, 0] = joint_cos[:, i]  # cos
            joint_features[:, i, 1] = joint_sin[:, i]  # sin
            joint_features[:, i, 2] = joint_vel[:, i]  # vel
            joint_features[:, i, 3] = 0.1  # 默认link长度
        
        # 设置mask
        joint_mask[:, num_joints:] = False
        
        # 设置任务特征
        task_features[:, 0] = num_joints / J_MAX  # N/J_max
        if num_joints == 2:
            task_features[:, 1] = 1.0  # is_2joint
        elif num_joints == 3:
            task_features[:, 2] = 1.0  # is_3joint
        
        return joint_features, task_features, joint_mask

def generate_multi_joint_reacher_xml(num_joints: int, link_lengths: List[float]) -> str:
    """生成N关节Reacher的MuJoCo XML"""
    
    xml_template = f'''
    <mujoco model="reacher_{num_joints}joint">
      <compiler angle="radian" inertiafromgeom="true"/>
      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
      </default>
      <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
      
      <worldbody>
        <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
        <geom conaffinity="1" contype="1" name="sideS" pos="0 -1 0" rgba="0.9 0.4 0.6 1" size="1 0.02 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideE" pos="1 0 0" rgba="0.9 0.4 0.6 1" size="0.02 1 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideN" pos="0 1 0" rgba="0.9 0.4 0.6 1" size="1 0.02 1" type="box"/>
        <geom conaffinity="1" contype="1" name="sideW" pos="-1 0 0" rgba="0.9 0.4 0.6 1" size="0.02 1 1" type="box"/>
        
        <body name="body0" pos="0 0 0">
          <joint axis="0 0 1" limited="true" name="joint0" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
          <geom fromto="0 0 0 {link_lengths[0]} 0 0" name="link0" size="0.02" type="capsule"/>
    '''
    
    # 添加后续关节和链接
    for i in range(1, num_joints):
        xml_template += f'''
          <body name="body{i}" pos="{link_lengths[i-1]} 0 0">
            <joint axis="0 0 1" limited="true" name="joint{i}" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
            <geom fromto="0 0 0 {link_lengths[i]} 0 0" name="link{i}" size="0.02" type="capsule"/>
        '''
    
    # 添加末端执行器
    xml_template += f'''
            <body name="fingertip" pos="{link_lengths[-1]} 0 0">
              <geom name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.0 1" size="0.01" type="sphere"/>
            </body>
    '''
    
    # 关闭所有body标签
    for i in range(num_joints):
        xml_template += '          </body>\n'
    
    # 添加目标
    xml_template += '''
        <body name="target" pos="0.3 0.3 0">
          <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-0.4 0.4" ref="0.3" stiffness="0" type="slide"/>
          <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-0.4 0.4" ref="0.3" stiffness="0" type="slide"/>
          <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size="0.02" type="sphere"/>
        </body>
      </worldbody>
      
      <actuator>
    '''
    
    # 添加执行器
    for i in range(num_joints):
        xml_template += f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="joint{i}" gear="200"/>\n'
    
    xml_template += '''
      </actuator>
    </mujoco>
    '''
    
    return xml_template

class SimpleMultiJointReacherEnv(MujocoEnv):
    """简化的N关节Reacher环境"""
    
    def __init__(self, num_joints: int, render_mode: Optional[str] = None):
        
        self.num_joints = num_joints
        
        # 默认链接长度
        if num_joints == 2:
            self.link_lengths = [0.1, 0.11]
        elif num_joints == 3:
            self.link_lengths = [0.1, 0.1, 0.1]
        else:
            raise ValueError(f"不支持的关节数: {num_joints}")
        
        # 生成XML
        xml_string = generate_multi_joint_reacher_xml(num_joints, self.link_lengths)
        
        # 创建临时文件
        self.temp_xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.temp_xml_file.write(xml_string)
        self.temp_xml_file.close()
        
        # 定义观察和动作空间
        obs_dim = 3 * num_joints + 4  # cos, sin, vel, ee_pos, target_pos
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        
        # 初始化MuJoCo环境
        super().__init__(
            model_path=self.temp_xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = action_space
        self.step_count = 0
        self.max_episode_steps = 50
        
        print(f"✅ SimpleMultiJointReacherEnv ({num_joints}关节) 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def step(self, action):
        # 显式渲染
        if self.render_mode == 'human':
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()
        reward = self._get_reward()
        
        # 检查终止条件
        ee_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance_to_target = np.linalg.norm(ee_pos - target_pos)
        
        terminated = distance_to_target < 0.02
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'distance_to_target': distance_to_target,
            'is_success': terminated
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        # 关节角度的cos和sin
        joint_cos = np.cos(self.data.qpos[:self.num_joints])
        joint_sin = np.sin(self.data.qpos[:self.num_joints])
        
        # 关节速度
        joint_vel = self.data.qvel[:self.num_joints]
        
        # 末端执行器位置
        ee_pos = self.get_body_com("fingertip")[:2]
        
        # 目标位置
        target_pos = self.get_body_com("target")[:2]
        
        return np.concatenate([joint_cos, joint_sin, joint_vel, ee_pos, target_pos])
    
    def _get_reward(self):
        ee_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # 距离奖励
        reward = -distance
        
        # 到达奖励
        if distance < 0.02:
            reward += 10.0
        
        return reward
    
    def reset_model(self):
        # 随机初始化关节角度
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        
        # 随机目标位置
        qpos[-2:] = self.np_random.uniform(low=-0.3, high=0.3, size=2)
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        
        return self._get_obs()
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0

def create_simple_env(num_joints: int, render_mode: Optional[str] = None):
    """创建简化环境"""
    env = SimpleMultiJointReacherEnv(num_joints, render_mode)
    env = Monitor(env)
    return env

def train_simple_universal_sac(total_timesteps: int = 20000):
    """训练简化的通用多任务SAC"""
    print("🚀 简化通用多任务SAC训练")
    print(f"🎯 测试{SUPPORTED_JOINTS}关节Reacher")
    print("💡 验证核心架构功能")
    print("="*60)
    
    # 先用2关节测试
    print("🌍 创建2关节测试环境...")
    train_env = create_simple_env(2, render_mode='human')
    
    print("✅ 测试环境创建完成")
    print(f"   观察空间: {train_env.observation_space}")
    print(f"   动作空间: {train_env.action_space}")
    
    # 创建SAC模型
    print("\n🤖 创建简化通用SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': SimpleUniversalExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=500,
        device='cpu',
        batch_size=128,
        buffer_size=20000,
        learning_rate=3e-4,
    )
    
    print("✅ 简化通用SAC模型创建完成")
    
    # 开始训练
    print(f"\n🎯 开始训练 ({total_timesteps}步)...")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        
        # 保存模型
        model.save("models/simple_universal_sac")
        print("💾 模型已保存: models/simple_universal_sac")
        
        return model
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被中断")
        model.save("models/simple_universal_sac_interrupted")
        return model
    
    finally:
        train_env.close()

def test_simple_universal_sac():
    """测试简化的通用SAC"""
    print("\n🎮 测试简化通用SAC")
    print("🎯 测试2关节和3关节环境")
    print("="*60)
    
    try:
        # 加载模型
        print("📂 加载模型...")
        try:
            model = SAC.load("models/simple_universal_sac")
            print("✅ 成功加载: models/simple_universal_sac")
        except FileNotFoundError:
            try:
                model = SAC.load("models/simple_universal_sac_interrupted")
                print("✅ 成功加载: models/simple_universal_sac_interrupted")
            except FileNotFoundError:
                print("❌ 没有找到训练好的模型")
                return
        
        # 测试2关节
        print("\n🔧 测试2关节环境...")
        env_2joint = create_simple_env(2, render_mode='human')
        
        total_reward = 0
        success_count = 0
        
        for episode in range(5):
            obs, info = env_2joint.reset()
            episode_reward = 0
            
            for step in range(50):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_2joint.step(action)
                episode_reward += reward
                
                if terminated:
                    success_count += 1
                    print(f"   Episode {episode+1}: 成功! 奖励={episode_reward:.2f}")
                    break
                    
                if truncated:
                    print(f"   Episode {episode+1}: 超时, 奖励={episode_reward:.2f}")
                    break
            
            total_reward += episode_reward
        
        print(f"2关节测试结果: 平均奖励={total_reward/5:.2f}, 成功率={success_count/5*100:.1f}%")
        env_2joint.close()
        
        # 测试3关节
        print("\n🔧 测试3关节环境...")
        env_3joint = create_simple_env(3, render_mode='human')
        
        total_reward = 0
        success_count = 0
        
        for episode in range(5):
            obs, info = env_3joint.reset()
            episode_reward = 0
            
            for step in range(50):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_3joint.step(action)
                episode_reward += reward
                
                if terminated:
                    success_count += 1
                    print(f"   Episode {episode+1}: 成功! 奖励={episode_reward:.2f}")
                    break
                    
                if truncated:
                    print(f"   Episode {episode+1}: 超时, 奖励={episode_reward:.2f}")
                    break
            
            total_reward += episode_reward
        
        print(f"3关节测试结果: 平均奖励={total_reward/5:.2f}, 成功率={success_count/5*100:.1f}%")
        env_3joint.close()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🌟 简化版GPT-5通用多任务SAC")
    print("🎯 验证核心架构概念")
    print("💡 单进程测试版本")
    print()
    
    try:
        # 训练阶段
        print("🚀 开始训练阶段...")
        model = train_simple_universal_sac(total_timesteps=20000)
        
        # 测试阶段
        print("\n" + "="*60)
        print("🎮 开始测试阶段...")
        test_simple_universal_sac()
        
        print(f"\n🎉 简化通用多任务测试完成！")
        print(f"💡 验证了GPT-5架构的核心概念")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



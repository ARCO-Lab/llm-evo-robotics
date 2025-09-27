#!/usr/bin/env python3
"""
纯Baseline SAC训练3关节Reacher
移除所有自定义组件，使用最简单的配置
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import tempfile
import os
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# 简单的3关节XML配置
def get_simple_3joint_xml():
    """最简单的3关节XML，基于标准Reacher"""
    return """
<mujoco model="reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 0.01">
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="cylinder"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="cylinder"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="cylinder"/>
          <body name="fingertip" pos="0.1 0 0">
            <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.2" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
  </actuator>
</mujoco>
"""

class Simple3JointReacherEnv(gym.Env):
    """最简单的3关节Reacher环境，完全模仿标准Reacher-v5"""
    
    def __init__(self, render_mode=None):
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_simple_3joint_xml())
        self.xml_file.flush()
        
        # 使用MuJoCo环境
        
        # 观察空间：11维 (3*cos + 3*sin + 3*vel + 2*fingertip_pos)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        
        # 继承MujocoEnv
        super().__init__()
        self.observation_space = observation_space
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 初始化MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.xml_file.name)
        self.data = mujoco.MjData(self.model)
        
        # 渲染设置
        self.render_mode = render_mode
        if render_mode == "human":
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None
            
        # 环境参数
        self.max_episode_steps = 100
        self.step_count = 0
        
        # 保存初始状态
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置到初始状态 + 小随机扰动
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        
        # 随机目标位置
        target_distance = self.np_random.uniform(0.05, 0.25)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        qpos[-2:] = [target_x, target_y]
        
        # 设置状态
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        # 执行动作
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=2)  # frame_skip=2
        
        # 渲染
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
        
        # 计算奖励 (完全按照标准Reacher-v5)
        fingertip_pos = self.data.geom('fingertip').xpos[:2]
        target_pos = self.data.geom('target').xpos[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 标准Reacher-v5奖励
        reward_dist = -distance
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        
        # 终止条件
        self.step_count += 1
        terminated = False  # 不提前终止
        truncated = self.step_count >= self.max_episode_steps
        
        # 信息
        info = {
            'reward_dist': reward_dist,
            'reward_ctrl': reward_ctrl,
            'distance_to_target': distance,
            'is_success': distance < 0.05,  # 5cm成功阈值
        }
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        # 观察：cos(θ), sin(θ), θ̇, fingertip_pos
        theta = self.data.qpos[:3]  # 3个关节角度
        obs = np.concatenate([
            np.cos(theta),                           # 3个cos值
            np.sin(theta),                           # 3个sin值  
            self.data.qvel[:3],                      # 3个关节速度
            self.data.geom('fingertip').xpos[:2],    # 末端位置(x,y)
        ])
        return obs
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if hasattr(self, 'xml_file'):
            os.unlink(self.xml_file.name)

def train_baseline_3joint():
    """训练纯baseline SAC"""
    print("🚀 开始训练纯Baseline SAC 3关节Reacher")
    print("📋 配置:")
    print("  - 移除所有自定义特征提取器")
    print("  - 使用标准SAC默认参数")
    print("  - 简化环境实现")
    print("  - 标准Reacher-v5奖励函数")
    
    # 创建环境
    env = Simple3JointReacherEnv(render_mode=None)
    env = Monitor(env)
    
    print(f"✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space.shape}")
    print(f"   动作空间: {env.action_space.shape}")
    
    # 创建纯baseline SAC模型
    model = SAC(
        'MlpPolicy',  # 使用标准MLP策略
        env,
        verbose=1,
        learning_rate=3e-4,  # 标准学习率
        buffer_size=1000000,  # 标准buffer大小
        batch_size=256,       # 标准batch大小
        tau=0.005,           # 标准tau
        gamma=0.99,          # 标准gamma
        train_freq=1,        # 标准训练频率
        gradient_steps=1,    # 标准梯度步数
        # 不使用任何自定义组件
    )
    
    print("✅ 纯Baseline SAC模型创建完成")
    print("   - 使用标准MlpPolicy")
    print("   - 所有参数为SAC默认值")
    print("   - 无自定义特征提取器")
    
    # 训练
    print("\n🎯 开始训练...")
    model.learn(total_timesteps=10000, progress_bar=True)
    
    # 保存模型
    model.save('models/baseline_3joint_sac')
    print("💾 模型已保存: models/baseline_3joint_sac.zip")
    
    # 快速测试
    print("\n🧪 快速测试...")
    success_count = 0
    rewards = []
    
    for i in range(10):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if info.get('is_success', False):
                episode_success = True
            
            if terminated or truncated:
                break
        
        if episode_success:
            success_count += 1
        
        rewards.append(episode_reward)
        print(f"  Episode {i+1}: 奖励={episode_reward:.1f}, 成功={'✅' if episode_success else '❌'}")
    
    print(f"\n📊 Baseline测试结果:")
    print(f"  成功率: {success_count/10:.1%}")
    print(f"  平均奖励: {np.mean(rewards):.1f}")
    
    env.close()
    return model

if __name__ == "__main__":
    train_baseline_3joint()

#!/usr/bin/env python3
"""
简单的3关节Reacher环境
直接基于标准MuJoCo Reacher-v5的XML，只添加一个关节
确保渲染和物理模拟正常工作
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import tempfile

def create_simple_3joint_xml():
    """
    创建简单的3关节Reacher XML
    直接基于标准reacher.xml，只添加第三个关节
    """
    xml_content = """
<mujoco model="reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.2 0.2 10" type="plane"/>
    <geom conaffinity="0" fromto="-.2 -.2 .01 .2 -.2 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .2 -.2 .01 .2  .2 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.2  .2 .01 .2  .2 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.2 -.2 .01 -.2  .2 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
          <body name="fingertip" pos="0.1 0 0">
            <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".1 -.1 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".1" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.1" stiffness="0" type="slide"/>
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
    return xml_content

class Simple3JointReacherEnv(MujocoEnv):
    """
    简单的3关节Reacher环境
    直接继承MujocoEnv，使用标准的渲染和物理配置
    """
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Simple3JointReacherEnv 初始化")
        
        # 创建临时XML文件
        xml_content = create_simple_3joint_xml()
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(xml_content)
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        
        # 定义观察和动作空间
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        # 13维观察: [cos0, cos1, cos2, sin0, sin1, sin2, vel0, vel1, vel2, ee_x, ee_y, target_x, target_y]
        
        # 初始化MujocoEnv
        super().__init__(
            self.xml_file.name,
            frame_skip=2,  # 与标准Reacher相同
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )
        
        # 设置动作空间
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 初始化计数器
        self.step_count = 0
        
        print("✅ Simple3JointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def step(self, action):
        """执行一步"""
        # 确保动作是正确的形状
        action = np.clip(action, -1.0, 1.0)
        
        # 执行动作
        self.do_simulation(action, self.frame_skip)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._compute_reward(obs, action)
        
        # 检查是否结束
        self.step_count += 1
        distance_to_target = np.linalg.norm(obs[9:11] - obs[11:13])
        
        terminated = distance_to_target < 0.05  # 成功条件
        truncated = self.step_count >= 50  # 最大步数，与标准Reacher相同
        
        info = {
            'distance_to_target': distance_to_target,
            'is_success': terminated
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        """重置模型"""
        # 重置关节位置（小随机扰动）
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        
        # 重置关节速度
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        
        self.set_state(qpos, qvel)
        
        # 重置计数器
        self.step_count = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        """获取观察"""
        # 获取关节角度的cos和sin
        cos_angles = np.cos(self.data.qpos[:3])  # 3个关节
        sin_angles = np.sin(self.data.qpos[:3])
        
        # 获取关节速度
        joint_velocities = self.data.qvel[:3]
        
        # 获取末端执行器位置
        fingertip_pos = self.data.body("fingertip").xpos[:2]
        
        # 获取目标位置
        target_pos = self.data.body("target").xpos[:2]
        
        # 组合观察
        obs = np.concatenate([
            cos_angles,      # [0:3]
            sin_angles,      # [3:6]
            joint_velocities, # [6:9]
            fingertip_pos,   # [9:11]
            target_pos       # [11:13]
        ])
        
        return obs
    
    def _compute_reward(self, obs, action):
        """计算奖励"""
        # 距离奖励（主要奖励）
        distance = np.linalg.norm(obs[9:11] - obs[11:13])
        distance_reward = -distance
        
        # 控制惩罚（鼓励平滑控制）
        control_penalty = -0.1 * np.sum(np.square(action))
        
        return distance_reward + control_penalty
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
            os.unlink(self.xml_file.name)

def test_simple_3joint_env():
    """测试简单3关节环境"""
    print("🧪 测试简单3关节Reacher环境")
    
    # 创建环境
    env = Simple3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建成功")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 测试几个episode
    for episode in range(3):
        print(f"\n📍 Episode {episode + 1}")
        obs, info = env.reset()
        print(f"   初始观察形状: {obs.shape}")
        
        episode_reward = 0
        for step in range(50):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if step % 10 == 0:
                print(f"   Step {step}: 距离={info['distance_to_target']:.3f}, 奖励={reward:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"   Episode结束: 总奖励={episode_reward:.3f}, 最终距离={info['distance_to_target']:.3f}")
    
    env.close()
    print("✅ 测试完成")

if __name__ == "__main__":
    test_simple_3joint_env()



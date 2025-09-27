#!/usr/bin/env python3
"""
简单的可视化位置标签训练脚本
使用MuJoCo的文本渲染功能显示end-effector位置
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import time
import tempfile
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 🎯 统一为标准Reacher-v5奖励参数
SUCCESS_THRESHOLD_2JOINT = 0.05  # 2关节保持原有阈值 5cm
SUCCESS_THRESHOLD_RATIO = 0.25   # 3+关节：k = 0.25，成功阈值与可达半径的比例
REWARD_NEAR_WEIGHT = 1.0         # 距离奖励权重（标准Reacher-v5）
REWARD_CONTROL_WEIGHT = 0.1      # 控制惩罚权重（标准Reacher-v5）
TARGET_MIN_RATIO = 0.15          # 目标最小距离比例（3+关节）
TARGET_MAX_RATIO = 0.85          # 目标最大距离比例（3+关节）

class SimpleVisualReacherEnv(MujocoEnv):
    """简单的可视化Reacher环境，使用文本显示位置信息"""
    
    def __init__(self, xml_content, num_joints, link_lengths, render_mode=None, **kwargs):
        self.num_joints = num_joints
        self.link_lengths = link_lengths
        
        # 🎯 计算可达半径R和统一的成功阈值（仅3+关节）
        self.max_reach = sum(link_lengths)
        if num_joints >= 3:
            self.success_threshold = SUCCESS_THRESHOLD_RATIO * self.max_reach
            self.use_unified_reward = True
        else:
            self.success_threshold = SUCCESS_THRESHOLD_2JOINT
            self.use_unified_reward = False
        
        # 创建临时XML文件
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(xml_content)
        self.xml_file.flush()
        
        # 计算观察空间维度
        obs_dim = num_joints * 3 + 4  # cos, sin, vel + ee_pos + target_pos
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            width=640,  # 更大的窗口以显示文本
            height=480
        )
        
        self.step_count = 0
        self.max_episode_steps = 100
        
        print(f"✅ {num_joints}关节简单可视化Reacher创建完成")
        print(f"   🎯 可达半径R: {self.max_reach:.3f}")
        print(f"   🎯 成功阈值: {self.success_threshold:.3f}")
        print(f"   📍 将在渲染窗口显示实时位置文本")
    
    def calculate_unified_target_min(self):
        return TARGET_MIN_RATIO * self.max_reach
    
    def calculate_unified_target_max(self):
        return TARGET_MAX_RATIO * self.max_reach
    
    def generate_unified_target(self):
        if self.use_unified_reward:
            min_distance = self.calculate_unified_target_min()
            max_distance = self.calculate_unified_target_max()
        else:
            max_distance = self.max_reach * 0.85
            min_distance = 0.05
        
        target_distance = self.np_random.uniform(min_distance, max_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward = self.reward(action)
        
        # 计算距离和位置信息
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 渲染并添加文本信息
        if self.render_mode == "human":
            self.render()
            self._add_text_overlay(fingertip_pos, target_pos, distance)
        
        terminated = False
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        normalized_distance = distance / self.max_reach if self.use_unified_reward else None
        
        info = {
            'distance_to_target': distance,
            'normalized_distance': normalized_distance,
            'is_success': distance < self.success_threshold,
            'max_reach': self.max_reach,
            'success_threshold': self.success_threshold,
            'use_unified_reward': self.use_unified_reward,
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _add_text_overlay(self, fingertip_pos, target_pos, distance):
        """在MuJoCo渲染窗口中添加文本覆盖层"""
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                # 使用MuJoCo的文本渲染功能
                if hasattr(self.viewer, 'add_overlay'):
                    # 清除之前的文本
                    self.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "", "")
                    
                    # 添加位置信息文本
                    success_status = "✅ SUCCESS" if distance < self.success_threshold else "❌ TRYING"
                    
                    text_lines = [
                        f"Step: {self.step_count}",
                        f"End-Effector: ({fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f})",
                        f"Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f})",
                        f"Distance: {distance:.3f}",
                        f"Status: {success_status}",
                        f"Threshold: {self.success_threshold:.3f}"
                    ]
                    
                    text = "\n".join(text_lines)
                    self.viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Position Info", text)
                
                # 在控制台也输出（每30步一次）
                if self.step_count % 30 == 0:
                    joint_angles = self.data.qpos[:self.num_joints]
                    success_status = "✅" if distance < self.success_threshold else "❌"
                    print(f"📍 Step {self.step_count}: EE=({fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}), "
                          f"Target=({target_pos[0]:.3f}, {target_pos[1]:.3f}), 距离={distance:.3f} {success_status}")
                    
        except Exception as e:
            # 如果文本显示失败，在控制台输出
            if self.step_count % 20 == 0:
                success_status = "✅" if distance < self.success_threshold else "❌"
                print(f"📍 Step {self.step_count}: EE=({fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}), "
                      f"Target=({target_pos[0]:.3f}, {target_pos[1]:.3f}), 距离={distance:.3f} {success_status}")
    
    def reward(self, action):
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 统一使用标准Reacher-v5奖励函数
        distance_reward = -REWARD_NEAR_WEIGHT * distance
        control_penalty = -REWARD_CONTROL_WEIGHT * np.sum(np.square(action))
        total_reward = distance_reward + control_penalty
        
        return total_reward
    
    def _get_obs(self):
        theta = self.data.qpos.flat[:self.num_joints]
        obs = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qvel.flat[:self.num_joints],
            self.get_body_com("fingertip")[:2],
            self.get_body_com("target")[:2],
        ])
        return obs
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel.copy()
        
        qvel[:self.num_joints] += self.np_random.standard_normal(self.num_joints) * 0.1
        
        target_x, target_y = self.generate_unified_target()
        qpos[-2:] = [target_x, target_y]
        
        self.set_state(qpos, qvel)
        self.step_count = 0
        return self._get_obs()

def get_3joint_xml_with_visual_markers():
    """3关节XML配置，添加可视化标记"""
    return """
<mujoco model="3joint_reacher_visual">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
  </default>
  <contact>
    <pair geom1="link0" geom2="link2" condim="3"/>
    <pair geom1="fingertip" geom2="link0" condim="3"/>
    <pair geom1="fingertip" geom2="link1" condim="3"/>
  </contact>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.5 0.5 10" type="plane"/>
    <geom conaffinity="0" contype="0" fromto="-.5 -.5 .01 .5 -.5 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto=" .5 -.5 .01 .5  .5 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.5  .5 .01 .5  .5 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="-.5 -.5 .01 -.5 .5 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="1" conaffinity="1"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="2" conaffinity="2"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule" contype="4" conaffinity="4"/>
          <body name="fingertip" pos="0.11 0 0">
            <geom contype="16" conaffinity="16" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".015" type="sphere"/>
            <!-- 添加一个更明显的end-effector标记 -->
            <geom contype="0" conaffinity="0" name="ee_marker" pos="0 0 0.01" rgba="1 1 0 0.8" size=".008" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.5 .5" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.5 .5" ref="-.2" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".012" type="sphere"/>
      <!-- 添加一个更明显的目标标记 -->
      <geom conaffinity="0" contype="0" name="target_marker" pos="0 0 0.01" rgba="1 0 0 0.8" size=".008" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
  </actuator>
</mujoco>
"""

class SimpleVisual3JointReacherEnv(SimpleVisualReacherEnv):
    """简单可视化3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 SimpleVisual3JointReacherEnv 初始化")
        
        super().__init__(
            xml_content=get_3joint_xml_with_visual_markers(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )

def test_simple_visual_position():
    """测试简单可视化位置功能"""
    print("🎯 测试简单可视化位置功能")
    print("=" * 60)
    
    # 创建可视化环境
    env = SimpleVisual3JointReacherEnv(render_mode='human')
    env = Monitor(env)
    
    print("✅ 环境创建完成，开始测试...")
    print("📍 将在渲染窗口左上角显示实时位置信息")
    print("🎮 使用随机动作进行测试 (按Ctrl+C停止)")
    print()
    
    try:
        obs, info = env.reset()
        episode_count = 1
        
        for step in range(500):  # 测试500步
            # 使用小幅度随机动作
            action = env.action_space.sample() * 0.2
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 每150步显示episode信息
            if step % 150 == 0 and step > 0:
                print(f"\n🔄 Episode {episode_count}, 总步数: {step}")
                print(f"   当前距离: {info.get('distance_to_target', 'N/A'):.4f}")
                print(f"   成功状态: {'✅' if info.get('is_success', False) else '❌'}")
            
            # 重置episode
            if terminated or truncated:
                print(f"\n🏁 Episode {episode_count} 结束，重置环境...")
                obs, info = env.reset()
                episode_count += 1
            
            # 控制速度
            time.sleep(0.02)
        
        print(f"\n✅ 测试完成！共运行了 {episode_count} 个episodes")
        
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("✅ 环境已关闭")

def train_with_simple_visual():
    """带简单可视化的训练"""
    print("🚀 开始带简单可视化的3关节Reacher训练")
    print("=" * 60)
    
    # 创建训练环境
    train_env = SimpleVisual3JointReacherEnv(render_mode='human')
    train_env = Monitor(train_env)
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=2,
        learning_starts=1000,
        device='cpu',
        tensorboard_log="./tensorboard_logs/simple_visual_3joint/",
        batch_size=256,
        buffer_size=100000,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
    )
    
    print("✅ 模型创建完成，开始训练...")
    print("📍 训练过程中将在窗口左上角显示实时位置信息")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=15000,  # 较短的训练用于演示
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        
        # 保存模型
        model_path = "models/simple_visual_3joint_reacher"
        model.save(model_path)
        print(f"💾 模型已保存: {model_path}")
        
        return model
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        return model
    
    finally:
        train_env.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_with_simple_visual()
    else:
        test_simple_visual_position()


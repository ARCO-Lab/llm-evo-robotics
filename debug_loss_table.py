#!/usr/bin/env python3
"""
调试loss table显示问题
找出为什么自定义环境不显示训练统计
"""

import os
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'

def get_simple_3joint_xml():
    """简单的3关节XML"""
    return """
<mujoco model="debug_3joint">
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

class Debug3JointReacherEnv(MujocoEnv):
    """
    调试用的3关节Reacher环境
    添加正确的episode终止条件
    """
    
    def __init__(self, render_mode=None, **kwargs):
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_simple_3joint_xml())
        self.xml_file.flush()
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 添加episode计数器
        self.step_count = 0
        self.max_episode_steps = 50  # 与标准Reacher相同
    
    def step(self, action):
        """执行一步 - 添加正确的episode终止"""
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        
        # 计算奖励
        vec = self.data.body("fingertip").xpos[:2] - self.data.body("target").xpos[:2]
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        
        # 增加步数计数
        self.step_count += 1
        
        # 检查终止条件
        distance = np.linalg.norm(vec)
        terminated = distance < 0.02  # 成功条件
        truncated = self.step_count >= self.max_episode_steps  # 最大步数
        
        info = {
            'reward_dist': reward_dist, 
            'reward_ctrl': reward_ctrl,
            'distance_to_target': distance,
            'is_success': terminated
        }
        
        # 不渲染，专注于训练统计
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """获取观察"""
        theta = self.data.qpos.flat[:3]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qvel.flat[:3],
            self.data.body("fingertip").xpos[:2],
            self.data.body("target").xpos[:2],
        ])
    
    def reset_model(self):
        """重置模型"""
        # 重置步数计数器
        self.step_count = 0
        
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
            os.unlink(self.xml_file.name)

def debug_episode_behavior():
    """调试episode行为"""
    print("🔍 调试episode行为")
    
    env = Debug3JointReacherEnv()
    
    print("📊 测试episode长度和终止条件:")
    
    for episode in range(3):
        obs, info = env.reset()
        episode_length = 0
        total_reward = 0
        
        print(f"\n   Episode {episode + 1}:")
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_length += 1
            total_reward += reward
            
            if episode_length % 10 == 0:
                print(f"     Step {episode_length}: 距离={info['distance_to_target']:.3f}, 奖励={reward:.3f}")
            
            if terminated:
                print(f"     ✅ Episode成功终止 (距离 < 0.02)")
                break
            elif truncated:
                print(f"     ⏰ Episode达到最大步数")
                break
        
        print(f"     Episode长度: {episode_length}, 总奖励: {total_reward:.3f}")
    
    env.close()

def test_with_monitor():
    """测试Monitor包装的训练"""
    print("\n🧪 测试Monitor包装的训练")
    
    # 创建环境并用Monitor包装
    env = Debug3JointReacherEnv()
    env = Monitor(env)
    
    print("✅ 环境创建并Monitor包装完成")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        env,
        verbose=2,
        learning_starts=50,  # 更快开始学习
        device='cpu'
    )
    
    print("✅ SAC模型创建完成")
    print("🎯 开始训练 (1000步)...")
    print("💡 应该显示loss table")
    
    try:
        model.learn(
            total_timesteps=1000,
            log_interval=4
        )
        
        print("✅ 训练完成")
        
    except KeyboardInterrupt:
        print("⚠️ 训练被中断")
    
    finally:
        env.close()

def compare_environments():
    """对比标准Reacher和自定义环境"""
    print("\n📊 对比环境行为")
    
    # 测试标准Reacher
    print("\n1️⃣ 标准Reacher-v5:")
    standard_env = gym.make('Reacher-v5')
    
    obs, info = standard_env.reset()
    episode_lengths = []
    
    for episode in range(3):
        obs, info = standard_env.reset()
        length = 0
        
        while True:
            action = standard_env.action_space.sample()
            obs, reward, terminated, truncated, info = standard_env.step(action)
            length += 1
            
            if terminated or truncated:
                break
        
        episode_lengths.append(length)
        print(f"   Episode {episode + 1}: {length} 步")
    
    print(f"   平均episode长度: {np.mean(episode_lengths):.1f}")
    standard_env.close()
    
    # 测试自定义环境
    print("\n2️⃣ 自定义3关节:")
    custom_env = Debug3JointReacherEnv()
    
    episode_lengths = []
    
    for episode in range(3):
        obs, info = custom_env.reset()
        length = 0
        
        while True:
            action = custom_env.action_space.sample()
            obs, reward, terminated, truncated, info = custom_env.step(action)
            length += 1
            
            if terminated or truncated:
                break
        
        episode_lengths.append(length)
        print(f"   Episode {episode + 1}: {length} 步")
    
    print(f"   平均episode长度: {np.mean(episode_lengths):.1f}")
    custom_env.close()

def main():
    """主函数"""
    print("🌟 调试loss table显示问题")
    print("💡 找出为什么自定义环境不显示训练统计")
    print()
    
    try:
        # 1. 调试episode行为
        debug_episode_behavior()
        
        # 2. 对比环境
        compare_environments()
        
        # 3. 测试训练
        test_with_monitor()
        
        print(f"\n🎉 调试完成！")
        
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



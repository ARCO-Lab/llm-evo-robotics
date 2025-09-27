#!/usr/bin/env python3
"""
工作的3关节Reacher训练
使用修复后的渲染方案
"""

import os
import tempfile
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# 设置正确的渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

def get_working_3joint_xml():
    """获取工作的3关节XML"""
    return """
<mujoco model="working_3joint">
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

class Working3JointReacherEnv(MujocoEnv):
    """
    工作的3关节Reacher环境
    修复了渲染问题，FPS正常
    """
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Working3JointReacherEnv 初始化")
        
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_working_3joint_xml())
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        print("✅ Working3JointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def step(self, action):
        """执行一步 - 关键：显式调用render()"""
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        
        # 计算奖励
        vec = self.data.body("fingertip").xpos[:2] - self.data.body("target").xpos[:2]
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        
        terminated = False
        truncated = False
        info = {'reward_dist': reward_dist, 'reward_ctrl': reward_ctrl}
        
        # 关键：如果是human模式，显式调用render
        if self.render_mode == 'human':
            self.render()
        
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
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
            os.unlink(self.xml_file.name)

def test_working_3joint():
    """测试工作的3关节环境"""
    print("🧪 测试工作的3关节环境")
    
    # 创建环境
    env = Working3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建成功")
    
    # 测试关节运动
    print("\n🔧 测试关节运动 (每个关节5步):")
    obs, info = env.reset()
    
    test_actions = [
        ([1.0, 0.0, 0.0], "第1关节"),
        ([0.0, 1.0, 0.0], "第2关节"),
        ([0.0, 0.0, 1.0], "第3关节"),
        ([1.0, 1.0, 1.0], "所有关节"),
    ]
    
    for action, description in test_actions:
        print(f"\n   🎯 测试{description}:")
        
        for step in range(5):
            prev_obs = obs.copy()
            prev_angles = np.arctan2(prev_obs[3:6], prev_obs[0:3])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            new_angles = np.arctan2(obs[3:6], obs[0:3])
            angle_changes = new_angles - prev_angles
            
            print(f"     Step {step+1}: 角度变化 {np.degrees(angle_changes)}度, 奖励 {reward:.3f}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        # 暂停让您观察
        import time
        time.sleep(1.0)
    
    env.close()
    print("✅ 测试完成")

def train_working_3joint():
    """训练工作的3关节环境"""
    print("\n🚀 训练工作的3关节Reacher")
    
    # 创建训练环境 (无渲染，提高训练速度)
    train_env = Working3JointReacherEnv(render_mode=None)
    train_env = Monitor(train_env)
    
    # 创建评估环境 (有渲染，观察效果)
    eval_env = Working3JointReacherEnv(render_mode='human')
    
    print("✅ 训练和评估环境创建完成")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=2,
        learning_starts=100,
        device='cpu'
    )
    
    print("✅ SAC模型创建完成")
    print("🎯 开始训练 (5000步)...")
    print("💡 训练无渲染，评估时会显示渲染")
    
    try:
        import time
        start_time = time.time()
        
        model.learn(
            total_timesteps=5000,
            log_interval=4
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {5000/training_time:.1f}")
        
        # 保存模型
        model.save("models/working_3joint_reacher_sac")
        print("💾 模型已保存: models/working_3joint_reacher_sac")
        
        # 评估模型 (带渲染)
        print("\n🎮 评估模型 (带渲染):")
        obs, info = eval_env.reset()
        
        for step in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            if step % 10 == 0:
                distance = np.linalg.norm(obs[9:11] - obs[11:13])
                print(f"   Step {step}: 距离目标 {distance:.3f}, 奖励 {reward:.3f}")
            
            if terminated or truncated:
                obs, info = eval_env.reset()
        
        print("✅ 评估完成")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/working_3joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        train_env.close()
        eval_env.close()

def main():
    """主函数"""
    print("🌟 工作的3关节Reacher训练")
    print("💡 使用修复后的渲染方案，FPS正常")
    print()
    
    try:
        # 1. 测试环境
        test_working_3joint()
        
        # 2. 训练测试
        print("\n" + "="*50)
        print("环境测试完成！准备开始训练...")
        print("按Enter继续训练，Ctrl+C退出")
        print("="*50)
        input("按Enter继续...")
        
        train_working_3joint()
        
        print(f"\n🎉 所有测试完成！")
        print(f"💡 现在您有了一个正常工作的3关节Reacher环境")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

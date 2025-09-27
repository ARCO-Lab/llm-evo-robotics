#!/usr/bin/env python3
"""
扩大场地的3关节Reacher环境
为3关节机械臂提供更大的活动空间
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

def get_expanded_3joint_xml():
    """获取扩大场地的3关节XML"""
    return """
<mujoco model="expanded_3joint">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <!-- 扩大的场地：从0.2x0.2扩大到0.5x0.5 -->
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.5 0.5 10" type="plane"/>
    
    <!-- 扩大的边界 -->
    <geom conaffinity="0" fromto="-.5 -.5 .01 .5 -.5 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .5 -.5 .01 .5  .5 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.5  .5 .01 .5  .5 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.5 -.5 .01 -.5  .5 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    
    <!-- 3关节机械臂 -->
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
    
    <!-- 扩大目标活动范围：从±0.27扩大到±0.45 -->
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.45 .45" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.45 .45" ref="-.2" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".012" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
  </actuator>
</mujoco>
"""

class Expanded3JointReacherEnv(MujocoEnv):
    """
    扩大场地的3关节Reacher环境
    场地从0.2x0.2扩大到0.5x0.5
    目标范围从±0.27扩大到±0.45
    """
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Expanded3JointReacherEnv 初始化")
        
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_expanded_3joint_xml())
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        print(f"   场地尺寸: 1.0m x 1.0m (扩大2.5倍)")
        print(f"   目标范围: ±0.45m (扩大1.67倍)")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        print("✅ Expanded3JointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
        print(f"   机械臂总长度: 0.3m")
        print(f"   最大工作半径: 0.3m")
        print(f"   场地半径: 0.5m")
    
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

def test_expanded_environment():
    """测试扩大的环境"""
    print("🧪 测试扩大场地的3关节环境")
    
    # 创建环境
    env = Expanded3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建成功")
    
    # 测试大幅度运动
    print("\n🔧 测试大幅度关节运动:")
    obs, info = env.reset()
    
    # 显示初始状态
    fingertip_pos = obs[9:11]
    target_pos = obs[11:13]
    distance = np.linalg.norm(fingertip_pos - target_pos)
    
    print(f"   初始末端位置: [{fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}]")
    print(f"   初始目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
    print(f"   初始距离: {distance:.3f}m")
    
    # 测试极限运动
    extreme_actions = [
        ([1.0, 1.0, 1.0], "最大正向"),
        ([-1.0, -1.0, -1.0], "最大负向"),
        ([1.0, -1.0, 1.0], "混合动作1"),
        ([-1.0, 1.0, -1.0], "混合动作2"),
        ([0.5, 0.8, -0.3], "中等动作"),
    ]
    
    for action, description in extreme_actions:
        print(f"\n   🎯 测试{description} {action}:")
        
        # 连续执行10步观察运动范围
        for step in range(10):
            obs, reward, terminated, truncated, info = env.step(action)
            
            fingertip_pos = obs[9:11]
            target_pos = obs[11:13]
            distance = np.linalg.norm(fingertip_pos - target_pos)
            
            if step % 3 == 0:  # 每3步报告一次
                print(f"     Step {step+1}: 末端[{fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f}], 距离{distance:.3f}m, 奖励{reward:.3f}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        # 暂停观察
        import time
        time.sleep(1.5)
    
    # 测试工作空间覆盖
    print(f"\n📊 测试工作空间覆盖:")
    positions = []
    
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        fingertip_pos = obs[9:11]
        positions.append(fingertip_pos.copy())
        
        if terminated or truncated:
            obs, info = env.reset()
    
    positions = np.array(positions)
    
    print(f"   X范围: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"   Y范围: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"   最大距离原点: {np.linalg.norm(positions, axis=1).max():.3f}m")
    print(f"   场地边界: ±0.5m")
    
    if np.linalg.norm(positions, axis=1).max() < 0.5:
        print("   ✅ 机械臂完全在场地内活动")
    else:
        print("   ⚠️ 机械臂可能超出场地边界")
    
    env.close()
    print("✅ 测试完成")

def train_expanded_3joint():
    """训练扩大场地的3关节环境"""
    print("\n🚀 训练扩大场地的3关节Reacher")
    
    # 创建训练环境 (无渲染)
    train_env = Expanded3JointReacherEnv(render_mode=None)
    train_env = Monitor(train_env)
    
    # 创建评估环境 (有渲染)
    eval_env = Expanded3JointReacherEnv(render_mode='human')
    
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
    print("💡 扩大的场地应该能让机械臂更好地探索")
    
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
        model.save("models/expanded_3joint_reacher_sac")
        print("💾 模型已保存: models/expanded_3joint_reacher_sac")
        
        # 评估模型 (带渲染)
        print("\n🎮 评估模型 (带渲染，扩大场地):")
        obs, info = eval_env.reset()
        
        episode_rewards = []
        episode_distances = []
        
        for step in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            distance = np.linalg.norm(obs[9:11] - obs[11:13])
            episode_rewards.append(reward)
            episode_distances.append(distance)
            
            if step % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
                avg_distance = np.mean(episode_distances[-20:]) if len(episode_distances) >= 20 else np.mean(episode_distances)
                print(f"   Step {step}: 平均距离 {avg_distance:.3f}m, 平均奖励 {avg_reward:.3f}")
            
            if terminated or truncated:
                obs, info = eval_env.reset()
        
        final_avg_distance = np.mean(episode_distances)
        final_avg_reward = np.mean(episode_rewards)
        
        print(f"\n📊 评估结果:")
        print(f"   平均距离目标: {final_avg_distance:.3f}m")
        print(f"   平均奖励: {final_avg_reward:.3f}")
        print(f"   成功率 (<0.05m): {np.mean(np.array(episode_distances) < 0.05)*100:.1f}%")
        
        print("✅ 评估完成")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/expanded_3joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        train_env.close()
        eval_env.close()

def main():
    """主函数"""
    print("🌟 扩大场地的3关节Reacher")
    print("💡 场地从0.4x0.4扩大到1.0x1.0，目标范围也相应扩大")
    print()
    
    try:
        # 1. 测试扩大的环境
        test_expanded_environment()
        
        # 2. 训练测试
        print("\n" + "="*60)
        print("扩大场地测试完成！准备开始训练...")
        print("按Enter继续训练，Ctrl+C退出")
        print("="*60)
        input("按Enter继续...")
        
        train_expanded_3joint()
        
        print(f"\n🎉 所有测试完成！")
        print(f"💡 现在您有了一个扩大场地的3关节Reacher环境")
        print(f"🎯 机械臂可以在更大的空间中自由活动")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



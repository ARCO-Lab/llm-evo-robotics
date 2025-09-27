#!/usr/bin/env python3
"""
修复训练显示的3关节Reacher
确保显示loss table和训练进度
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
    <!-- 扩大的场地：1.0m x 1.0m -->
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
    
    <!-- 扩大目标活动范围 -->
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

class Fixed3JointReacherEnv(MujocoEnv):
    """
    修复训练显示的3关节Reacher环境
    """
    
    def __init__(self, render_mode=None, **kwargs):
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_expanded_3joint_xml())
        self.xml_file.flush()
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    def step(self, action):
        """执行一步"""
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
        
        # 只在human模式下渲染
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

def train_with_proper_logging():
    """训练并显示完整的loss table和进度"""
    print("🚀 3关节Reacher训练 - 完整日志显示")
    print("💡 扩大场地，显示训练进度和loss table")
    print()
    
    # 创建训练环境 (无渲染，提高训练速度)
    print("🌍 创建训练环境...")
    train_env = Fixed3JointReacherEnv(render_mode=None)
    train_env = Monitor(train_env)
    
    print("✅ 训练环境创建完成")
    print(f"   观察空间: {train_env.observation_space}")
    print(f"   动作空间: {train_env.action_space}")
    
    # 创建SAC模型 - 关键参数设置
    print("\n🤖 创建SAC模型...")
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=2,              # 重要：设置为2显示详细日志
        learning_starts=100,    # 100步后开始学习
        device='cpu',
        tensorboard_log="./tensorboard_logs/",  # 添加tensorboard日志
    )
    
    print("✅ SAC模型创建完成")
    print("   verbose=2: 显示详细训练日志")
    print("   learning_starts=100: 100步后开始学习")
    print("   tensorboard_log: 启用tensorboard日志")
    
    print("\n🎯 开始训练 (8000步)...")
    print("💡 应该能看到完整的loss table和进度条")
    print("📊 训练统计将每200步显示一次")
    print()
    
    try:
        import time
        start_time = time.time()
        
        # 训练模型 - 关键参数
        model.learn(
            total_timesteps=8000,
            log_interval=4,         # 重要：每4次更新显示一次日志
            progress_bar=True       # 显示进度条
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {8000/training_time:.1f}")
        
        # 保存模型
        model.save("models/fixed_3joint_reacher_sac")
        print("💾 模型已保存: models/fixed_3joint_reacher_sac")
        
        # 简单评估
        print("\n🎮 快速评估 (20步):")
        obs, info = train_env.reset()
        
        total_reward = 0
        distances = []
        
        for step in range(20):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = train_env.step(action)
            
            total_reward += reward
            distance = np.linalg.norm(obs[9:11] - obs[11:13])
            distances.append(distance)
            
            if step % 5 == 0:
                print(f"   Step {step}: 距离={distance:.3f}m, 奖励={reward:.3f}")
            
            if terminated or truncated:
                obs, info = train_env.reset()
        
        avg_distance = np.mean(distances)
        avg_reward = total_reward / 20
        
        print(f"\n📊 评估结果:")
        print(f"   平均距离: {avg_distance:.3f}m")
        print(f"   平均奖励: {avg_reward:.3f}")
        print(f"   成功率 (<0.05m): {np.mean(np.array(distances) < 0.05)*100:.1f}%")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/fixed_3joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        train_env.close()

def test_with_rendering():
    """测试训练好的模型并显示渲染"""
    print("\n🎮 测试训练好的模型 (带渲染)")
    
    try:
        # 加载模型
        print("📂 加载训练好的模型...")
        model = SAC.load("models/fixed_3joint_reacher_sac")
        print("✅ 模型加载成功")
        
        # 创建渲染环境
        print("🌍 创建渲染环境...")
        render_env = Fixed3JointReacherEnv(render_mode='human')
        
        print("✅ 渲染环境创建完成")
        print("🎯 开始测试 (50步，带渲染)...")
        
        obs, info = render_env.reset()
        
        episode_rewards = []
        episode_distances = []
        
        for step in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            
            distance = np.linalg.norm(obs[9:11] - obs[11:13])
            episode_rewards.append(reward)
            episode_distances.append(distance)
            
            if step % 10 == 0:
                print(f"   Step {step}: 距离={distance:.3f}m, 奖励={reward:.3f}")
            
            if terminated or truncated:
                obs, info = render_env.reset()
        
        avg_distance = np.mean(episode_distances)
        avg_reward = np.mean(episode_rewards)
        success_rate = np.mean(np.array(episode_distances) < 0.05) * 100
        
        print(f"\n📊 测试结果:")
        print(f"   平均距离: {avg_distance:.3f}m")
        print(f"   平均奖励: {avg_reward:.3f}")
        print(f"   成功率: {success_rate:.1f}%")
        
        render_env.close()
        
    except FileNotFoundError:
        print("❌ 没有找到训练好的模型，请先运行训练")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """主函数"""
    print("🌟 修复训练显示的3关节Reacher")
    print("💡 确保显示完整的loss table和训练进度")
    print()
    
    try:
        # 训练
        train_with_proper_logging()
        
        # 询问是否测试
        print("\n" + "="*50)
        print("训练完成！是否测试模型 (带渲染)?")
        print("按Enter测试，Ctrl+C退出")
        print("="*50)
        input("按Enter继续...")
        
        # 测试
        test_with_rendering()
        
        print(f"\n🎉 所有任务完成！")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



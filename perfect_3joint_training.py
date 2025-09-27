#!/usr/bin/env python3
"""
完美的3关节Reacher训练
同时显示loss table和进度条
"""

import os
import tempfile
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

def get_expanded_3joint_xml():
    """获取扩大场地的3关节XML"""
    return """
<mujoco model="perfect_3joint">
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

class Perfect3JointReacherEnv(MujocoEnv):
    """
    完美的3关节Reacher环境
    - 扩大场地
    - 正确的episode终止
    - 显示训练统计
    """
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Perfect3JointReacherEnv 初始化")
        
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(get_expanded_3joint_xml())
        self.xml_file.flush()
        
        print(f"   XML文件: {self.xml_file.name}")
        print(f"   场地尺寸: 1.0m x 1.0m")
        print(f"   目标范围: ±0.45m")
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        
        super().__init__(
            self.xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode
        )
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Episode管理
        self.step_count = 0
        self.max_episode_steps = 50  # 与标准Reacher相同
        
        print("✅ Perfect3JointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
        print(f"   最大episode步数: {self.max_episode_steps}")
    
    def step(self, action):
        """执行一步"""
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

def train_perfect_3joint():
    """完美的3关节训练 - 同时显示loss table和进度条"""
    print("🚀 完美的3关节Reacher训练")
    print("💡 同时显示loss table和进度条")
    print("🎯 扩大场地，正确的episode终止")
    print()
    
    # 创建训练环境 (带渲染)
    print("🌍 创建训练环境...")
    train_env = Perfect3JointReacherEnv(render_mode='human')
    train_env = Monitor(train_env)
    
    print("✅ 训练环境创建完成")
    
    # 创建SAC模型 - 关键参数设置
    print("\n🤖 创建SAC模型...")
    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=2,              # 显示详细日志 (loss table)
        learning_starts=100,    # 100步后开始学习
        device='cpu',
        tensorboard_log="./tensorboard_logs/",
    )
    
    print("✅ SAC模型创建完成")
    print("   ✅ verbose=2: 显示loss table")
    print("   ✅ learning_starts=100: 快速开始学习")
    print("   ✅ tensorboard_log: 启用详细日志")
    
    print("\n🎯 开始训练 (10000步)...")
    print("💡 您应该能看到:")
    print("   📊 Loss table (每200步显示)")
    print("   📈 进度条 (实时显示)")
    print("   🎮 训练统计 (episode长度、奖励、成功率)")
    print()
    
    try:
        import time
        start_time = time.time()
        
        # 训练模型 - 关键：同时设置两个参数
        model.learn(
            total_timesteps=30000,
            log_interval=4,         # 显示loss table
            progress_bar=True       # 显示进度条
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {10000/training_time:.1f}")
        
        # 保存模型
        model.save("models/perfect_3joint_reacher_sac")
        print("💾 模型已保存: models/perfect_3joint_reacher_sac")
        
        # 快速评估
        print("\n🎮 快速评估 (30步):")
        obs, info = train_env.reset()
        
        total_reward = 0
        distances = []
        successes = 0
        
        for step in range(30):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = train_env.step(action)
            
            total_reward += reward
            distance = info['distance_to_target']
            distances.append(distance)
            
            if info['is_success']:
                successes += 1
            
            if step % 10 == 0:
                print(f"   Step {step}: 距离={distance:.3f}m, 奖励={reward:.3f}")
            
            if terminated or truncated:
                obs, info = train_env.reset()
        
        avg_distance = np.mean(distances)
        avg_reward = total_reward / 30
        success_rate = successes / 30 * 100
        
        print(f"\n📊 训练后评估结果:")
        print(f"   平均距离: {avg_distance:.3f}m")
        print(f"   平均奖励: {avg_reward:.3f}")
        print(f"   成功率: {success_rate:.1f}%")
        
        if avg_distance < 0.1:
            print("   ✅ 训练效果良好!")
        elif avg_distance < 0.2:
            print("   🔶 训练效果一般，可以继续训练")
        else:
            print("   ⚠️ 训练效果较差，可能需要调整参数")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/perfect_3joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        train_env.close()

def test_with_rendering():
    """测试训练好的模型并显示渲染 - 10个episode，每个episode 100步"""
    print("\n🎮 测试训练好的模型 (带渲染)")
    
    try:
        # 加载模型
        print("📂 加载训练好的模型...")
        model = SAC.load("models/perfect_3joint_reacher_sac")
        print("✅ 模型加载成功")
        
        # 创建渲染环境
        print("🌍 创建渲染环境...")
        render_env = Perfect3JointReacherEnv(render_mode='human')
        
        print("✅ 渲染环境创建完成")
        print("🎯 开始测试 (10个episode，每个episode最多100步)...")
        print("💡 观察机械臂是否能成功到达目标")
        
        # 统计所有episode的结果
        all_episode_rewards = []
        all_episode_lengths = []
        all_episode_successes = []
        all_episode_final_distances = []
        
        for episode in range(10):
            print(f"\n📍 Episode {episode + 1}/10:")
            
            obs, info = render_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_success = False
            
            for step in range(100):  # 每个episode最多100步
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = render_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                distance = info['distance_to_target']
                
                # 每20步打印一次状态
                if step % 20 == 0:
                    print(f"   Step {step}: 距离={distance:.3f}m, 奖励={reward:.3f}")
                
                # 检查是否成功
                if info['is_success']:
                    episode_success = True
                    print(f"   ✅ 成功! 在第{step+1}步到达目标，距离={distance:.3f}m")
                    break
                
                # 检查是否结束
                if terminated or truncated:
                    final_distance = distance
                    if terminated and not episode_success:
                        print(f"   ⚠️ Episode结束，最终距离={final_distance:.3f}m")
                    break
            else:
                # 如果循环正常结束（没有break），说明达到了100步
                final_distance = distance
                print(f"   ⏰ 达到最大步数(100)，最终距离={final_distance:.3f}m")
            
            # 记录episode统计
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_episode_successes.append(episode_success)
            all_episode_final_distances.append(final_distance)
            
            print(f"   📊 Episode {episode + 1} 总结: 奖励={episode_reward:.2f}, 长度={episode_length}, 成功={'是' if episode_success else '否'}")
        
        # 计算总体统计
        avg_reward = np.mean(all_episode_rewards)
        avg_length = np.mean(all_episode_lengths)
        success_rate = np.mean(all_episode_successes) * 100
        avg_final_distance = np.mean(all_episode_final_distances)
        
        print(f"\n📊 完整测试结果 (10个episode):")
        print(f"   平均episode奖励: {avg_reward:.3f}")
        print(f"   平均episode长度: {avg_length:.1f}步")
        print(f"   平均最终距离: {avg_final_distance:.3f}m")
        print(f"   成功率: {success_rate:.1f}% ({int(success_rate/10)}/10 episodes)")
        
        # 性能评估
        if success_rate >= 80:
            print("   🎉 训练效果优秀!")
        elif success_rate >= 50:
            print("   ✅ 训练效果良好!")
        elif success_rate >= 20:
            print("   🔶 训练效果一般，可以继续训练")
        else:
            print("   ⚠️ 训练效果较差，建议调整参数或延长训练")
        
        # 详细统计
        successful_episodes = [i+1 for i, success in enumerate(all_episode_successes) if success]
        if successful_episodes:
            print(f"   🎯 成功的episode: {successful_episodes}")
        
        render_env.close()
        
    except FileNotFoundError:
        print("❌ 没有找到训练好的模型，请先运行训练")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """主函数"""
    print("🌟 完美的3关节Reacher训练系统")
    print("💡 同时显示loss table和进度条")
    print("🎯 扩大场地，正确的episode管理")
    print()
    
    try:
        # 训练
        train_perfect_3joint()
        
        # 询问是否测试
        print("\n" + "="*60)
        print("训练完成！是否测试模型 (带渲染)?")
        print("按Enter测试，Ctrl+C退出")
        print("="*60)
        input("按Enter继续...")
        
        # 测试
        test_with_rendering()
        
        print(f"\n🎉 所有任务完成！")
        print(f"💡 现在您有了一个完美工作的3关节Reacher环境")
        print(f"✅ 同时显示loss table和进度条")
        print(f"✅ 扩大场地适合3关节活动")
        print(f"✅ 正确的episode终止和统计")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

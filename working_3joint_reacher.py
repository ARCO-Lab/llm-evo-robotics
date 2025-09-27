#!/usr/bin/env python3
"""
工作的3关节Reacher
直接修改标准Reacher-v5的XML文件，添加第三个关节
确保使用相同的物理和渲染配置
"""

import os
import shutil
import tempfile
import gymnasium as gym
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv
from gymnasium.spaces import Box
import numpy as np

def get_3joint_reacher_xml():
    """获取3关节Reacher的XML内容，基于标准reacher.xml"""
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

class Working3JointReacherEnv(ReacherEnv):
    """
    工作的3关节Reacher环境
    继承标准ReacherEnv，只修改XML和观察/动作空间
    """
    
    def __init__(self, **kwargs):
        print("🌟 Working3JointReacherEnv 初始化")
        
        # 创建临时XML文件
        xml_content = get_3joint_reacher_xml()
        self.temp_xml = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.temp_xml.write(xml_content)
        self.temp_xml.flush()
        
        print(f"   临时XML: {self.temp_xml.name}")
        
        # 临时修改类的model_path
        original_model_path = self.__class__.metadata.get('model_path', None)
        self.__class__.metadata['model_path'] = self.temp_xml.name
        
        try:
            # 调用父类初始化
            super().__init__(**kwargs)
            
            # 修改动作空间为3维
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            
            # 修改观察空间为13维 (3个cos + 3个sin + 3个vel + 2个fingertip + 2个target)
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
            
            print("✅ Working3JointReacherEnv 创建完成")
            print(f"   观察空间: {self.observation_space}")
            print(f"   动作空间: {self.action_space}")
            
        finally:
            # 恢复原始model_path
            if original_model_path is not None:
                self.__class__.metadata['model_path'] = original_model_path
    
    def step(self, action):
        """重写step函数以支持3维动作"""
        # 确保动作是3维的
        action = np.clip(action, -1.0, 1.0)
        assert action.shape == (3,), f"Expected 3D action, got {action.shape}"
        
        # 执行仿真
        self.do_simulation(action, self.frame_skip)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励（使用父类的奖励函数逻辑）
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        
        # 检查终止条件
        terminated = False  # Reacher通常不会提前终止
        truncated = False   # 由TimeLimit包装器处理
        
        info = {
            'reward_dist': reward_dist,
            'reward_ctrl': reward_ctrl,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """重写观察函数以支持3关节"""
        theta = self.data.qpos.flat[:3]  # 3个关节角度
        return np.concatenate([
            np.cos(theta),                    # [0:3] cos angles
            np.sin(theta),                    # [3:6] sin angles  
            self.data.qvel.flat[:3],         # [6:9] joint velocities
            self.get_body_com("fingertip")[:2],  # [9:11] fingertip position
            self.get_body_com("target")[:2],     # [11:13] target position
        ])
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'temp_xml') and os.path.exists(self.temp_xml.name):
            os.unlink(self.temp_xml.name)

def test_working_3joint():
    """测试工作的3关节环境"""
    print("🧪 测试工作的3关节Reacher环境")
    
    # 创建环境
    env = Working3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建成功")
    
    # 测试FPS
    print("\n📊 FPS测试:")
    obs, info = env.reset()
    
    import time
    num_steps = 50
    start_time = time.time()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            current_fps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"   Step {step}: FPS = {current_fps:.1f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    total_time = time.time() - start_time
    fps = num_steps / total_time
    
    print(f"\n📈 FPS结果:")
    print(f"   平均FPS: {fps:.1f}")
    print(f"   每步时间: {total_time/num_steps*1000:.1f}ms")
    
    if fps < 100:
        print("✅ FPS正常!")
    else:
        print("⚠️ FPS仍然异常")
    
    # 测试关节运动
    print(f"\n🔧 关节运动测试:")
    obs, info = env.reset()
    
    test_actions = [
        [1.0, 0.0, 0.0],   # 第1关节
        [0.0, 1.0, 0.0],   # 第2关节
        [0.0, 0.0, 1.0],   # 第3关节
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n   动作 {i+1}: {action}")
        
        prev_obs = obs.copy()
        prev_angles = np.arctan2(prev_obs[3:6], prev_obs[0:3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        new_angles = np.arctan2(obs[3:6], obs[0:3])
        angle_changes = new_angles - prev_angles
        
        print(f"   角度变化: {np.degrees(angle_changes):.1f}度")
        print(f"   关节速度: {obs[6:9]:.3f}")
        print(f"   末端位置: {obs[9:11]:.3f}")
        
        time.sleep(1.0)  # 暂停观察
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("✅ 测试完成")

def train_working_3joint():
    """训练工作的3关节环境"""
    print("\n🚀 训练工作的3关节Reacher")
    
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    
    # 创建环境
    env = Working3JointReacherEnv(render_mode='human')
    env = Monitor(env)
    
    print("✅ 训练环境创建完成")
    
    # 创建SAC模型
    model = SAC(
        'MlpPolicy',
        env,
        verbose=2,
        learning_starts=100,
        device='cpu'
    )
    
    print("✅ SAC模型创建完成")
    print("🎯 开始训练 (3000步)...")
    print("💡 观察MuJoCo窗口中的3关节机器人训练")
    
    try:
        import time
        start_time = time.time()
        
        model.learn(
            total_timesteps=3000,
            log_interval=4
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {3000/training_time:.1f}")
        
        # 保存模型
        model.save("models/working_3joint_reacher_sac")
        print("💾 模型已保存: models/working_3joint_reacher_sac")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/working_3joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        env.close()

def main():
    """主函数"""
    print("🌟 工作的3关节Reacher测试")
    print("💡 继承标准ReacherEnv，确保物理和渲染正常")
    print()
    
    try:
        # 1. 基础测试
        test_working_3joint()
        
        # 2. 训练测试
        print("\n" + "="*50)
        print("准备开始训练测试...")
        print("如果FPS和关节运动正常，按Enter继续训练")
        print("如果仍有问题，按Ctrl+C退出")
        print("="*50)
        input("按Enter继续...")
        
        train_working_3joint()
        
        print(f"\n🎉 所有测试完成！")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

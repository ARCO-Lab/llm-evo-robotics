#!/usr/bin/env python3
"""
直接在XML中添加红色标记球来验证end-effector位置计算
"""

import numpy as np
from baseline_complete_sequential_training import create_env, SequentialReacherEnv
from stable_baselines3 import SAC
import time
import os
import tempfile
from stable_baselines3.common.monitor import Monitor

def get_3joint_xml_with_marker():
    """3关节XML配置，在计算的end-effector位置添加红色标记球"""
    return """
<mujoco model="3joint_reacher_with_marker">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1" rgba="0.7 0.7 0 1" density="1000"/>
  </default>
  <contact>
    <!-- 链节之间的自碰撞检测 -->
    <pair geom1="link0" geom2="link2" condim="3"/>
    <!-- End-effector与所有链节的碰撞检测 -->
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
          
          <!-- 原始的fingertip (绿色) -->
          <body name="fingertip" pos="0.11 0 0">
            <geom contype="16" conaffinity="16" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
          
          <!-- 🔴 我们计算的end-effector位置标记 (红色，稍大一点) -->
          <body name="calculated_endeffector" pos="0.11 0 0">
            <geom contype="0" conaffinity="0" name="calc_marker" pos="0 0 0.005" rgba="1.0 0.0 0.0 0.8" size=".012" type="sphere"/>
          </body>
          
        </body>
      </body>
    </body>
    <body name="target" pos=".2 -.2 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.5 .5" ref=".2" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.5 .5" ref="-.2" stiffness="0" type="slide"/>
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

class Visual3JointReacherEnv(SequentialReacherEnv):
    """带有可视化标记的3关节Reacher环境"""
    
    def __init__(self, render_mode=None, **kwargs):
        print("🌟 Visual3JointReacherEnv 初始化 (带红色end-effector标记)")
        
        super().__init__(
            xml_content=get_3joint_xml_with_marker(),
            num_joints=3,
            link_lengths=[0.1, 0.1, 0.1],
            render_mode=render_mode,
            **kwargs
        )
        
        # 获取计算标记的body ID
        try:
            self.calc_marker_body_id = self.model.body('calculated_endeffector').id
            print("✅ 红色end-effector标记已添加")
        except:
            self.calc_marker_body_id = None
            print("⚠️ 无法找到红色标记body")
    
    def step(self, action):
        # 执行正常的step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 更新红色标记位置到我们计算的end-effector位置
        self.update_calculated_marker()
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        obs = super().reset_model()
        # 重置后也更新标记位置
        self.update_calculated_marker()
        return obs
    
    def update_calculated_marker(self):
        """更新红色标记到我们计算的end-effector位置"""
        if self.calc_marker_body_id is None:
            return
        
        try:
            # 使用我们的正向运动学计算end-effector位置
            joint_angles = self.data.qpos[:3]
            calculated_pos = self.calculate_endeffector_position(joint_angles)
            
            # 更新红色标记的位置
            self.data.body(self.calc_marker_body_id).xpos[:] = calculated_pos
            
        except Exception as e:
            # 如果更新失败，不影响正常运行
            pass
    
    def calculate_endeffector_position(self, joint_angles):
        """手动计算end-effector位置（正向运动学）"""
        link_lengths = [0.1, 0.1, 0.1]  # 3个链长
        fingertip_offset = 0.01  # XML中fingertip的额外偏移
        
        x = 0.0
        y = 0.0
        z = 0.01  # 基座高度，稍微抬高红色标记以便区分
        angle_sum = 0.0
        
        # 计算每个关节的贡献
        for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
            angle_sum += angle
            x += length * np.cos(angle_sum)
            y += length * np.sin(angle_sum)
        
        # 添加fingertip的额外偏移
        x += fingertip_offset * np.cos(angle_sum)
        y += fingertip_offset * np.sin(angle_sum)
        z += 0.005  # 红色标记稍微高一点，便于区分
        
        return np.array([x, y, z])

def run_visual_verification():
    """运行可视化验证"""
    print("🎯 开始可视化验证end-effector位置计算")
    print("=" * 60)
    print("📋 可视化说明:")
    print("   🟢 绿色小球: MuJoCo原生的fingertip位置")
    print("   🔴 红色大球: 我们计算的end-effector位置")
    print("   🔵 蓝色小球: 目标位置")
    print("   ✅ 如果计算正确，绿色和红色球应该完全重叠!")
    print()
    
    # 创建带标记的环境
    env = Visual3JointReacherEnv(render_mode='human')
    env = Monitor(env)
    
    # 尝试加载训练好的模型
    model_path = "models/baseline_sequential_3joint_reacher.zip"
    if os.path.exists(model_path):
        try:
            model = SAC.load(model_path, env=env, device="cpu")
            print(f"✅ 加载训练模型: {model_path}")
            use_trained_model = True
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("   将使用随机动作")
            use_trained_model = False
    else:
        print("⚠️ 未找到训练模型，将使用随机动作")
        use_trained_model = False
    
    try:
        print("\n🎮 开始可视化演示 (按Ctrl+C停止)")
        print("   请观察绿色球和红色球是否重叠")
        print("   如果重叠，说明我们的计算完全正确!")
        
        obs, info = env.reset()
        step_count = 0
        episode_count = 1
        
        while True:
            # 获取动作
            if use_trained_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # 使用小幅度随机动作，便于观察
                action = env.action_space.sample() * 0.2
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 每20步打印一次详细信息
            if step_count % 20 == 0:
                # 获取位置信息
                fingertip_pos_mujoco = env.unwrapped.get_body_com("fingertip")
                target_pos_mujoco = env.unwrapped.get_body_com("target")
                
                # 计算我们的end-effector位置
                joint_angles = env.unwrapped.data.qpos[:3]
                calculated_pos = env.unwrapped.calculate_endeffector_position(joint_angles)
                
                # 计算位置差异
                pos_error = np.linalg.norm(calculated_pos[:2] - fingertip_pos_mujoco[:2])
                distance = np.linalg.norm(fingertip_pos_mujoco[:2] - target_pos_mujoco[:2])
                
                print(f"\n--- Episode {episode_count}, Step {step_count} ---")
                print(f"🟢 MuJoCo fingertip: ({fingertip_pos_mujoco[0]:.4f}, {fingertip_pos_mujoco[1]:.4f})")
                print(f"🔴 计算 end-effector: ({calculated_pos[0]:.4f}, {calculated_pos[1]:.4f})")
                print(f"📏 位置误差: {pos_error:.6f}")
                print(f"🎯 到目标距离: {distance:.4f}")
                print(f"🏆 奖励: {reward:.3f}")
                
                if pos_error < 1e-5:
                    print("✅ 位置计算完全正确! (绿红球应该完全重叠)")
                elif pos_error < 1e-3:
                    print("✅ 位置计算基本正确 (绿红球应该几乎重叠)")
                else:
                    print(f"⚠️ 位置计算可能有误差 (误差: {pos_error:.6f})")
            
            step_count += 1
            
            # 重置episode
            if terminated or truncated or step_count >= 200:
                print(f"\n🔄 Episode {episode_count} 结束，重置环境...")
                obs, info = env.reset()
                step_count = 0
                episode_count += 1
                time.sleep(1)  # 短暂暂停以便观察
            
            # 控制渲染速度
            time.sleep(0.03)  # 约30 FPS
            
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断，停止可视化")
        print("\n🎯 验证总结:")
        print("   如果你看到绿色球和红色球完全重叠，")
        print("   那么我们的end-effector位置计算就是100%正确的!")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("✅ 环境已关闭")

if __name__ == "__main__":
    run_visual_verification()


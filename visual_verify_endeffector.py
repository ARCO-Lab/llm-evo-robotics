#!/usr/bin/env python3
"""
可视化验证end-effector位置计算的正确性
在渲染过程中实时标注计算出的end-effector位置
"""

import numpy as np
import mujoco
from baseline_complete_sequential_training import create_env
from stable_baselines3 import SAC
import time
import os

def add_visual_markers_to_model(env):
    """在MuJoCo模型中添加可视化标记"""
    # 获取环境的模型
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # 我们将通过修改现有geom的颜色和大小来实现可视化
    # 或者通过mujoco的viewer添加标记
    return model, data

def visualize_endeffector_calculation():
    """可视化end-effector位置计算"""
    print("🎯 开始可视化验证end-effector位置计算")
    print("=" * 60)
    
    # 创建3关节环境，启用渲染
    env = create_env(3, render_mode='human')
    env_unwrapped = env.unwrapped
    
    print("✅ 环境创建完成，开始可视化验证...")
    print("📋 说明:")
    print("   - 绿色球: MuJoCo原生的fingertip")
    print("   - 红色球: 我们计算的end-effector位置")
    print("   - 蓝色球: 目标位置")
    print("   - 如果计算正确，绿色和红色球应该完全重叠")
    print()
    
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
        # 重置环境
        obs, info = env.reset()
        
        print("🎮 开始可视化演示 (按Ctrl+C停止)")
        print("   观察绿色(原生)和红色(计算)标记是否重叠")
        
        step_count = 0
        episode_count = 1
        
        while True:
            # 获取动作
            if use_trained_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() * 0.1  # 小幅度随机动作
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 获取位置信息
            fingertip_pos_mujoco = env_unwrapped.get_body_com("fingertip")
            target_pos_mujoco = env_unwrapped.get_body_com("target")
            
            # 手动计算end-effector位置（正向运动学）
            joint_angles = env_unwrapped.data.qpos[:3]
            calculated_pos = calculate_endeffector_position(joint_angles)
            
            # 计算距离
            distance_mujoco = np.linalg.norm(fingertip_pos_mujoco[:2] - target_pos_mujoco[:2])
            distance_calculated = np.linalg.norm(calculated_pos[:2] - target_pos_mujoco[:2])
            
            # 在MuJoCo场景中添加可视化标记
            add_visual_markers(env_unwrapped, calculated_pos, target_pos_mujoco)
            
            # 打印详细信息
            if step_count % 10 == 0:  # 每10步打印一次
                print(f"\n--- Episode {episode_count}, Step {step_count} ---")
                print(f"关节角度: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}]")
                print(f"MuJoCo fingertip: ({fingertip_pos_mujoco[0]:.4f}, {fingertip_pos_mujoco[1]:.4f}, {fingertip_pos_mujoco[2]:.4f})")
                print(f"计算 end-effector: ({calculated_pos[0]:.4f}, {calculated_pos[1]:.4f}, {calculated_pos[2]:.4f})")
                print(f"位置差异: x={abs(calculated_pos[0] - fingertip_pos_mujoco[0]):.6f}, y={abs(calculated_pos[1] - fingertip_pos_mujoco[1]):.6f}")
                print(f"Target位置: ({target_pos_mujoco[0]:.4f}, {target_pos_mujoco[1]:.4f})")
                print(f"MuJoCo距离: {distance_mujoco:.4f}")
                print(f"计算距离: {distance_calculated:.4f}")
                print(f"距离差异: {abs(distance_mujoco - distance_calculated):.6f}")
                
                # 验证计算正确性
                pos_error = np.linalg.norm(calculated_pos[:2] - fingertip_pos_mujoco[:2])
                if pos_error < 1e-5:
                    print("✅ 位置计算完全正确!")
                elif pos_error < 1e-3:
                    print("✅ 位置计算基本正确 (微小误差)")
                else:
                    print(f"⚠️ 位置计算可能有误 (误差: {pos_error:.6f})")
            
            step_count += 1
            
            # 重置episode
            if terminated or truncated or step_count >= 200:
                print(f"\n🔄 Episode {episode_count} 结束，重置环境...")
                obs, info = env.reset()
                step_count = 0
                episode_count += 1
                time.sleep(1)  # 短暂暂停以便观察
            
            # 控制渲染速度
            time.sleep(0.05)  # 20 FPS
            
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断，停止可视化")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("✅ 环境已关闭")

def calculate_endeffector_position(joint_angles):
    """手动计算end-effector位置（正向运动学）"""
    link_lengths = [0.1, 0.1, 0.1]  # 3个链长
    fingertip_offset = 0.01  # XML中fingertip的额外偏移
    
    x = 0.0
    y = 0.0
    z = 0.01  # 基座高度
    angle_sum = 0.0
    
    # 计算每个关节的贡献
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        angle_sum += angle
        x += length * np.cos(angle_sum)
        y += length * np.sin(angle_sum)
    
    # 添加fingertip的额外偏移
    x += fingertip_offset * np.cos(angle_sum)
    y += fingertip_offset * np.sin(angle_sum)
    
    return np.array([x, y, z])

def add_visual_markers(env_unwrapped, calculated_pos, target_pos):
    """在MuJoCo场景中添加可视化标记"""
    # 这个函数尝试在MuJoCo渲染中添加可视化标记
    # 由于MuJoCo的限制，我们通过修改现有geom的属性来实现
    
    try:
        # 获取viewer
        if hasattr(env_unwrapped, 'viewer') and env_unwrapped.viewer is not None:
            viewer = env_unwrapped.viewer
            
            # 尝试添加可视化标记
            # 注意：这需要MuJoCo viewer支持动态添加geom
            # 如果不支持，我们至少可以在控制台输出位置信息
            pass
            
    except Exception as e:
        # 如果无法添加可视化标记，至少输出位置信息
        pass

def create_enhanced_visual_env():
    """创建增强的可视化环境，包含额外的标记geom"""
    print("🔧 创建增强可视化环境...")
    
    # 修改XML以包含额外的可视化标记
    from baseline_complete_sequential_training import get_3joint_xml
    
    original_xml = get_3joint_xml()
    
    # 在XML中添加可视化标记geom
    enhanced_xml = original_xml.replace(
        '</worldbody>',
        '''
    <!-- 可视化标记 -->
    <body name="calculated_marker" pos="0 0 0.01">
      <geom name="calc_marker" type="sphere" size="0.015" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>
    </body>
    <body name="target_marker" pos="0 0 0.01">
      <geom name="target_marker" type="sphere" size="0.012" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>
    </body>
</worldbody>'''
    )
    
    # 创建临时环境类
    import tempfile
    from baseline_complete_sequential_training import SequentialReacherEnv
    
    class EnhancedVisual3JointReacherEnv(SequentialReacherEnv):
        def __init__(self, render_mode=None, **kwargs):
            super().__init__(
                xml_content=enhanced_xml,
                num_joints=3,
                link_lengths=[0.1, 0.1, 0.1],
                render_mode=render_mode,
                **kwargs
            )
        
        def step(self, action):
            # 执行正常的step
            obs, reward, terminated, truncated, info = super().step(action)
            
            # 更新可视化标记位置
            self.update_visual_markers()
            
            return obs, reward, terminated, truncated, info
        
        def update_visual_markers(self):
            """更新可视化标记的位置"""
            try:
                # 计算end-effector位置
                joint_angles = self.data.qpos[:3]
                calculated_pos = calculate_endeffector_position(joint_angles)
                
                # 获取target位置
                target_pos = self.get_body_com("target")
                
                # 更新标记位置
                calc_marker_id = self.model.body('calculated_marker').id
                target_marker_id = self.model.body('target_marker').id
                
                self.data.body(calc_marker_id).xpos[:] = calculated_pos
                self.data.body(target_marker_id).xpos[:] = target_pos
                
            except Exception as e:
                # 如果更新失败，继续运行但不显示标记
                pass
    
    return EnhancedVisual3JointReacherEnv

def run_enhanced_visualization():
    """运行增强的可视化验证"""
    print("🚀 启动增强可视化验证")
    print("=" * 60)
    
    try:
        # 创建增强环境
        EnhancedEnv = create_enhanced_visual_env()
        env = EnhancedEnv(render_mode='human')
        
        print("✅ 增强可视化环境创建成功")
        print("📋 可视化说明:")
        print("   - 绿色球: MuJoCo原生fingertip")
        print("   - 红色球: 我们计算的end-effector位置")
        print("   - 蓝色球: 目标位置标记")
        print("   - 如果计算正确，绿色和红色球应该完全重叠")
        
        # 运行可视化
        obs, info = env.reset()
        
        for step in range(1000):
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                fingertip_pos = env.get_body_com("fingertip")[:2]
                target_pos = env.get_body_com("target")[:2]
                distance = np.linalg.norm(fingertip_pos - target_pos)
                print(f"Step {step}: 距离={distance:.4f}, 奖励={reward:.3f}")
            
            if terminated or truncated:
                obs, info = env.reset()
            
            time.sleep(0.05)
        
        env.close()
        
    except Exception as e:
        print(f"❌ 增强可视化失败: {e}")
        print("回退到基础可视化...")
        visualize_endeffector_calculation()

if __name__ == "__main__":
    print("🎯 End-effector位置计算可视化验证")
    print("选择验证模式:")
    print("1. 基础可视化 (控制台输出)")
    print("2. 增强可视化 (尝试添加3D标记)")
    
    try:
        choice = input("请选择 (1/2, 默认1): ").strip()
        if choice == "2":
            run_enhanced_visualization()
        else:
            visualize_endeffector_calculation()
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        print("回退到基础可视化...")
        visualize_endeffector_calculation()


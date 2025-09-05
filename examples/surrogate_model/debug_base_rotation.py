#!/usr/bin/env python3
"""
调试基座关节旋转幅度小的问题
增加torque_to_speed_ratio，观察旋转效果
"""

import sys
import os
import numpy as np
import pygame
import time
import math

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def debug_base_rotation():
    """调试基座关节旋转"""
    print("🔧 调试基座关节旋转幅度")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'  # 减少日志输出
    )
    
    env.reset()
    
    # 直接修改环境的torque_to_speed_ratio来测试
    print(f"🔧 临时修改torque_to_speed_ratio: 0.01 → 0.1")
    
    print(f"\n🎮 增强旋转测试:")
    print("  使用 torque_to_speed_ratio = 0.1 (增加10倍)")
    print("  阶段1: 基座关节大力逆时针旋转 (action = -100)")
    print("  阶段2: 基座关节大力顺时针旋转 (action = +100)")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    phase_duration = 200  # 每个阶段200步
    
    # 记录初始角度
    initial_angle = env.bodies[0].angle
    last_angle = initial_angle
    
    while running and step_count < 600:  # 总共600步，3个阶段
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    step_count = 0
                    initial_angle = env.bodies[0].angle
                    last_angle = initial_angle
                    print("🔄 重置测试")
        
        # 确定当前阶段和动作
        current_phase = (step_count // phase_duration) + 1
        
        # 根据阶段设置动作
        actions = np.array([0.0, 0.0, 0.0, 0.0])
        phase_name = ""
        
        if current_phase == 1:
            # 阶段1: 大力逆时针旋转
            actions[0] = -100
            phase_name = "大力逆时针 ⟳"
        elif current_phase == 2:
            # 阶段2: 大力顺时针旋转
            actions[0] = 100
            phase_name = "大力顺时针 ⟲"
        else:
            # 阶段3: 停止
            actions[0] = 0
            phase_name = "停止 ⏸"
        
        # 临时修改torque_to_speed_ratio
        if step_count == 0:
            # 修改环境的step方法中的torque_to_speed_ratio
            import types
            original_step = env.step
            
            def patched_step(self, actions):
                # 复制原始step方法的逻辑，但修改torque_to_speed_ratio
                actions = np.clip(actions, -self.max_torque, self.max_torque)
                
                pre_step_velocities = []
                if self.explosion_detection:
                    for body in self.bodies:
                        pre_step_velocities.append({
                            'velocity': np.array(body.velocity),
                            'angular_velocity': body.angular_velocity
                        })
                
                # 🔧 使用更大的torque_to_speed_ratio
                torque_to_speed_ratio = 0.1  # 增加10倍！
                
                for i, torque in enumerate(actions):
                    if i < len(self.motors):
                        motor = self.motors[i]
                        target_angular_velocity = torque * torque_to_speed_ratio
                        motor.rate = float(target_angular_velocity)
                        
                        if i == 0:  # 基座关节调试信息
                            print(f"🔧 基座关节: action={torque:.1f} → target_vel={target_angular_velocity:.2f} rad/s ({np.degrees(target_angular_velocity):.1f}°/s)")

                self.space.step(self.dt)
                
                if self.explosion_detection and pre_step_velocities:
                    self._detect_and_fix_explosion(pre_step_velocities)
                
                observation = self._get_observation()
                reward = self._compute_reward()
                truncated = False
                info = self._build_info_dict()

                if self.gym_api_version == "old":
                    done = False  # 简化处理
                    return observation, reward, done, info
                else:
                    return observation, reward, False, truncated, info
            
            env.step = types.MethodType(patched_step, env)
            print("🔧 已应用增强的torque_to_speed_ratio")
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 渲染
        env.render()
        
        # 获取基座关节状态
        base_pos = env.bodies[0].position
        base_angle = env.bodies[0].angle
        base_angular_vel = env.bodies[0].angular_velocity
        
        # 计算角度变化
        angle_change = base_angle - initial_angle
        angle_change_deg = np.degrees(angle_change)
        
        # 计算这一步的角度变化
        step_angle_change = base_angle - last_angle
        step_angle_change_deg = np.degrees(step_angle_change)
        last_angle = base_angle
        
        # 显示实时信息
        phase_step = step_count % phase_duration
        progress = (phase_step / phase_duration) * 100
        
        info_texts = [
            f"步数: {step_count} / 600",
            f"阶段 {current_phase}/3: {phase_name}",
            f"进度: {progress:.0f}%",
            "",
            "🤖 基座关节状态:",
            f"位置: ({base_pos[0]:.0f}, {base_pos[1]:.0f})",
            f"当前角度: {np.degrees(base_angle):.1f}°",
            f"总角度变化: {angle_change_deg:.1f}°",
            f"本步角度变化: {step_angle_change_deg:.2f}°",
            f"角速度: {np.degrees(base_angular_vel):.1f}°/s",
            "",
            "🎮 当前动作:",
            f"基座关节: {actions[0]:.0f}",
            f"目标角速度: {actions[0] * 0.1:.1f} rad/s",
            f"目标角速度: {np.degrees(actions[0] * 0.1):.1f}°/s",
            "",
            f"🔍 旋转观察:",
            f"实际方向: {'逆时针 ⟳' if base_angular_vel < -0.1 else '顺时针 ⟲' if base_angular_vel > 0.1 else '静止 ⏸'}",
            f"期望方向: {'逆时针 ⟳' if actions[0] < -10 else '顺时针 ⟲' if actions[0] > 10 else '静止 ⏸'}",
            "",
            "🔧 调试信息:",
            f"torque_to_speed_ratio: 0.1",
            f"motor.max_force: 50000",
            "",
            "R: 重置 | Q: 退出"
        ]
        
        # 显示信息面板
        info_surface = pygame.Surface((450, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if f"阶段 {current_phase}/3" in text:
                    color = (100, 200, 255)
                elif "基座关节状态" in text:
                    color = (255, 200, 100)
                elif "总角度变化" in text and abs(angle_change_deg) > 10:
                    color = (100, 255, 100)  # 绿色表示有明显旋转
                elif "实际方向" in text:
                    color = (255, 100, 255)
                elif "逆时针 ⟳" in text:
                    color = (100, 255, 255)
                elif "顺时针 ⟲" in text:
                    color = (255, 255, 100)
                elif "调试信息" in text:
                    color = (255, 150, 150)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每50步输出统计
        if step_count % 50 == 0:
            print(f"\n📊 步数{step_count}统计 (阶段{current_phase}):")
            print(f"   总角度变化: {angle_change_deg:.1f}°")
            print(f"   当前角速度: {np.degrees(base_angular_vel):.1f}°/s")
            print(f"   动作: {actions[0]:.0f}")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终结果
    final_angle_change = np.degrees(env.bodies[0].angle - initial_angle)
    
    print(f"\n🎯 调试旋转测试结果:")
    print("=" * 40)
    print(f"测试步数: {step_count}")
    print(f"总角度变化: {final_angle_change:.1f}°")
    print(f"torque_to_speed_ratio: 0.1 (增加10倍)")
    
    if abs(final_angle_change) > 30:
        print(f"\n🎉 旋转幅度明显改善!")
        print("   建议修改环境中的torque_to_speed_ratio")
    else:
        print(f"\n🤔 旋转幅度仍然较小")
        print("   可能需要检查其他约束或参数")
    
    env.close()

if __name__ == "__main__":
    debug_base_rotation()

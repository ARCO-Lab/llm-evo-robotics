#!/usr/bin/env python3
"""
诊断基座关节旋转受限的具体原因
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

def diagnose_base_rotation():
    """诊断基座关节旋转问题"""
    print("🔍 诊断基座关节旋转受限问题")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )
    
    env.reset()
    
    # 🔍 检查基座关节的约束配置
    print(f"\n🔧 基座关节约束诊断:")
    print(f"   基座关节角度限制: {env.joint_angle_limits[0]}")
    print(f"   基座关节limit_joint: {env.joint_limits[0]}")
    
    # 🔍 检查虚拟锚点参数
    base_anchor = env.base_anchor_body
    print(f"\n🔧 虚拟锚点参数:")
    print(f"   质量: {base_anchor.mass}")
    print(f"   转动惯量: {base_anchor.moment}")
    print(f"   位置: {base_anchor.position}")
    
    # 🔍 检查锚点固定约束
    anchor_constraints = []
    for constraint in env.space.constraints:
        if hasattr(constraint, 'a') and hasattr(constraint, 'b'):
            if constraint.a == env.space.static_body and constraint.b == base_anchor:
                anchor_constraints.append(constraint)
                print(f"\n🔧 锚点固定约束:")
                print(f"   类型: {type(constraint).__name__}")
                if hasattr(constraint, 'stiffness'):
                    print(f"   刚度: {constraint.stiffness}")
                if hasattr(constraint, 'damping'):
                    print(f"   阻尼: {constraint.damping}")
    
    # 🔍 检查基座关节motor
    base_motor = env.motors[0]
    print(f"\n🔧 基座关节Motor:")
    print(f"   类型: {type(base_motor).__name__}")
    print(f"   max_force: {base_motor.max_force}")
    print(f"   当前rate: {base_motor.rate}")
    print(f"   连接: {type(base_motor.a).__name__} -> {type(base_motor.b).__name__}")
    
    # 🔍 检查基座关节joint
    base_joint = env.joints[0]
    print(f"\n🔧 基座关节Joint:")
    print(f"   类型: {type(base_joint).__name__}")
    print(f"   连接: {type(base_joint.a).__name__} -> {type(base_joint.b).__name__}")
    if hasattr(base_joint, 'collide_bodies'):
        print(f"   collide_bodies: {base_joint.collide_bodies}")
    
    # 🔍 运行旋转测试
    print(f"\n🎮 开始旋转测试...")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    test_duration = 100
    
    # 记录初始状态
    initial_angle = env.bodies[0].angle
    initial_anchor_angle = base_anchor.angle
    
    # 设置一个很大的motor rate测试
    base_motor.rate = -20.0  # 很大的角速度
    print(f"🔧 设置base_motor.rate = {base_motor.rate} rad/s")
    
    while running and step_count < test_duration:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # 执行物理步进
        env.space.step(env.dt)
        
        # 获取状态
        base_body = env.bodies[0]
        base_angle = base_body.angle
        base_angular_vel = base_body.angular_velocity
        
        anchor_angle = base_anchor.angle
        anchor_angular_vel = base_anchor.angular_velocity
        
        # 检查约束力
        constraint_impulse = 0
        for constraint in env.space.constraints:
            if hasattr(constraint, 'impulse'):
                constraint_impulse += abs(constraint.impulse)
        
        # 渲染
        env.render()
        
        # 显示诊断信息
        info_texts = [
            f"🔍 基座关节旋转诊断 - 步数: {step_count}/{test_duration}",
            "",
            "🤖 基座Link状态:",
            f"   角度: {np.degrees(base_angle):.1f}° (初始: {np.degrees(initial_angle):.1f}°)",
            f"   角速度: {np.degrees(base_angular_vel):.1f}°/s",
            f"   角度变化: {np.degrees(base_angle - initial_angle):.1f}°",
            "",
            "🔧 虚拟锚点状态:",
            f"   角度: {np.degrees(anchor_angle):.1f}° (初始: {np.degrees(initial_anchor_angle):.1f}°)",
            f"   角速度: {np.degrees(anchor_angular_vel):.1f}°/s",
            f"   角度变化: {np.degrees(anchor_angle - initial_anchor_angle):.1f}°",
            "",
            "⚙️ Motor状态:",
            f"   目标rate: {base_motor.rate:.1f} rad/s ({np.degrees(base_motor.rate):.0f}°/s)",
            f"   max_force: {base_motor.max_force}",
            "",
            "🔗 约束状态:",
            f"   总约束冲量: {constraint_impulse:.1f}",
            "",
            "🎯 诊断结果:",
        ]
        
        # 分析结果
        angle_change = abs(np.degrees(base_angle - initial_angle))
        if angle_change < 1:
            info_texts.append("   ❌ 基座关节几乎没有旋转")
            if abs(anchor_angular_vel) > 0.1:
                info_texts.append("   🔍 虚拟锚点在旋转 - 这不正常!")
            if constraint_impulse > 100:
                info_texts.append("   🔍 约束冲量很大 - 可能被约束限制")
        elif angle_change < 10:
            info_texts.append("   ⚠️ 基座关节旋转幅度很小")
        else:
            info_texts.append("   ✅ 基座关节正常旋转")
        
        info_texts.append("")
        info_texts.append("Q: 退出诊断")
        
        # 显示信息面板
        info_surface = pygame.Surface((600, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "基座关节旋转诊断" in text:
                    color = (100, 200, 255)
                elif "基座Link状态" in text or "虚拟锚点状态" in text:
                    color = (255, 200, 100)
                elif "❌" in text:
                    color = (255, 100, 100)
                elif "⚠️" in text:
                    color = (255, 255, 100)
                elif "✅" in text:
                    color = (100, 255, 100)
                elif "🔍" in text:
                    color = (255, 150, 255)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        clock.tick(60)
        step_count += 1
        
        # 每20步输出一次状态
        if step_count % 20 == 0:
            print(f"📊 步数{step_count}: 基座角度变化={angle_change:.1f}°, 角速度={np.degrees(base_angular_vel):.1f}°/s, 锚点角速度={np.degrees(anchor_angular_vel):.1f}°/s")
    
    # 最终诊断
    final_angle_change = abs(np.degrees(env.bodies[0].angle - initial_angle))
    final_anchor_change = abs(np.degrees(base_anchor.angle - initial_anchor_angle))
    
    print(f"\n🎯 最终诊断结果:")
    print(f"   基座关节最终角度变化: {final_angle_change:.1f}°")
    print(f"   虚拟锚点最终角度变化: {final_anchor_change:.1f}°")
    
    if final_angle_change < 5:
        print(f"   ❌ 基座关节旋转严重受限")
        if final_anchor_change > 1:
            print(f"   🔍 问题可能是: 虚拟锚点本身在旋转，而不是基座关节相对锚点旋转")
        else:
            print(f"   🔍 问题可能是: 虚拟锚点的约束过于刚硬，阻止了旋转")
    else:
        print(f"   ✅ 基座关节旋转正常")
    
    env.close()

if __name__ == "__main__":
    diagnose_base_rotation()


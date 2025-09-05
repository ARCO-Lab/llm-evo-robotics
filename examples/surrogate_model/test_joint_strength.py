#!/usr/bin/env python3
"""
测试Joint强度修复 - 验证Links挤压时Joint不会被扯开
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

def test_joint_strength():
    """测试Joint在挤压情况下的强度"""
    print("🔧 测试Joint强度修复效果")
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
    
    # 检查Joint设置
    print(f"\n🔍 Joint配置检查:")
    for i, joint in enumerate(env.joints):
        print(f"   Joint {i}: max_force = {joint.max_force}")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    test_phases = [
        {"name": "正常运动", "duration": 100, "action": [50, 30, 20, 10]},
        {"name": "强制挤压", "duration": 200, "action": [100, -100, 100, -100]},  # 反向运动造成挤压
        {"name": "极限挤压", "duration": 200, "action": [100, 100, -100, -100]},   # 更强的挤压
        {"name": "恢复运动", "duration": 100, "action": [20, 20, 20, 20]},
    ]
    
    current_phase = 0
    phase_step = 0
    explosion_count = 0
    joint_separation_detected = False
    
    # 记录初始关节距离
    initial_joint_distances = []
    for i in range(len(env.bodies) - 1):
        pos1 = env.bodies[i].position
        pos2 = env.bodies[i + 1].position
        distance = np.linalg.norm([pos2[0] - pos1[0], pos2[1] - pos1[1]])
        initial_joint_distances.append(distance)
    
    while running and current_phase < len(test_phases):
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    current_phase = 0
                    phase_step = 0
                    explosion_count = 0
                    joint_separation_detected = False
        
        # 获取当前测试阶段
        phase = test_phases[current_phase]
        actions = np.array(phase["action"], dtype=float)
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 渲染
        env.render()
        
        # 检测Joint分离
        current_joint_distances = []
        for i in range(len(env.bodies) - 1):
            pos1 = env.bodies[i].position
            pos2 = env.bodies[i + 1].position
            distance = np.linalg.norm([pos2[0] - pos1[0], pos2[1] - pos1[1]])
            current_joint_distances.append(distance)
            
            # 检测异常分离（距离增加超过50%）
            if distance > initial_joint_distances[i] * 1.5:
                if not joint_separation_detected:
                    joint_separation_detected = True
                    print(f"⚠️ 检测到Joint {i}异常分离! 距离: {distance:.1f} (初始: {initial_joint_distances[i]:.1f})")
        
        # 检测炸开现象
        max_velocity = max([np.linalg.norm(body.velocity) for body in env.bodies])
        max_angular_velocity = max([abs(body.angular_velocity) for body in env.bodies])
        
        if max_velocity > 300 or max_angular_velocity > 15:
            explosion_count += 1
            if explosion_count == 1:
                print(f"💥 检测到炸开现象! 最大速度: {max_velocity:.1f}, 最大角速度: {np.degrees(max_angular_velocity):.1f}°/s")
        
        # 显示实时信息
        info_texts = [
            f"🔧 Joint强度测试",
            f"步数: {step_count}",
            "",
            f"🎯 当前阶段: {phase['name']} ({phase_step}/{phase['duration']})",
            f"动作: {actions}",
            "",
            "📊 Joint状态:",
        ]
        
        # 显示每个Joint的距离状态
        for i, distance in enumerate(current_joint_distances):
            initial_dist = initial_joint_distances[i]
            change_percent = ((distance - initial_dist) / initial_dist) * 100
            status = "✅ 正常" if abs(change_percent) < 20 else "⚠️ 异常" if abs(change_percent) < 50 else "❌ 分离"
            info_texts.append(f"   Joint {i}: {distance:.1f}px ({change_percent:+.1f}%) {status}")
        
        info_texts.extend([
            "",
            "🔍 系统状态:",
            f"   最大速度: {max_velocity:.1f} px/s",
            f"   最大角速度: {np.degrees(max_angular_velocity):.1f}°/s",
            f"   炸开次数: {explosion_count}",
            f"   Joint分离: {'是' if joint_separation_detected else '否'}",
            "",
            "💡 测试目标:",
            "   验证Joint在挤压时不会被扯开",
            "   验证机器人结构完整性",
            "",
            "R: 重置 | Q: 退出"
        ])
        
        # 显示信息面板
        info_surface = pygame.Surface((500, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "Joint强度测试" in text:
                    color = (100, 200, 255)
                elif "当前阶段" in text:
                    if "正常运动" in text:
                        color = (100, 255, 100)
                    elif "挤压" in text:
                        color = (255, 200, 100)
                    else:
                        color = (255, 255, 100)
                elif "✅ 正常" in text:
                    color = (100, 255, 100)
                elif "⚠️ 异常" in text:
                    color = (255, 255, 100)
                elif "❌ 分离" in text:
                    color = (255, 100, 100)
                elif "炸开次数" in text and explosion_count > 0:
                    color = (255, 150, 150)
                elif "Joint分离" in text and joint_separation_detected:
                    color = (255, 100, 100)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        phase_step += 1
        
        # 切换到下一阶段
        if phase_step >= phase["duration"]:
            print(f"✅ 完成阶段: {phase['name']}")
            current_phase += 1
            phase_step = 0
            
            if current_phase < len(test_phases):
                print(f"🔄 进入阶段: {test_phases[current_phase]['name']}")
        
        clock.tick(60)
    
    # 最终测试结果
    print(f"\n🎯 Joint强度测试结果:")
    print(f"   总测试步数: {step_count}")
    print(f"   炸开次数: {explosion_count}")
    print(f"   Joint分离检测: {'是' if joint_separation_detected else '否'}")
    
    # 最终Joint距离检查
    print(f"\n📊 最终Joint距离:")
    for i, distance in enumerate(current_joint_distances):
        initial_dist = initial_joint_distances[i]
        change_percent = ((distance - initial_dist) / initial_dist) * 100
        print(f"   Joint {i}: {distance:.1f}px (变化: {change_percent:+.1f}%)")
    
    if explosion_count == 0 and not joint_separation_detected:
        print(f"\n🎉 测试通过! Joint强度修复成功")
        print(f"   ✅ 无炸开现象")
        print(f"   ✅ 无Joint异常分离")
        print(f"   ✅ 机器人结构保持完整")
    else:
        print(f"\n⚠️ 测试发现问题:")
        if explosion_count > 0:
            print(f"   ❌ 检测到{explosion_count}次炸开现象")
        if joint_separation_detected:
            print(f"   ❌ 检测到Joint异常分离")
        print(f"   🔧 可能需要进一步调整Joint参数")
    
    env.close()

if __name__ == "__main__":
    test_joint_strength()


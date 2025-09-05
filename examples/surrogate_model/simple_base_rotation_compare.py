#!/usr/bin/env python3
"""
简化的基座关节旋转对比测试
对比虚拟锚点设计和调整max_force的效果
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

def test_rotation_with_settings(title, max_force, torque_ratio, test_duration=200):
    """测试不同参数设置下的基座关节旋转效果"""
    print(f"\n🎮 测试: {title}")
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
    
    # 🔧 调整motor max_force
    for i, motor in enumerate(env.motors):
        motor.max_force = max_force
        print(f"🔧 Motor {i} max_force: {motor.max_force}")
    
    # 临时修改step方法
    import types
    
    def patched_step(self, actions):
        actions = np.clip(actions, -self.max_torque, self.max_torque)
        
        pre_step_velocities = []
        if self.explosion_detection:
            for body in self.bodies:
                pre_step_velocities.append({
                    'velocity': np.array(body.velocity),
                    'angular_velocity': body.angular_velocity
                })
        
        # 🔧 使用指定的torque_to_speed_ratio
        for i, torque in enumerate(actions):
            if i < len(self.motors):
                motor = self.motors[i]
                target_angular_velocity = torque * torque_ratio
                motor.rate = float(target_angular_velocity)
                
                if i == 0 and step_count % 30 == 0:  # 每30步输出一次基座关节信息
                    print(f"🔧 基座关节: action={torque:.0f} → target_vel={target_angular_velocity:.2f} rad/s ({np.degrees(target_angular_velocity):.0f}°/s)")

        self.space.step(self.dt)
        
        if self.explosion_detection and pre_step_velocities:
            self._detect_and_fix_explosion(pre_step_velocities)
        
        observation = self._get_observation()
        reward = self._compute_reward()
        truncated = False
        info = self._build_info_dict()

        return observation, reward, False, info
    
    env.step = types.MethodType(patched_step, env)
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    
    # 记录初始角度
    initial_angle = env.bodies[0].angle
    max_angle_change = 0
    angle_history = []
    
    while running and step_count < test_duration:
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
                    max_angle_change = 0
                    angle_history = []
        
        # 设置动作：基座关节持续逆时针旋转
        actions = np.array([0.0, 0.0, 0.0, 0.0])
        actions[0] = -100  # 基座关节逆时针旋转
        
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
        max_angle_change = max(max_angle_change, abs(angle_change_deg))
        angle_history.append(angle_change_deg)
        
        # 显示实时信息
        info_texts = [
            f"🔧 测试: {title}",
            f"步数: {step_count} / {test_duration}",
            "",
            "🤖 基座关节状态:",
            f"位置: ({base_pos[0]:.0f}, {base_pos[1]:.0f})",
            f"当前角度: {np.degrees(base_angle):.1f}°",
            f"总角度变化: {angle_change_deg:.1f}°",
            f"最大角度变化: {max_angle_change:.1f}°",
            f"角速度: {np.degrees(base_angular_vel):.1f}°/s",
            "",
            "🎮 当前动作:",
            f"基座关节: {actions[0]:.0f}",
            f"目标角速度: {actions[0] * torque_ratio:.2f} rad/s",
            f"目标角速度: {np.degrees(actions[0] * torque_ratio):.0f}°/s",
            "",
            f"🔍 旋转评估:",
            f"实际方向: {'逆时针 ⟳' if base_angular_vel < -0.1 else '顺时针 ⟲' if base_angular_vel > 0.1 else '静止 ⏸'}",
            f"旋转效果: {'优秀' if max_angle_change > 45 else '良好' if max_angle_change > 15 else '较差' if max_angle_change > 5 else '极差'}",
            "",
            "🔧 测试参数:",
            f"motor.max_force: {max_force}",
            f"torque_to_speed_ratio: {torque_ratio}",
            "",
            "R: 重置 | Q: 退出"
        ]
        
        # 显示信息面板
        info_surface = pygame.Surface((500, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if title in text:
                    color = (100, 200, 255)
                elif "基座关节状态" in text:
                    color = (255, 200, 100)
                elif "总角度变化" in text:
                    if abs(angle_change_deg) > 45:
                        color = (100, 255, 100)  # 绿色表示大幅旋转
                    elif abs(angle_change_deg) > 15:
                        color = (255, 255, 100)  # 黄色表示中等旋转
                    else:
                        color = (255, 100, 100)  # 红色表示小幅旋转
                elif "旋转效果" in text:
                    if "优秀" in text:
                        color = (100, 255, 100)
                    elif "良好" in text:
                        color = (255, 255, 100)
                    elif "较差" in text:
                        color = (255, 150, 100)
                    else:
                        color = (255, 100, 100)
                elif "逆时针 ⟳" in text:
                    color = (100, 255, 255)
                elif "测试参数" in text:
                    color = (255, 150, 150)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每50步输出统计
        if step_count % 50 == 0:
            avg_angle_change = np.mean(angle_history[-50:]) if len(angle_history) >= 50 else np.mean(angle_history)
            print(f"📊 步数{step_count}: 角度变化={angle_change_deg:.1f}°, 最大={max_angle_change:.1f}°, 平均={avg_angle_change:.1f}°, 角速度={np.degrees(base_angular_vel):.1f}°/s")
        
        clock.tick(60)
    
    # 最终结果
    final_angle_change = np.degrees(env.bodies[0].angle - initial_angle)
    avg_angle_change = np.mean(angle_history) if angle_history else 0
    
    result = {
        'title': title,
        'max_force': max_force,
        'torque_ratio': torque_ratio,
        'steps': step_count,
        'final_angle_change': final_angle_change,
        'max_angle_change': max_angle_change,
        'avg_angle_change': avg_angle_change,
        'final_angular_velocity': np.degrees(env.bodies[0].angular_velocity)
    }
    
    print(f"\n🎯 {title} 测试结果:")
    print(f"   测试步数: {result['steps']}")
    print(f"   最终角度变化: {result['final_angle_change']:.1f}°")
    print(f"   最大角度变化: {result['max_angle_change']:.1f}°")
    print(f"   平均角度变化: {result['avg_angle_change']:.1f}°")
    print(f"   最终角速度: {result['final_angular_velocity']:.1f}°/s")
    
    env.close()
    return result

def compare_base_rotation_settings():
    """对比不同参数设置下的基座关节旋转效果"""
    print("🎯 基座关节旋转参数对比测试")
    print("=" * 60)
    
    results = []
    
    # 测试配置
    test_configs = [
        {
            'title': '当前虚拟锚点设计 (max_force=50000)',
            'max_force': 50000,
            'torque_ratio': 0.05
        },
        {
            'title': '统一max_force设计 (max_force=30000)', 
            'max_force': 30000,
            'torque_ratio': 0.05
        },
        {
            'title': '增强torque_ratio (max_force=30000)',
            'max_force': 30000,
            'torque_ratio': 0.1
        },
        {
            'title': '双倍增强 (max_force=60000, ratio=0.1)',
            'max_force': 60000,
            'torque_ratio': 0.1
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{i+1}️⃣ 测试配置 {i+1}/4")
        result = test_rotation_with_settings(
            config['title'],
            config['max_force'], 
            config['torque_ratio'],
            200
        )
        results.append(result)
        
        # 等待一下
        time.sleep(1)
    
    # 对比结果
    print("\n" + "=" * 60)
    print("🏆 参数对比结果总结")
    print("=" * 60)
    
    for result in results:
        print(f"\n📊 {result['title']}:")
        print(f"   参数: max_force={result['max_force']}, ratio={result['torque_ratio']}")
        print(f"   最大角度变化: {result['max_angle_change']:.1f}°")
        print(f"   平均角度变化: {result['avg_angle_change']:.1f}°")
        print(f"   最终角速度: {result['final_angular_velocity']:.1f}°/s")
        
        # 评估效果
        if result['max_angle_change'] > 45:
            print(f"   评估: 🎉 优秀 - 基座关节旋转效果良好")
        elif result['max_angle_change'] > 15:
            print(f"   评估: 👍 良好 - 基座关节有明显旋转")
        elif result['max_angle_change'] > 5:
            print(f"   评估: ⚠️ 较差 - 基座关节旋转幅度小")
        else:
            print(f"   评估: ❌ 极差 - 基座关节几乎不旋转")
    
    # 找出最佳配置
    best_result = max(results, key=lambda x: x['max_angle_change'])
    
    print(f"\n💡 最佳配置推荐:")
    print(f"   🏆 {best_result['title']}")
    print(f"   📈 最大角度变化: {best_result['max_angle_change']:.1f}°")
    print(f"   🔧 参数: max_force={best_result['max_force']}, torque_ratio={best_result['torque_ratio']}")
    
    print(f"\n🎯 结论:")
    if best_result['max_angle_change'] > 30:
        print("   ✅ 找到了有效的参数配置，基座关节可以正常旋转")
        print("   🔧 建议将此参数应用到主环境中")
    else:
        print("   ❌ 所有配置的旋转效果都不理想")
        print("   🔧 可能需要检查虚拟锚点的物理约束设置")

if __name__ == "__main__":
    compare_base_rotation_settings()



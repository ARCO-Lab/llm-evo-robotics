#!/usr/bin/env python3
"""
对比原始基座关节和虚拟锚点基座关节的旋转效果
确保所有关节max_force一致
"""

import sys
import os
import numpy as np
import pygame
import time
import math
import pymunk

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def create_original_env():
    """创建原始基座关节环境（直接连接static_body）"""
    
    class OriginalReacher2DEnv(Reacher2DEnv):
        def _create_robot(self):
            """重写_create_robot方法，使用原始基座关节设计"""
            print("🔧 创建原始基座关节设计...")
            
            # 清空现有数据
            self.bodies = []
            self.joints = []
            self.motors = []
            self.joint_limits = []
            
            # 🔧 从锚点开始构建机器人，每个link都有明确的位置
            current_pos = list(self.anchor_point)  # [x, y]
            
            for i in range(self.num_links):
                # 🔧 创建link body
                body = self.space.static_body if i == 0 else self.bodies[i-1]
                
                if i == 0:
                    # 基座link，固定在锚点
                    # 🔧 手动创建link body
                    moment = pymunk.moment_for_circle(self.link_mass, 0, self.link_radius)
                    link_body = pymunk.Body(self.link_mass, moment)
                    link_body.position = current_pos
                    link_body.angle = math.radians(90)  # 初始角度
                    
                    # 创建shape
                    shape = pymunk.Circle(link_body, self.link_radius)
                    shape.friction = self.link_friction
                    shape.collision_type = i + 1  # collision_type从1开始
                    
                    self.space.add(link_body, shape)
                    
                    # 🔧 **原始设计**: 基座关节直接连接到static_body
                    joint = pymunk.PinJoint(self.space.static_body, link_body, self.anchor_point, (0, 0))
                    joint.collide_bodies = False
                    self.space.add(joint)
                    self.joints.append(joint)
                    
                    # 🔧 **原始设计**: Motor直接连接到static_body
                    motor = pymunk.SimpleMotor(self.space.static_body, link_body, 0.0)
                    motor.max_force = 30000  # 🔧 所有关节统一max_force
                    self.space.add(motor)
                    self.motors.append(motor)
                    
                    # 🔧 **原始设计**: 角度限制直接连接到static_body
                    if i < len(self.joint_angle_limits) and self.joint_angle_limits[i] is not None:
                        min_angle, max_angle = self.joint_angle_limits[i]
                        limit_joint = pymunk.RotaryLimitJoint(
                            self.space.static_body, link_body, 
                            min_angle, max_angle
                        )
                        limit_joint.max_force = 100000
                        self.space.add(limit_joint)
                        self.joint_limits.append(limit_joint)
                    else:
                        # 基座关节无角度限制
                        self.joint_limits.append(None)
                        
                else:
                    # 其他关节，连接到前一个link
                    prev_body = self.bodies[i-1]
                    
                    # 🔧 手动创建link body
                    moment = pymunk.moment_for_circle(self.link_mass, 0, self.link_radius)
                    link_body = pymunk.Body(self.link_mass, moment)
                    link_body.position = current_pos
                    link_body.angle = math.radians(90)  # 初始角度
                    
                    # 创建shape
                    shape = pymunk.Circle(link_body, self.link_radius)
                    shape.friction = self.link_friction
                    shape.collision_type = i + 1  # collision_type从1开始
                    
                    self.space.add(link_body, shape)
                    
                    # 🔧 连接到前一个link的末端
                    joint = pymunk.PivotJoint(prev_body, link_body, (self.link_lengths[i-1], 0), (0, 0))
                    joint.collide_bodies = False
                    self.space.add(joint)
                    self.joints.append(joint)
                    
                    # 🔧 添加Motor控制器
                    motor = pymunk.SimpleMotor(prev_body, link_body, 0.0)
                    motor.max_force = 30000  # 🔧 所有关节统一max_force
                    self.space.add(motor)
                    self.motors.append(motor)
                    
                    # 🔧 添加角度限制约束
                    if i < len(self.joint_angle_limits) and self.joint_angle_limits[i] is not None:
                        min_angle, max_angle = self.joint_angle_limits[i]
                        limit_joint = pymunk.RotaryLimitJoint(
                            prev_body, link_body, 
                            min_angle, max_angle
                        )
                        limit_joint.max_force = 100000
                        self.space.add(limit_joint)
                        self.joint_limits.append(limit_joint)
                    else:
                        self.joint_limits.append(None)
                
                self.bodies.append(link_body)
                
                # 🔧 更新下一个link的起始位置
                current_pos[0] += self.link_lengths[i] * math.cos(math.radians(90))
                current_pos[1] += self.link_lengths[i] * math.sin(math.radians(90))
            
            print(f"🔧 原始设计创建完成: {len(self.bodies)}个link, {len(self.motors)}个motor")
            print(f"🔧 所有motor max_force: 30000")
    
    return OriginalReacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )

def create_virtual_anchor_env():
    """创建虚拟锚点基座关节环境"""
    
    class VirtualAnchorReacher2DEnv(Reacher2DEnv):
        def _create_robot(self):
            """确保虚拟锚点设计的max_force一致"""
            # 调用父类方法
            super()._create_robot()
            
            # 🔧 统一所有motor的max_force
            for motor in self.motors:
                motor.max_force = 30000  # 🔧 与原始设计保持一致
            
            print(f"🔧 虚拟锚点设计: {len(self.bodies)}个link, {len(self.motors)}个motor")
            print(f"🔧 所有motor max_force: 30000")
    
    return VirtualAnchorReacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )

def test_rotation(env, env_name, test_duration=300):
    """测试基座关节旋转效果"""
    print(f"\n🎮 测试 {env_name} 基座关节旋转")
    print("=" * 50)
    
    env.reset()
    
    # 临时修改torque_to_speed_ratio进行测试
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
        
        # 🔧 使用一致的torque_to_speed_ratio
        torque_to_speed_ratio = 0.05  # 适中的值
        
        for i, torque in enumerate(actions):
            if i < len(self.motors):
                motor = self.motors[i]
                target_angular_velocity = torque * torque_to_speed_ratio
                motor.rate = float(target_angular_velocity)

        self.space.step(self.dt)
        
        if self.explosion_detection and pre_step_velocities:
            self._detect_and_fix_explosion(pre_step_velocities)
        
        observation = self._get_observation()
        reward = self._compute_reward()
        truncated = False
        info = self._build_info_dict()

        if self.gym_api_version == "old":
            done = False
            return observation, reward, done, info
        else:
            return observation, reward, False, truncated, info
    
    env.step = types.MethodType(patched_step, env)
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    
    # 记录初始角度
    initial_angle = env.bodies[0].angle
    max_angle_change = 0
    
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
        
        # 显示实时信息
        info_texts = [
            f"🔧 测试: {env_name}",
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
            f"目标角速度: {actions[0] * 0.05:.1f} rad/s",
            f"目标角速度: {np.degrees(actions[0] * 0.05):.1f}°/s",
            "",
            f"🔍 旋转观察:",
            f"实际方向: {'逆时针 ⟳' if base_angular_vel < -0.1 else '顺时针 ⟲' if base_angular_vel > 0.1 else '静止 ⏸'}",
            "",
            "🔧 设计参数:",
            f"torque_to_speed_ratio: 0.05",
            f"motor.max_force: 30000 (统一)",
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
                if env_name in text:
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
                elif "最大角度变化" in text:
                    color = (255, 150, 255)
                elif "逆时针 ⟳" in text:
                    color = (100, 255, 255)
                elif "设计参数" in text:
                    color = (255, 150, 150)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每50步输出统计
        if step_count % 50 == 0:
            print(f"📊 {env_name} - 步数{step_count}: 角度变化={angle_change_deg:.1f}°, 角速度={np.degrees(base_angular_vel):.1f}°/s")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终结果
    final_angle_change = np.degrees(env.bodies[0].angle - initial_angle)
    
    result = {
        'env_name': env_name,
        'steps': step_count,
        'final_angle_change': final_angle_change,
        'max_angle_change': max_angle_change,
        'final_angular_velocity': np.degrees(env.bodies[0].angular_velocity)
    }
    
    print(f"\n🎯 {env_name} 测试结果:")
    print(f"   测试步数: {result['steps']}")
    print(f"   最终角度变化: {result['final_angle_change']:.1f}°")
    print(f"   最大角度变化: {result['max_angle_change']:.1f}°")
    print(f"   最终角速度: {result['final_angular_velocity']:.1f}°/s")
    
    env.close()
    return result

def compare_base_rotation():
    """对比两种基座关节设计"""
    print("🎯 对比基座关节旋转效果")
    print("=" * 60)
    
    results = []
    
    # 测试原始设计
    print("\n1️⃣ 测试原始基座关节设计 (直接连接static_body)")
    original_env = create_original_env()
    original_result = test_rotation(original_env, "原始设计", 300)
    results.append(original_result)
    
    # 等待一下
    time.sleep(2)
    
    # 测试虚拟锚点设计
    print("\n2️⃣ 测试虚拟锚点基座关节设计")
    virtual_env = create_virtual_anchor_env()
    virtual_result = test_rotation(virtual_env, "虚拟锚点", 300)
    results.append(virtual_result)
    
    # 对比结果
    print("\n" + "=" * 60)
    print("🏆 对比结果总结")
    print("=" * 60)
    
    for result in results:
        print(f"\n📊 {result['env_name']}:")
        print(f"   最终角度变化: {result['final_angle_change']:.1f}°")
        print(f"   最大角度变化: {result['max_angle_change']:.1f}°")
        print(f"   最终角速度: {result['final_angular_velocity']:.1f}°/s")
    
    # 分析差异
    original = results[0]
    virtual = results[1]
    
    angle_diff = abs(original['max_angle_change']) - abs(virtual['max_angle_change'])
    
    print(f"\n🔍 差异分析:")
    print(f"   最大角度变化差异: {angle_diff:.1f}°")
    
    if abs(angle_diff) < 5:
        print("   ✅ 两种设计旋转效果相似")
    elif angle_diff > 5:
        print("   📈 原始设计旋转幅度更大")
    else:
        print("   📉 虚拟锚点设计旋转幅度更大")
    
    print(f"\n💡 结论:")
    if abs(original['max_angle_change']) > 30 and abs(virtual['max_angle_change']) > 30:
        print("   🎉 两种设计都能实现有效的基座关节旋转")
        print("   🎯 虚拟锚点设计在保持旋转能力的同时修复了穿透问题")
    elif abs(original['max_angle_change']) > 30:
        print("   ⚠️ 原始设计旋转效果更好，但存在穿透问题")
        print("   🔧 需要优化虚拟锚点设计的约束参数")
    else:
        print("   ❌ 两种设计的旋转效果都不理想")
        print("   🔧 需要进一步调整motor参数或物理约束")

if __name__ == "__main__":
    compare_base_rotation()

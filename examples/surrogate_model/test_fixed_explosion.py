#!/usr/bin/env python3
"""
测试修复后的防炸开环境
- 验证reacher2d_env.py中的防炸开功能
- 对比修复前后的效果
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

class CollisionDetector:
    """Link间碰撞检测器"""
    
    def __init__(self, env):
        self.env = env
        self.link_collisions = {}  # {(link_i, link_j): count}
        self.obstacle_collisions = 0
        self.total_collisions = 0
        self.collision_history = []  # 记录碰撞历史
        self.penetration_detections = 0
        
        # 设置碰撞检测
        self._setup_link_collision_detection()
        
    def _setup_link_collision_detection(self):
        """设置link间碰撞检测"""
        try:
            # 为所有link对设置碰撞检测（包括相邻和非相邻）
            for i in range(self.env.num_links):
                for j in range(i + 1, self.env.num_links):  # 所有link对
                    link_i_type = i + 1
                    link_j_type = j + 1
                    
                    # 创建碰撞处理器
                    def make_collision_handler(link_i, link_j):
                        def collision_handler(arbiter, space, data):
                            return self._handle_link_collision(arbiter, space, data, link_i, link_j)
                        return collision_handler
                    
                    try:
                        self.env.space.on_collision(
                            collision_type_a=link_i_type,
                            collision_type_b=link_j_type,
                            begin=make_collision_handler(i, j)
                        )
                        print(f"✅ 设置Link{i}-Link{j}碰撞检测")
                    except Exception as e:
                        print(f"⚠️ 设置Link{i}-Link{j}碰撞检测失败: {e}")
                        
        except Exception as e:
            print(f"❌ 碰撞检测设置失败: {e}")
    
    def _handle_link_collision(self, arbiter, space, data, link_i, link_j):
        """处理link间碰撞"""
        # 记录碰撞
        collision_key = (link_i, link_j)
        if collision_key not in self.link_collisions:
            self.link_collisions[collision_key] = 0
        self.link_collisions[collision_key] += 1
        self.total_collisions += 1
        
        # 分析碰撞类型
        is_adjacent = abs(link_i - link_j) == 1
        collision_type = "相邻" if is_adjacent else "非相邻"
        
        # 计算穿透深度
        contact_set = arbiter.contact_point_set
        max_penetration = 0
        if contact_set.count > 0:
            for i in range(contact_set.count):
                contact = contact_set.points[i]
                if contact.distance < 0:  # 负值表示穿透
                    penetration_depth = abs(contact.distance)
                    max_penetration = max(max_penetration, penetration_depth)
        
        # 记录碰撞信息
        collision_info = {
            'step': getattr(self.env, 'step_counter', 0),
            'links': (link_i, link_j),
            'type': collision_type,
            'penetration': max_penetration,
            'is_severe': max_penetration > 5.0  # 严重穿透阈值
        }
        self.collision_history.append(collision_info)
        
        # 严重穿透时输出警告
        if max_penetration > 5.0:
            self.penetration_detections += 1
            print(f"🚨 严重穿透检测: Link{link_i}-Link{j} 深度:{max_penetration:.1f}px")
        
        return True  # 允许物理碰撞处理
    
    def get_collision_stats(self):
        """获取碰撞统计信息"""
        adjacent_collisions = sum(count for (i, j), count in self.link_collisions.items() if abs(i - j) == 1)
        non_adjacent_collisions = sum(count for (i, j), count in self.link_collisions.items() if abs(i - j) > 1)
        
        return {
            'total_collisions': self.total_collisions,
            'adjacent_collisions': adjacent_collisions,
            'non_adjacent_collisions': non_adjacent_collisions,
            'penetration_detections': self.penetration_detections,
            'collision_pairs': len(self.link_collisions),
            'collision_details': dict(self.link_collisions)
        }
    
    def check_current_penetrations(self):
        """检查当前的穿透情况"""
        current_penetrations = []
        
        # 检查所有link对的距离
        for i in range(self.env.num_links):
            for j in range(i + 2, self.env.num_links):  # 跳过相邻link
                body_i = self.env.bodies[i]
                body_j = self.env.bodies[j]
                
                # 计算link中心距离
                pos_i = np.array(body_i.position)
                pos_j = np.array(body_j.position)
                distance = np.linalg.norm(pos_i - pos_j)
                
                # 估算link半径（基于segment长度）
                radius_i = self.env.link_lengths[i] / 2 + 8  # +8是shape半径
                radius_j = self.env.link_lengths[j] / 2 + 8
                min_safe_distance = 16  # 两个shape的最小安全距离
                
                if distance < min_safe_distance:
                    penetration_depth = min_safe_distance - distance
                    current_penetrations.append({
                        'links': (i, j),
                        'distance': distance,
                        'penetration': penetration_depth,
                        'severity': 'severe' if penetration_depth > 10 else 'mild'
                    })
        
        return current_penetrations

def test_fixed_explosion():
    """测试修复后的防炸开环境"""
    print("🛡️ 测试修复后的防炸开环境 + Link碰撞检测")
    print("📋 测试项目:")
    print("  1. 持续折叠测试 - 按A键让机器人折叠")
    print("  2. 突然释放测试 - 释放按键观察是否炸开")
    print("  3. 速度监控 - 实时显示速度状态")
    print("  4. 自动压力测试 - 按Space启用")
    print("  5. Link碰撞检测 - 监控相邻和非相邻Link碰撞")
    print("  6. 穿透深度分析 - 检测严重穿透现象")
    print("  7. 实时碰撞统计 - 显示碰撞频率和类型")
    
    # 创建环境（现在内置防炸开功能）
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'
    )
    
    # 验证防炸开功能已启用
    print(f"✅ 防炸开功能状态:")
    print(f"  explosion_detection: {getattr(env, 'explosion_detection', 'NOT SET')}")
    print(f"  max_safe_velocity: {getattr(env, 'max_safe_velocity', 'NOT SET')}")
    print(f"  max_safe_angular_velocity: {getattr(env, 'max_safe_angular_velocity', 'NOT SET')}")
    print(f"  gentle_separation: {getattr(env, 'gentle_separation', 'NOT SET')}")
    
    print(f"\n🔍 碰撞检测功能状态:")
    print(f"  Link数量: {env.num_links}")
    print(f"  可能的Link对数: {env.num_links * (env.num_links - 1) // 2}")
    print(f"  碰撞检测器: 已初始化")
    
    env.reset()
    
    # 初始化pygame
    pygame.init()
    clock = pygame.time.Clock()
    
    running = True
    step_count = 0
    explosion_count = 0
    max_velocity_recorded = 0
    max_angular_velocity_recorded = 0
    
    # 初始化碰撞检测器
    collision_detector = CollisionDetector(env)
    
    # 统计数据
    stats = {
        'total_steps': 0,
        'explosion_detections': 0,
        'speed_corrections': 0,
        'max_velocity_ever': 0,
        'max_angular_velocity_ever': 0,
        'collision_stats': {}
    }
    
    print("\n🎮 控制说明:")
    print("  WASD: 手动控制前两个关节")
    print("  1234: 控制后面的关节")
    print("  A: 持续按住让机器人折叠")
    print("  Space: 自动压力测试模式")
    print("  Q: 退出")
    print("\n📊 实时监控:")
    print("  - 速度条: 绿色=安全, 红色=超限")
    print("  - 穿透度: 绿色=无穿透, 橙色=轻微穿透, 红色=严重穿透")
    print("  - 碰撞统计: 实时显示Link间碰撞情况")
    
    auto_test = False
    fold_phase = True
    fold_counter = 0
    
    while running and step_count < 5000:
        # 处理事件
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_test = not auto_test
                    print(f"🔄 {'启用' if auto_test else '禁用'}自动压力测试模式")
        
        # 生成动作
        if auto_test:
            # 自动压力测试：极端折叠动作
            if fold_phase:
                # 极端折叠阶段：使用很大的力
                actions = np.array([80, -80, 80, -80])  # 更大的力测试
                fold_counter += 1
                if fold_counter > 150:  # 折叠150步
                    fold_phase = False
                    fold_counter = 0
                    print("🔄 切换到突然释放阶段")
            else:
                # 突然释放阶段：完全停止
                actions = np.array([0, 0, 0, 0])
                fold_counter += 1
                if fold_counter > 100:  # 释放100步
                    fold_phase = True
                    fold_counter = 0
                    print("🔄 切换到极端折叠阶段")
        else:
            # 手动控制
            actions = np.zeros(4)
            if keys[pygame.K_a]:
                actions[1] = 70  # 大力折叠
            if keys[pygame.K_d]:
                actions[1] = -70
            if keys[pygame.K_w]:
                actions[0] = 70
            if keys[pygame.K_s]:
                actions[0] = -70
            if keys[pygame.K_1]:
                actions[2] = 50
            if keys[pygame.K_2]:
                actions[2] = -50
            if keys[pygame.K_3]:
                actions[3] = 50
            if keys[pygame.K_4]:
                actions[3] = -50
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 收集速度统计
        velocities = [np.linalg.norm(body.velocity) for body in env.bodies]
        angular_velocities = [abs(body.angular_velocity) for body in env.bodies]
        
        max_velocity = max(velocities) if velocities else 0
        max_angular_velocity = max(angular_velocities) if angular_velocities else 0
        
        # 检测当前穿透情况
        current_penetrations = collision_detector.check_current_penetrations()
        severe_penetrations = [p for p in current_penetrations if p['severity'] == 'severe']
        
        # 更新记录
        stats['max_velocity_ever'] = max(stats['max_velocity_ever'], max_velocity)
        stats['max_angular_velocity_ever'] = max(stats['max_angular_velocity_ever'], max_angular_velocity)
        
        # 检测是否触发了防炸开系统
        if max_velocity > env.max_safe_velocity or max_angular_velocity > env.max_safe_angular_velocity:
            stats['speed_corrections'] += 1
        
        # 检测潜在的炸开（如果没有防炸开系统会发生的情况）
        if max_velocity > 300 or max_angular_velocity > 15:
            explosion_count += 1
            stats['explosion_detections'] += 1
            print(f"⚠️ 检测到潜在炸开情况！步数: {step_count}, 最大速度: {max_velocity:.1f}")
        
        # 渲染
        env.render()
        
        # 获取碰撞统计
        collision_stats = collision_detector.get_collision_stats()
        
        # 在屏幕上显示统计信息
        font = pygame.font.Font(None, 28)
        info_texts = [
            f"步数: {step_count}",
            f"潜在炸开: {explosion_count}",
            f"速度修正: {stats['speed_corrections']}",
            f"最大速度: {max_velocity:.1f}/{env.max_safe_velocity}",
            f"最大角速度: {max_angular_velocity:.1f}/{env.max_safe_angular_velocity}",
            f"模式: {'自动压力测试' if auto_test else '手动控制'}",
            f"阶段: {'极端折叠' if fold_phase else '突然释放'}" if auto_test else "",
            "🛡️ 防炸开系统: 已启用",
            "",  # 分隔线
            "📊 碰撞检测:",
            f"总碰撞: {collision_stats['total_collisions']}",
            f"相邻Link: {collision_stats['adjacent_collisions']}",
            f"非相邻Link: {collision_stats['non_adjacent_collisions']}",
            f"严重穿透: {collision_stats['penetration_detections']}",
            f"当前穿透: {len(current_penetrations)}",
            f"当前严重穿透: {len(severe_penetrations)}"
        ]
        
        # 创建半透明背景
        info_surface = pygame.Surface((380, len(info_texts) * 30 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        # 显示信息
        for i, text in enumerate(info_texts):
            if text:  # 跳过空字符串
                color = (255, 255, 255)
                if "炸开" in text and explosion_count > 0:
                    color = (255, 100, 100)
                elif "修正" in text and stats['speed_corrections'] > 0:
                    color = (100, 255, 100)
                elif "防炸开系统" in text:
                    color = (100, 255, 100)
                elif "碰撞检测" in text:
                    color = (100, 200, 255)
                elif "严重穿透" in text and collision_stats['penetration_detections'] > 0:
                    color = (255, 150, 100)
                elif "当前穿透" in text and len(current_penetrations) > 0:
                    color = (255, 200, 100)
                elif "当前严重穿透" in text and len(severe_penetrations) > 0:
                    color = (255, 100, 100)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 30))
        
        # 速度条和碰撞指示器显示
        bar_width = 200
        bar_height = 8
        bar_x = 20
        
        # 线速度条
        vel_ratio = min(max_velocity / (env.max_safe_velocity * 1.5), 1.0)
        vel_color = (255, 0, 0) if max_velocity > env.max_safe_velocity else (0, 255, 0)
        pygame.draw.rect(env.screen, (100, 100, 100), (bar_x, 520, bar_width, bar_height))
        pygame.draw.rect(env.screen, vel_color, (bar_x, 520, int(bar_width * vel_ratio), bar_height))
        
        # 角速度条
        ang_ratio = min(max_angular_velocity / (env.max_safe_angular_velocity * 1.5), 1.0)
        ang_color = (255, 0, 0) if max_angular_velocity > env.max_safe_angular_velocity else (0, 255, 0)
        pygame.draw.rect(env.screen, (100, 100, 100), (bar_x, 535, bar_width, bar_height))
        pygame.draw.rect(env.screen, ang_color, (bar_x, 535, int(bar_width * ang_ratio), bar_height))
        
        # 碰撞指示器
        collision_ratio = min(len(current_penetrations) / 10.0, 1.0)  # 最多10个穿透
        collision_color = (255, 0, 0) if len(severe_penetrations) > 0 else ((255, 150, 0) if len(current_penetrations) > 0 else (0, 255, 0))
        pygame.draw.rect(env.screen, (100, 100, 100), (bar_x, 550, bar_width, bar_height))
        pygame.draw.rect(env.screen, collision_color, (bar_x, 550, int(bar_width * collision_ratio), bar_height))
        
        # 添加标签
        label_font = pygame.font.Font(None, 20)
        vel_label = label_font.render("线速度", True, (255, 255, 255))
        ang_label = label_font.render("角速度", True, (255, 255, 255))
        col_label = label_font.render("穿透度", True, (255, 255, 255))
        env.screen.blit(vel_label, (bar_x + bar_width + 10, 515))
        env.screen.blit(ang_label, (bar_x + bar_width + 10, 530))
        env.screen.blit(col_label, (bar_x + bar_width + 10, 545))
        
        pygame.display.flip()
        
        step_count += 1
        stats['total_steps'] = step_count
        stats['collision_stats'] = collision_stats
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终统计
    final_collision_stats = collision_detector.get_collision_stats()
    
    print(f"\n📊 测试结果总结:")
    print(f"  总步数: {stats['total_steps']}")
    print(f"  潜在炸开次数: {stats['explosion_detections']}")
    print(f"  速度修正次数: {stats['speed_corrections']}")
    print(f"  历史最大速度: {stats['max_velocity_ever']:.1f}")
    print(f"  历史最大角速度: {stats['max_angular_velocity_ever']:.1f}")
    print(f"  炸开率: {stats['explosion_detections']/stats['total_steps']*100:.2f}%")
    print(f"  修正率: {stats['speed_corrections']/stats['total_steps']*100:.2f}%")
    
    print(f"\n🔍 碰撞检测结果:")
    print(f"  总碰撞次数: {final_collision_stats['total_collisions']}")
    print(f"  相邻Link碰撞: {final_collision_stats['adjacent_collisions']}")
    print(f"  非相邻Link碰撞: {final_collision_stats['non_adjacent_collisions']}")
    print(f"  严重穿透次数: {final_collision_stats['penetration_detections']}")
    print(f"  涉及的Link对: {final_collision_stats['collision_pairs']}")
    
    if final_collision_stats['collision_details']:
        print(f"\n📋 详细碰撞统计:")
        for (i, j), count in final_collision_stats['collision_details'].items():
            collision_type = "相邻" if abs(i - j) == 1 else "非相邻"
            print(f"    Link{i}-Link{j} ({collision_type}): {count}次")
    
    # 综合评估
    if stats['explosion_detections'] == 0:
        print("🎉 完美！没有检测到任何炸开现象！")
    elif stats['speed_corrections'] > stats['explosion_detections']:
        print("✅ 防炸开系统工作良好，成功阻止了大部分炸开！")
    else:
        print("⚠️ 仍有一些炸开现象，可能需要进一步调优参数。")
    
    # 碰撞评估
    if final_collision_stats['total_collisions'] == 0:
        print("🎉 完美！没有检测到任何Link间碰撞！")
    elif final_collision_stats['penetration_detections'] == 0:
        print("✅ 碰撞检测正常，没有严重穿透现象！")
    elif final_collision_stats['penetration_detections'] < 10:
        print("⚠️ 检测到少量严重穿透，整体表现良好。")
    else:
        print("❌ 检测到较多严重穿透，需要优化物理参数。")
    
    env.close()

if __name__ == "__main__":
    test_fixed_explosion()

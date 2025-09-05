#!/usr/bin/env python3
"""
修复版碰撞检测测试脚本
- 直接修改环境的碰撞处理器来添加统计功能
"""

import sys
import os
import numpy as np
import pygame
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

class CollisionStats:
    """碰撞统计类"""
    def __init__(self):
        self.total_collisions = 0
        self.link_collisions = {}  # {(link_i, link_j): count}
        self.obstacle_collisions = 0
        self.severe_penetrations = 0
        self.collision_log = []
        
    def log_collision(self, collision_type, link_i=None, link_j=None, penetration=0):
        """记录碰撞"""
        self.total_collisions += 1
        
        if collision_type == "link":
            key = tuple(sorted([link_i, link_j]))
            if key not in self.link_collisions:
                self.link_collisions[key] = 0
            self.link_collisions[key] += 1
            
            if penetration > 5.0:
                self.severe_penetrations += 1
                print(f"🚨 严重穿透: Link{link_i}-Link{link_j} 深度:{penetration:.1f}px")
                
        elif collision_type == "obstacle":
            self.obstacle_collisions += 1
            
        self.collision_log.append({
            'type': collision_type,
            'links': (link_i, link_j) if collision_type == "link" else None,
            'penetration': penetration,
            'timestamp': time.time()
        })
    
    def get_stats(self):
        """获取统计信息"""
        adjacent_collisions = sum(count for (i, j), count in self.link_collisions.items() if abs(i - j) == 1)
        non_adjacent_collisions = sum(count for (i, j), count in self.link_collisions.items() if abs(i - j) > 1)
        
        return {
            'total_collisions': self.total_collisions,
            'link_collisions': len(self.link_collisions),
            'adjacent_collisions': adjacent_collisions,
            'non_adjacent_collisions': non_adjacent_collisions,
            'obstacle_collisions': self.obstacle_collisions,
            'severe_penetrations': self.severe_penetrations,
            'collision_details': dict(self.link_collisions)
        }

def patch_collision_handlers(env, collision_stats):
    """给环境的碰撞处理器添加统计功能"""
    
    # 保存原始的碰撞处理器
    original_handlers = {}
    
    # 为Link间碰撞添加统计
    for i in range(env.num_links):
        for j in range(i + 2, env.num_links):  # 只处理非相邻Link
            link_i_type = i + 1
            link_j_type = j + 1
            
            # 创建增强的碰撞处理器
            def make_enhanced_handler(orig_i, orig_j):
                def enhanced_collision_handler(arbiter, space, data):
                    # 计算穿透深度
                    penetration_depth = 0
                    contact_set = arbiter.contact_point_set
                    if contact_set.count > 0:
                        for k in range(contact_set.count):
                            contact = contact_set.points[k]
                            if contact.distance < 0:
                                penetration_depth = max(penetration_depth, abs(contact.distance))
                    
                    # 记录碰撞
                    collision_stats.log_collision("link", orig_i, orig_j, penetration_depth)
                    
                    # 调用原始的温和碰撞处理
                    if penetration_depth > 0:
                        # 温和分离
                        gentle_impulse = min(penetration_depth * 0.1, env.max_separation_impulse)
                        separation_impulse = arbiter.contact_point_set.normal * gentle_impulse
                        
                        # 应用温和分离力
                        for body in [arbiter.shapes[0].body, arbiter.shapes[1].body]:
                            if body != space.static_body:
                                body.velocity = body.velocity + separation_impulse / body.mass
                    
                    return True  # 允许物理处理继续
                
                return enhanced_collision_handler
            
            try:
                # 使用on_collision设置增强的处理器
                env.space.on_collision(
                    collision_type_a=link_i_type,
                    collision_type_b=link_j_type,
                    begin=make_enhanced_handler(i, j)
                )
                print(f"✅ 增强Link{i}-Link{j}碰撞检测")
            except Exception as e:
                print(f"❌ 设置Link{i}-Link{j}碰撞检测失败: {e}")
    
    # 为障碍物碰撞添加统计
    OBSTACLE_COLLISION_TYPE = 100
    for i in range(env.num_links):
        robot_link_type = i + 1
        
        def make_obstacle_handler(link_idx):
            def obstacle_collision_handler(arbiter, space, data):
                collision_stats.log_collision("obstacle")
                print(f"🚨 Link{link_idx}撞击障碍物!")
                return True
            return obstacle_collision_handler
        
        try:
            env.space.on_collision(
                collision_type_a=robot_link_type,
                collision_type_b=OBSTACLE_COLLISION_TYPE,
                begin=make_obstacle_handler(i)
            )
            print(f"✅ 增强Link{i}-障碍物碰撞检测")
        except Exception as e:
            print(f"❌ 设置Link{i}-障碍物碰撞检测失败: {e}")

def test_collision_detection_working():
    """测试工作版本的碰撞检测"""
    print("🛡️ 测试工作版本的碰撞检测系统")
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'  # 减少调试输出
    )
    
    env.reset()
    
    # 创建碰撞统计器
    collision_stats = CollisionStats()
    
    # 给环境添加碰撞统计功能
    print("\n🔧 添加碰撞统计功能...")
    patch_collision_handlers(env, collision_stats)
    
    print(f"\n🎮 开始测试 (WASD控制, Space自动测试, Q退出)...")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    
    running = True
    step_count = 0
    auto_test = False
    fold_phase = True
    fold_counter = 0
    
    while running and step_count < 3000:
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
                    print(f"🔄 {'启用' if auto_test else '禁用'}自动压力测试")
        
        # 生成动作
        if auto_test:
            # 自动压力测试：极端折叠
            if fold_phase:
                actions = np.array([80, -80, 80, -80])  # 极端折叠
                fold_counter += 1
                if fold_counter > 100:  # 折叠100步
                    fold_phase = False
                    fold_counter = 0
            else:
                actions = np.array([0, 0, 0, 0])  # 突然停止
                fold_counter += 1
                if fold_counter > 50:  # 停止50步
                    fold_phase = True
                    fold_counter = 0
        else:
            # 手动控制
            actions = np.zeros(4)
            if keys[pygame.K_a]:
                actions[1] = 60  # 折叠
            if keys[pygame.K_d]:
                actions[1] = -60
            if keys[pygame.K_w]:
                actions[0] = 60
            if keys[pygame.K_s]:
                actions[0] = -60
            if keys[pygame.K_1]:
                actions[2] = 40
            if keys[pygame.K_2]:
                actions[2] = -40
            if keys[pygame.K_3]:
                actions[3] = 40
            if keys[pygame.K_4]:
                actions[3] = -40
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 获取统计信息
        stats = collision_stats.get_stats()
        
        # 渲染
        env.render()
        
        # 显示碰撞统计
        info_texts = [
            f"步数: {step_count}",
            f"模式: {'自动压力测试' if auto_test else '手动控制'}",
            f"阶段: {'极端折叠' if fold_phase else '突然停止'}" if auto_test else "",
            "",
            "📊 碰撞统计:",
            f"总碰撞: {stats['total_collisions']}",
            f"Link碰撞对: {stats['link_collisions']}",
            f"相邻Link: {stats['adjacent_collisions']}",
            f"非相邻Link: {stats['non_adjacent_collisions']}",
            f"障碍物碰撞: {stats['obstacle_collisions']}",
            f"严重穿透: {stats['severe_penetrations']}",
            "",
            "🎮 控制:",
            "WASD: 前两关节",
            "1234: 后两关节",
            "Space: 自动测试",
            "Q: 退出"
        ]
        
        # 创建信息背景
        info_surface = pygame.Surface((300, len(info_texts) * 25 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        # 显示信息
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "碰撞统计" in text:
                    color = (100, 200, 255)
                elif "严重穿透" in text and stats['severe_penetrations'] > 0:
                    color = (255, 100, 100)
                elif "总碰撞" in text and stats['total_collisions'] > 0:
                    color = (100, 255, 100)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 25))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每500步输出详细统计
        if step_count % 500 == 0:
            print(f"\n📊 步数{step_count}统计:")
            print(f"   总碰撞: {stats['total_collisions']}")
            print(f"   严重穿透: {stats['severe_penetrations']}")
            if stats['collision_details']:
                print(f"   详细碰撞:")
                for (i, j), count in stats['collision_details'].items():
                    collision_type = "相邻" if abs(i - j) == 1 else "非相邻"
                    print(f"     Link{i}-Link{j} ({collision_type}): {count}次")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终统计
    final_stats = collision_stats.get_stats()
    print(f"\n🎉 最终测试结果:")
    print(f"   总步数: {step_count}")
    print(f"   总碰撞: {final_stats['total_collisions']}")
    print(f"   碰撞率: {final_stats['total_collisions']/step_count*100:.2f}%")
    print(f"   Link间碰撞对: {final_stats['link_collisions']}")
    print(f"   相邻Link碰撞: {final_stats['adjacent_collisions']}")
    print(f"   非相邻Link碰撞: {final_stats['non_adjacent_collisions']}")
    print(f"   障碍物碰撞: {final_stats['obstacle_collisions']}")
    print(f"   严重穿透: {final_stats['severe_penetrations']}")
    
    if final_stats['total_collisions'] > 0:
        print("🎉 碰撞检测系统工作正常！")
        print("📋 详细碰撞统计:")
        for (i, j), count in final_stats['collision_details'].items():
            collision_type = "相邻" if abs(i - j) == 1 else "非相邻"
            print(f"     Link{i}-Link{j} ({collision_type}): {count}次")
    else:
        print("⚠️ 没有检测到碰撞，可能需要更激进的测试动作")
    
    env.close()

if __name__ == "__main__":
    test_collision_detection_working()


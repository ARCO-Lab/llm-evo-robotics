#!/usr/bin/env python3
"""
修复版穿模调试脚本
- 修复机器人初始化问题
- 减少调试输出
- 改进穿模检测逻辑
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

class PenetrationDebugger:
    """穿模问题调试器 - 修复版"""
    
    def __init__(self):
        self.env = None
        self.current_test = 0
        self.test_configs = self._create_test_configs()
        self.penetration_count = 0
        self.collision_log = []
        
        # 控制参数
        self.manual_control = True
        self.auto_test_mode = False
        self.show_debug_info = True
        
    def _create_test_configs(self):
        """创建不同的测试配置"""
        return [
            {
                "name": "默认配置",
                "space_collision_slop": 0.5,
                "space_collision_bias": (1-0.1) ** 60,
                "shape_collision_slop": 0.5,
                "density": 0.8,
                "mass_multiplier": 5,
                "max_force": 30000,
                "joint_collide_bodies": False
            },
            {
                "name": "高碰撞容差",
                "space_collision_slop": 1.0,
                "space_collision_bias": (1-0.05) ** 60,
                "shape_collision_slop": 1.0,
                "density": 0.5,
                "mass_multiplier": 3,
                "max_force": 20000,
                "joint_collide_bodies": False
            },
            {
                "name": "低碰撞容差",
                "space_collision_slop": 0.1,
                "space_collision_bias": (1-0.2) ** 60,
                "shape_collision_slop": 0.1,
                "density": 1.0,
                "mass_multiplier": 8,
                "max_force": 50000,
                "joint_collide_bodies": False
            },
            {
                "name": "启用相邻Link碰撞",
                "space_collision_slop": 0.5,
                "space_collision_bias": (1-0.1) ** 60,
                "shape_collision_slop": 0.5,
                "density": 0.8,
                "mass_multiplier": 5,
                "max_force": 30000,
                "joint_collide_bodies": True  # 关键差异
            }
        ]
    
    def create_test_env(self, config):
        """根据配置创建测试环境"""
        print(f"\n🔧 创建测试环境: {config['name']}")
        
        # 创建环境 - 禁用调试输出
        env = Reacher2DEnv(
            num_links=4,
            link_lengths=[80, 60, 50, 40],
            render_mode="human",
            config_path="configs/reacher_with_zigzag_obstacles.yaml",
            debug_level='WARNING'  # 减少调试输出
        )
        
        # 应用测试配置
        self._apply_config_to_env(env, config)
        
        return env
    
    def _apply_config_to_env(self, env, config):
        """将配置应用到环境"""
        # 修改space参数
        env.space.collision_slop = config["space_collision_slop"]
        env.space.collision_bias = config["space_collision_bias"]
        
        # 修改所有shape的参数
        for body in env.bodies:
            for shape in body.shapes:
                shape.collision_slop = config["shape_collision_slop"]
        
        # 修改关节参数
        for joint in env.joints:
            if hasattr(joint, 'collide_bodies'):
                joint.collide_bodies = config["joint_collide_bodies"]
        
        # 修改约束力
        for limit_joint in env.joint_limits:
            if limit_joint is not None:
                limit_joint.max_force = config["max_force"]
        
        print(f"  ✅ Space collision_slop: {config['space_collision_slop']}")
        print(f"  ✅ Joint collide_bodies: {config['joint_collide_bodies']}")
        print(f"  ✅ Max constraint force: {config['max_force']}")
    
    def detect_penetrations(self):
        """检测穿模现象 - 改进版"""
        penetrations = []
        
        if not self.env or not hasattr(self.env, 'bodies'):
            return penetrations
        
        # 1. 检测Link间穿透 - 更严格的判断
        for i in range(len(self.env.bodies)):
            for j in range(i + 2, len(self.env.bodies)):  # 跳过相邻Link
                pos_i = np.array(self.env.bodies[i].position)
                pos_j = np.array(self.env.bodies[j].position)
                distance = np.linalg.norm(pos_i - pos_j)
                
                # Link半径约为8，如果距离小于12则可能有问题（更严格）
                if distance < 12:
                    penetrations.append({
                        'type': 'link_link',
                        'link1': i,
                        'link2': j,
                        'distance': distance,
                        'severity': 'high' if distance < 6 else 'medium'
                    })
        
        # 2. 检测Link与障碍物穿透
        for i, body in enumerate(self.env.bodies):
            body_pos = np.array(body.position)
            
            for obstacle in self.env.obstacles:
                if hasattr(obstacle, 'a') and hasattr(obstacle, 'b'):
                    # 计算点到线段的距离
                    seg_start = np.array(obstacle.a)
                    seg_end = np.array(obstacle.b)
                    distance = self._point_to_segment_distance(body_pos, seg_start, seg_end)
                    
                    # 障碍物半径约为5，Link半径约为8，如果距离小于10则可能穿透
                    if distance < 10:
                        penetrations.append({
                            'type': 'link_obstacle',
                            'link': i,
                            'distance': distance,
                            'severity': 'high' if distance < 3 else 'medium'
                        })
        
        return penetrations
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """计算点到线段的距离"""
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        
        seg_length_sq = np.dot(seg_vec, seg_vec)
        if seg_length_sq == 0:
            return np.linalg.norm(point_vec)
        
        t = np.dot(point_vec, seg_vec) / seg_length_sq
        t = max(0, min(1, t))
        
        closest_point = seg_start + t * seg_vec
        return np.linalg.norm(point - closest_point)
    
    def render_debug_info(self):
        """渲染调试信息 - 简化版"""
        if not self.show_debug_info or not self.env.screen:
            return
        
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)
        
        # 当前配置信息
        config = self.test_configs[self.current_test]
        y_offset = 10
        
        # 标题
        title_text = font.render(f"🔧 穿模调试器 - {config['name']}", True, (255, 255, 255))
        # 添加背景
        bg_rect = pygame.Rect(5, 5, 400, 200)
        pygame.draw.rect(self.env.screen, (0, 0, 0, 180), bg_rect)
        self.env.screen.blit(title_text, (10, y_offset))
        y_offset += 30
        
        # 控制说明
        controls = [
            "WASD: 手动控制机器人",
            "Space: 切换自动/手动模式", 
            "N: 下一个测试配置",
            "R: 重置环境",
            "Q: 退出"
        ]
        
        for control in controls:
            text = small_font.render(control, True, (200, 200, 200))
            self.env.screen.blit(text, (10, y_offset))
            y_offset += 18
        
        y_offset += 10
        
        # 检测穿模
        penetrations = self.detect_penetrations()
        
        # 穿模统计
        status_color = (255, 100, 100) if penetrations else (100, 255, 100)
        status_text = f"穿模检测: {len(penetrations)} 个问题"
        status_surface = font.render(status_text, True, status_color)
        self.env.screen.blit(status_surface, (10, y_offset))
        y_offset += 25
        
        # 详细穿模信息 - 只显示严重的
        high_severity = [p for p in penetrations if p['severity'] == 'high']
        for pen in high_severity[:3]:  # 最多显示3个严重的
            if pen['type'] == 'link_link':
                detail = f"🔴 Link{pen['link1']}-Link{pen['link2']}: {pen['distance']:.1f}px"
            else:
                detail = f"🔴 Link{pen['link']}-障碍物: {pen['distance']:.1f}px"
            
            detail_surface = small_font.render(detail, True, (255, 150, 150))
            self.env.screen.blit(detail_surface, (20, y_offset))
            y_offset += 18
    
    def handle_manual_control(self, keys):
        """处理手动控制"""
        if not self.manual_control:
            return np.zeros(self.env.num_links)
        
        actions = np.zeros(self.env.num_links)
        
        # WASD控制前两个关节
        if keys[pygame.K_w]:
            actions[0] = 50  # 第一个关节逆时针
        if keys[pygame.K_s]:
            actions[0] = -50  # 第一个关节顺时针
        if keys[pygame.K_a]:
            actions[1] = 50   # 第二个关节逆时针
        if keys[pygame.K_d]:
            actions[1] = -50  # 第二个关节顺时针
        
        # 数字键控制其他关节
        if keys[pygame.K_1]:
            actions[2] = 30 if len(actions) > 2 else 0
        if keys[pygame.K_2]:
            actions[2] = -30 if len(actions) > 2 else 0
        if keys[pygame.K_3]:
            actions[3] = 30 if len(actions) > 3 else 0
        if keys[pygame.K_4]:
            actions[3] = -30 if len(actions) > 3 else 0
        
        return actions
    
    def run_test(self, test_index=0):
        """运行指定的测试"""
        self.current_test = test_index % len(self.test_configs)
        config = self.test_configs[self.current_test]
        
        print(f"\n🚀 开始测试: {config['name']}")
        
        # 创建环境
        if self.env:
            self.env.close()
        
        self.env = self.create_test_env(config)
        obs = self.env.reset()
        
        print(f"🤖 机器人初始化完成")
        print(f"   末端执行器位置: {self.env._get_end_effector_position()}")
        print(f"   目标位置: {self.env.goal_pos}")
        
        # 初始化pygame
        pygame.init()
        clock = pygame.time.Clock()
        
        running = True
        step_count = 0
        
        print("\n🎮 控制说明:")
        print("  WASD: 控制前两个关节")
        print("  1234: 控制后面的关节")
        print("  Space: 切换自动/手动模式")
        print("  N: 下一个测试配置")
        print("  R: 重置环境")
        print("  Q: 退出")
        
        while running:
            # 处理事件
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_n:
                        # 下一个测试配置
                        self.run_test(self.current_test + 1)
                        return
                    elif event.key == pygame.K_r:
                        # 重置环境
                        obs = self.env.reset()
                        step_count = 0
                        print("🔄 环境已重置")
                    elif event.key == pygame.K_SPACE:
                        # 切换模式
                        self.manual_control = not self.manual_control
                        mode = "手动" if self.manual_control else "自动"
                        print(f"🔄 切换到{mode}控制模式")
            
            # 生成动作
            if self.manual_control:
                actions = self.handle_manual_control(keys)
            else:
                # 自动测试模式 - 生成随机动作
                actions = np.random.uniform(-30, 30, self.env.num_links)
            
            # 执行动作
            obs, reward, done, info = self.env.step(actions)
            
            # 检测穿模
            penetrations = self.detect_penetrations()
            if penetrations:
                self.penetration_count += len(penetrations)
                
                # 记录严重穿模
                high_severity = [p for p in penetrations if p['severity'] == 'high']
                if high_severity:
                    self.collision_log.append({
                        'step': step_count,
                        'config': config['name'],
                        'penetrations': high_severity
                    })
            
            # 渲染
            self.env.render()
            self.render_debug_info()
            
            step_count += 1
            
            # 每500步打印统计
            if step_count % 500 == 0:
                print(f"📊 步数: {step_count}, 穿模次数: {self.penetration_count}, 当前穿模: {len(penetrations)}")
            
            # 重置检查
            if done:
                obs = self.env.reset()
                step_count = 0
            
            clock.tick(60)
        
        if self.env:
            self.env.close()

def main():
    """主函数"""
    debugger = PenetrationDebugger()
    
    print("🔧 机器人穿模调试器 - 修复版")
    print("=" * 50)
    
    try:
        # 默认运行第一个测试
        debugger.run_test(0)
    except KeyboardInterrupt:
        print("\n👋 调试器已退出")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if debugger.env:
            debugger.env.close()

if __name__ == "__main__":
    main()

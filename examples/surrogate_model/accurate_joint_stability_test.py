#!/usr/bin/env python3
"""
准确的Joint稳定性测试 - 检测所有类型的Joint问题
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

class AccurateJointStabilityTest:
    def __init__(self):
        self.env = Reacher2DEnv(
            num_links=4,
            link_lengths=[80, 60, 50, 40],
            render_mode="human",
            config_path="configs/reacher_with_zigzag_obstacles.yaml",
            debug_level='WARNING'
        )
        
        pygame.init()
        self.screen = pygame.display.set_mode((900, 700))
        pygame.display.set_caption("准确Joint稳定性测试")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # 统计数据
        self.distance_issues = 0
        self.angle_issues = 0
        self.explosion_detections = 0
        self.collision_count = 0
        
        print("🔧 准确Joint稳定性测试初始化完成")
        
    def check_comprehensive_joint_stability(self):
        """全面检查Joint稳定性"""
        issues = {
            'distance_problems': 0,
            'angle_problems': 0,
            'explosion_detected': False,
            'collision_occurred': False
        }
        
        # 1. 检查Link间距离 - 更严格的检测
        for i in range(1, len(self.env.bodies)):
            body_a = self.env.bodies[i-1]
            body_b = self.env.bodies[i]
            expected_distance = self.env.link_lengths[i-1]
            
            actual_distance = math.sqrt(
                (body_a.position.x - body_b.position.x)**2 + 
                (body_a.position.y - body_b.position.y)**2
            )
            
            # 🔍 更严格的检测：距离偏差超过20%就认为有问题
            distance_error = abs(actual_distance - expected_distance)
            if distance_error > expected_distance * 0.2:
                issues['distance_problems'] += 1
                print(f"⚠️ Joint{i}距离异常: 实际{actual_distance:.1f}, 预期{expected_distance:.1f}, 偏差{distance_error:.1f}")
            
            # 🚨 严重散架检测：距离偏差超过100%
            if distance_error > expected_distance * 1.0:
                issues['explosion_detected'] = True
                print(f"🚨 Joint{i}严重散架! 距离偏差{distance_error:.1f} > {expected_distance:.1f}")
        
        # 2. 检查关节角度（如果有角度限制）
        for i in range(1, len(self.env.bodies)):
            body_a = self.env.bodies[i-1]
            body_b = self.env.bodies[i]
            
            # 计算相对角度
            relative_angle = body_b.angle - body_a.angle
            # 标准化到[-π, π]
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi
            
            # 检查是否超出合理范围（±150°）
            max_reasonable_angle = math.pi * 5/6  # 150°
            if abs(relative_angle) > max_reasonable_angle:
                issues['angle_problems'] += 1
        
        # 3. 检查是否有异常高速运动（爆炸迹象）
        for i, body in enumerate(self.env.bodies):
            velocity_magnitude = body.velocity.length
            angular_velocity_magnitude = abs(body.angular_velocity)
            
            # 🚨 线速度检测 - 降低阈值更敏感
            if velocity_magnitude > 500:  # 速度超过500像素/秒
                issues['explosion_detected'] = True
                print(f"🚨 Link{i}线速度异常: {velocity_magnitude:.1f} px/s")
            
            # 🚨 角速度检测 - 检测疯狂旋转
            if angular_velocity_magnitude > 10:  # 角速度超过10 rad/s
                issues['explosion_detected'] = True
                print(f"🚨 Link{i}角速度异常: {angular_velocity_magnitude:.1f} rad/s")
        
        # 4. 检查碰撞（简单检测）
        # 这里可以通过检查Link是否与障碍物重叠来判断
        # 简化处理：如果任何Link的位置接近障碍物区域
        for body in self.env.bodies:
            x, y = body.position.x, body.position.y
            # 检查是否在障碍物区域内（根据YAML配置）
            if (487 <= x <= 537 and 100 <= y <= 400) or (612 <= x <= 662 and 400 <= y <= 700):
                issues['collision_occurred'] = True
                break
        
        return issues
    
    def draw_comprehensive_status(self, surface, issues, step_count):
        """绘制全面的状态信息"""
        # 背景
        overlay = pygame.Surface((300, 400))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (580, 50))
        
        # 标题
        title = self.font.render("Joint稳定性全面检测", True, (255, 255, 255))
        surface.blit(title, (590, 60))
        
        # 当前状态
        y_offset = 90
        status_items = [
            f"步骤: {step_count}",
            f"距离问题: {issues['distance_problems']}",
            f"角度问题: {issues['angle_problems']}",
            f"爆炸检测: {'是' if issues['explosion_detected'] else '否'}",
            f"碰撞检测: {'是' if issues['collision_occurred'] else '否'}",
            "",
            "累计统计:",
            f"距离问题总数: {self.distance_issues}",
            f"角度问题总数: {self.angle_issues}",
            f"爆炸检测次数: {self.explosion_detections}",
            f"碰撞次数: {self.collision_count}",
        ]
        
        for item in status_items:
            if item == "":
                y_offset += 10
                continue
                
            color = (255, 255, 255)
            if "问题:" in item and not item.endswith("0"):
                color = (255, 100, 100)
            elif "爆炸检测: 是" in item or "碰撞检测: 是" in item:
                color = (255, 0, 0)
                
            text = self.font.render(item, True, color)
            surface.blit(text, (590, y_offset))
            y_offset += 25
        
        # 总体评估
        y_offset += 20
        total_issues = (self.distance_issues + self.angle_issues + 
                       self.explosion_detections + self.collision_count)
        
        if total_issues == 0:
            status_text = "✅ Joint完全稳定"
            status_color = (0, 255, 0)
        elif total_issues < 10:
            status_text = "⚠️ Joint轻微不稳定"
            status_color = (255, 255, 0)
        else:
            status_text = "❌ Joint严重不稳定"
            status_color = (255, 0, 0)
        
        status_surface = self.font.render(status_text, True, status_color)
        surface.blit(status_surface, (590, y_offset))
        
        # 控制说明
        y_offset += 50
        controls = [
            "控制说明:",
            "WASD: 基座关节",
            "QE: 关节1",
            "ZC: 关节2", 
            "RF: 关节3",
            "SPACE: 激进模式",
            "ESC: 退出"
        ]
        
        for control in controls:
            color = (200, 200, 200) if control != "控制说明:" else (255, 255, 255)
            text = self.font.render(control, True, color)
            surface.blit(text, (590, y_offset))
            y_offset += 20
    
    def run_test(self):
        """运行准确的稳定性测试"""
        print("🚀 开始准确Joint稳定性测试")
        print("   这个测试会检测:")
        print("   1. Link间距离偏差")
        print("   2. 关节角度异常")
        print("   3. 爆炸现象检测")
        print("   4. 碰撞检测")
        print("\n🎮 控制说明:")
        print("   WASD: 控制基座关节")
        print("   QE: 控制关节1")
        print("   ZC: 控制关节2")
        print("   RF: 控制关节3")
        print("   SPACE: 激进模式 (动作幅度翻倍)")
        print("   ESC: 退出测试")
        print("=" * 50)
        
        self.env.reset()
        running = True
        step_count = 0
        
        while running:
            # 处理输入事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 手动控制 - 增加动作幅度和响应速度
            action = np.zeros(4)
            action_multiplier = 3.0  # 🚀 增加动作幅度，让运动更快
            
            # 基座关节控制
            if keys[pygame.K_w]: action[0] = action_multiplier
            if keys[pygame.K_s]: action[0] = -action_multiplier
            if keys[pygame.K_a]: action[0] = -action_multiplier
            if keys[pygame.K_d]: action[0] = action_multiplier
            
            # 关节1控制
            if keys[pygame.K_q]: action[1] = action_multiplier
            if keys[pygame.K_e]: action[1] = -action_multiplier
            
            # 关节2控制
            if keys[pygame.K_z]: action[2] = action_multiplier
            if keys[pygame.K_c]: action[2] = -action_multiplier
            
            # 关节3控制
            if keys[pygame.K_r]: action[3] = action_multiplier
            if keys[pygame.K_f]: action[3] = -action_multiplier
            
            # 🔥 激进模式 - 按住SPACE键时动作幅度翻倍
            if keys[pygame.K_SPACE]:
                action *= 2.0
                print(f"🔥 激进模式: 动作幅度 = {action}")
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            step_count += 1
            
            # 全面检查Joint稳定性
            issues = self.check_comprehensive_joint_stability()
            
            # 更新统计
            self.distance_issues += issues['distance_problems']
            self.angle_issues += issues['angle_problems']
            if issues['explosion_detected']:
                self.explosion_detections += 1
            if issues['collision_occurred']:
                self.collision_count += 1
            
            # 渲染环境
            self.env.render()
            
            # 在我们自己的屏幕上绘制状态信息
            if hasattr(self.env, 'screen') and self.env.screen:
                # 获取环境的渲染表面
                env_surface = self.env.screen.copy()
                
                # 在环境表面上绘制我们的状态信息
                self.draw_comprehensive_status(env_surface, issues, step_count)
                
                # 将合成的表面显示到我们的屏幕上
                self.screen.blit(env_surface, (0, 0))
                pygame.display.flip()
            else:
                # 如果环境没有屏幕，就在我们自己的屏幕上绘制
                self.screen.fill((50, 50, 50))
                self.draw_comprehensive_status(self.screen, issues, step_count)
                pygame.display.flip()
            
            # 控制帧率
            self.clock.tick(60)
            
            # 每100步输出统计
            if step_count % 100 == 0:
                total_issues = (self.distance_issues + self.angle_issues + 
                               self.explosion_detections + self.collision_count)
                print(f"📊 步骤{step_count}: 总问题数={total_issues} "
                      f"(距离:{self.distance_issues}, 角度:{self.angle_issues}, "
                      f"爆炸:{self.explosion_detections}, 碰撞:{self.collision_count})")
        
        # 最终报告
        total_issues = (self.distance_issues + self.angle_issues + 
                       self.explosion_detections + self.collision_count)
        
        print(f"\n📊 最终测试结果:")
        print(f"   总步数: {step_count}")
        print(f"   距离问题: {self.distance_issues}")
        print(f"   角度问题: {self.angle_issues}")
        print(f"   爆炸检测: {self.explosion_detections}")
        print(f"   碰撞次数: {self.collision_count}")
        print(f"   总问题数: {total_issues}")
        
        if total_issues == 0:
            print("   ✅ Joint完全稳定")
        elif total_issues < 10:
            print("   ⚠️ Joint轻微不稳定")
        else:
            print("   ❌ Joint严重不稳定")
        
        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    test = AccurateJointStabilityTest()
    test.run_test()

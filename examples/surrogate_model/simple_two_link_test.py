#!/usr/bin/env python3
"""
简单的两Link连接测试 - 专门测试Joint稳定性
"""

import sys
import os
import numpy as np
import pygame
import pymunk
import math
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)

class SimpleTwoLinkTest:
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("简单两Link连接测试")
        self.clock = pygame.time.Clock()
        
        # 创建PyMunk空间
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)  # 使用标准重力
        
        # 简单的物理参数
        self.space.damping = 0.99
        self.space.iterations = 10
        
        # 创建两个Link
        self.create_two_links()
        
        # 统计信息
        self.step_count = 0
        self.joint_break_count = 0
        
        print("🔧 简单两Link测试初始化完成")
        print(f"   Link1质量: {self.body1.mass}")
        print(f"   Link2质量: {self.body2.mass}")
        print(f"   Joint约束力: {self.joint.max_force}")
        
    def create_two_links(self):
        """创建两个简单的Link"""
        
        # Link1 - 固定在屏幕中央
        mass1 = 10
        length1 = 100
        moment1 = pymunk.moment_for_segment(mass1, (0, 0), (length1, 0), 5)
        self.body1 = pymunk.Body(mass1, moment1)
        self.body1.position = 400, 200  # 屏幕中央偏上
        
        # 创建Link1的形状
        self.shape1 = pymunk.Segment(self.body1, (0, 0), (length1, 0), 5)
        self.shape1.friction = 0.5
        self.shape1.color = (255, 0, 0, 255)  # 红色
        
        # Link2 - 连接到Link1的末端
        mass2 = 8
        length2 = 80
        moment2 = pymunk.moment_for_segment(mass2, (0, 0), (length2, 0), 5)
        self.body2 = pymunk.Body(mass2, moment2)
        self.body2.position = 500, 200  # Link1末端位置
        
        # 创建Link2的形状
        self.shape2 = pymunk.Segment(self.body2, (0, 0), (length2, 0), 5)
        self.shape2.friction = 0.5
        self.shape2.color = (0, 255, 0, 255)  # 绿色
        
        # 将Link1固定到空间
        self.fixed_joint = pymunk.PivotJoint(self.space.static_body, self.body1, (400, 200), (0, 0))
        self.fixed_joint.max_force = 50000
        
        # 创建Link1和Link2之间的关节
        self.joint = pymunk.PivotJoint(self.body1, self.body2, (length1, 0), (0, 0))
        self.joint.max_force = 100000  # 🔧 增加约束力测试
        self.joint.collide_bodies = False
        
        # 添加到空间
        self.space.add(self.body1, self.shape1)
        self.space.add(self.body2, self.shape2)
        self.space.add(self.fixed_joint)
        self.space.add(self.joint)
        
        print(f"✅ 创建两个Link:")
        print(f"   Link1: 位置{self.body1.position}, 长度{length1}")
        print(f"   Link2: 位置{self.body2.position}, 长度{length2}")
        print(f"   Joint约束力: {self.joint.max_force}")
    
    def check_joint_stability(self):
        """检查Joint稳定性"""
        # 计算两个Link之间的实际距离
        joint_pos1 = self.body1.local_to_world((100, 0))  # Link1末端
        joint_pos2 = self.body2.local_to_world((0, 0))    # Link2起始点
        
        distance = math.sqrt((joint_pos1[0] - joint_pos2[0])**2 + (joint_pos1[1] - joint_pos2[1])**2)
        
        # 如果距离超过阈值，认为Joint断开
        if distance > 10:  # 10像素的容差
            self.joint_break_count += 1
            print(f"⚠️ 步骤{self.step_count}: Joint断开! 距离={distance:.2f}")
            return False
        return True
    
    def apply_external_force(self):
        """施加外力测试Joint强度"""
        # 每100步施加一次冲击力
        if self.step_count % 100 == 0:
            # 向Link2施加随机方向的冲击力
            force_x = np.random.uniform(-5000, 5000)
            force_y = np.random.uniform(-5000, 5000)
            self.body2.apply_force_at_world_point((force_x, force_y), self.body2.position)
            print(f"💥 步骤{self.step_count}: 施加冲击力 ({force_x:.0f}, {force_y:.0f})")
    
    def draw(self):
        """绘制场景"""
        self.screen.fill((255, 255, 255))  # 白色背景
        
        # 绘制Link1 (红色)
        start1 = self.body1.local_to_world((0, 0))
        end1 = self.body1.local_to_world((100, 0))
        pygame.draw.line(self.screen, (255, 0, 0), start1, end1, 10)
        
        # 绘制Link2 (绿色)
        start2 = self.body2.local_to_world((0, 0))
        end2 = self.body2.local_to_world((80, 0))
        pygame.draw.line(self.screen, (0, 255, 0), start2, end2, 8)
        
        # 绘制关节点 (蓝色圆圈)
        joint_pos1 = self.body1.local_to_world((100, 0))
        joint_pos2 = self.body2.local_to_world((0, 0))
        pygame.draw.circle(self.screen, (0, 0, 255), (int(joint_pos1[0]), int(joint_pos1[1])), 8)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(joint_pos2[0]), int(joint_pos2[1])), 6)
        
        # 绘制连接线 (如果Joint正常，连接线应该很短)
        distance = math.sqrt((joint_pos1[0] - joint_pos2[0])**2 + (joint_pos1[1] - joint_pos2[1])**2)
        color = (0, 255, 0) if distance < 5 else (255, 0, 0)  # 绿色=正常，红色=异常
        pygame.draw.line(self.screen, color, joint_pos1, joint_pos2, 3)
        
        # 显示信息
        font = pygame.font.Font(None, 36)
        info_text = [
            f"步骤: {self.step_count}",
            f"Joint断开次数: {self.joint_break_count}",
            f"Joint距离: {distance:.2f}px",
            f"Joint状态: {'✅ 稳定' if distance < 5 else '❌ 断开'}"
        ]
        
        for i, text in enumerate(info_text):
            color = (0, 0, 0) if i < 3 else ((0, 128, 0) if distance < 5 else (255, 0, 0))
            surface = font.render(text, True, color)
            self.screen.blit(surface, (10, 10 + i * 40))
        
        # 显示控制说明
        control_text = [
            "控制说明:",
            "空格键: 施加随机冲击力",
            "R键: 重置",
            "ESC键: 退出"
        ]
        
        small_font = pygame.font.Font(None, 24)
        for i, text in enumerate(control_text):
            surface = small_font.render(text, True, (100, 100, 100))
            self.screen.blit(surface, (10, 200 + i * 25))
        
        pygame.display.flip()
    
    def handle_input(self):
        """处理用户输入"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # 手动施加冲击力
                    force_x = np.random.uniform(-10000, 10000)
                    force_y = np.random.uniform(-10000, 10000)
                    self.body2.apply_force_at_world_point((force_x, force_y), self.body2.position)
                    print(f"💥 手动冲击力: ({force_x:.0f}, {force_y:.0f})")
                elif event.key == pygame.K_r:
                    # 重置
                    self.reset()
        return True
    
    def reset(self):
        """重置测试"""
        self.body1.position = 400, 200
        self.body1.velocity = 0, 0
        self.body1.angular_velocity = 0
        self.body1.angle = 0
        
        self.body2.position = 500, 200
        self.body2.velocity = 0, 0
        self.body2.angular_velocity = 0
        self.body2.angle = 0
        
        self.step_count = 0
        self.joint_break_count = 0
        print("🔄 测试重置")
    
    def run(self):
        """运行测试"""
        running = True
        dt = 1/60.0
        
        print("🚀 开始简单两Link连接测试")
        print("   观察红色和绿色Link之间的蓝色连接点")
        print("   绿色连接线=Joint稳定，红色连接线=Joint断开")
        
        while running:
            running = self.handle_input()
            
            # 施加测试力
            if self.step_count > 60:  # 前1秒让系统稳定
                self.apply_external_force()
            
            # 物理模拟步进
            self.space.step(dt)
            self.step_count += 1
            
            # 检查Joint稳定性
            self.check_joint_stability()
            
            # 绘制
            self.draw()
            self.clock.tick(60)
            
            # 每500步输出统计
            if self.step_count % 500 == 0:
                print(f"📊 步骤{self.step_count}: Joint断开次数={self.joint_break_count}")
        
        pygame.quit()
        
        # 最终统计
        print("\n📊 测试结果:")
        print(f"   总步数: {self.step_count}")
        print(f"   Joint断开次数: {self.joint_break_count}")
        if self.joint_break_count == 0:
            print("   ✅ Joint完全稳定")
        else:
            print(f"   ❌ Joint不稳定，断开率: {self.joint_break_count/self.step_count*100:.2f}%")

if __name__ == "__main__":
    test = SimpleTwoLinkTest()
    test.run()

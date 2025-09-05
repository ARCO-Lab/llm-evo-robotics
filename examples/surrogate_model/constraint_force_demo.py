#!/usr/bin/env python3
"""
演示PyMunk约束力机制的原理
"""

import sys
import os
import numpy as np
import pygame
import pymunk
import math

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)

class ConstraintForceDemo:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("PyMunk约束力机制演示")
        self.clock = pygame.time.Clock()
        
        # 创建多个测试场景
        self.create_test_scenarios()
        
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        
    def create_test_scenarios(self):
        """创建多个不同约束力的测试场景"""
        self.scenarios = []
        
        # 场景1: 约束力 = 1000 (很小)
        space1 = pymunk.Space()
        space1.gravity = (0, 500)
        scenario1 = self.create_scenario(space1, 1000, "约束力 = 1,000 (很小)", 100, 100)
        self.scenarios.append(scenario1)
        
        # 场景2: 约束力 = 10000 (中等)
        space2 = pymunk.Space()
        space2.gravity = (0, 500)
        scenario2 = self.create_scenario(space2, 10000, "约束力 = 10,000 (中等)", 400, 100)
        self.scenarios.append(scenario2)
        
        # 场景3: 约束力 = 100000 (大)
        space3 = pymunk.Space()
        space3.gravity = (0, 500)
        scenario3 = self.create_scenario(space3, 100000, "约束力 = 100,000 (大)", 700, 100)
        self.scenarios.append(scenario3)
        
        # 场景4: 约束力 = 无限 (float('inf'))
        space4 = pymunk.Space()
        space4.gravity = (0, 500)
        scenario4 = self.create_scenario(space4, float('inf'), "约束力 = 无限 (理论刚性)", 1000, 100)
        self.scenarios.append(scenario4)
        
        print("🔧 创建了4个不同约束力的测试场景")
        
    def create_scenario(self, space, max_force, title, x_offset, y_offset):
        """创建单个测试场景"""
        
        # 创建两个连接的物体
        # 物体1 - 固定
        body1 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body1.position = x_offset, y_offset
        shape1 = pymunk.Circle(body1, 20)
        shape1.color = (255, 0, 0, 255)  # 红色
        space.add(body1, shape1)
        
        # 物体2 - 动态，会受重力影响
        mass = 10
        moment = pymunk.moment_for_circle(mass, 0, 15)
        body2 = pymunk.Body(mass, moment)
        body2.position = x_offset, y_offset + 60
        shape2 = pymunk.Circle(body2, 15)
        shape2.color = (0, 255, 0, 255)  # 绿色
        space.add(body2, shape2)
        
        # 创建关节约束
        joint = pymunk.PinJoint(body1, body2, (0, 0), (0, 0))
        joint.max_force = max_force
        space.add(joint)
        
        return {
            'space': space,
            'title': title,
            'max_force': max_force,
            'body1': body1,
            'body2': body2,
            'joint': joint,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'broken': False,
            'distance_history': []
        }
    
    def update_scenarios(self, dt):
        """更新所有场景"""
        for scenario in self.scenarios:
            space = scenario['space']
            body1 = scenario['body1']
            body2 = scenario['body2']
            
            # 物理步进
            space.step(dt)
            
            # 计算距离
            distance = body1.position.get_distance(body2.position)
            scenario['distance_history'].append(distance)
            
            # 保持历史记录长度
            if len(scenario['distance_history']) > 100:
                scenario['distance_history'].pop(0)
            
            # 检查是否"断开"
            if distance > 100:  # 如果距离超过100像素，认为断开
                scenario['broken'] = True
    
    def draw_scenario(self, scenario):
        """绘制单个场景"""
        space = scenario['space']
        x_offset = scenario['x_offset']
        y_offset = scenario['y_offset']
        
        # 绘制物体
        for body in space.bodies:
            if body.body_type == pymunk.Body.KINEMATIC:
                # 固定物体 (红色)
                pos = int(body.position.x), int(body.position.y)
                pygame.draw.circle(self.screen, (255, 0, 0), pos, 20)
                pygame.draw.circle(self.screen, (0, 0, 0), pos, 20, 2)
            else:
                # 动态物体 (绿色)
                pos = int(body.position.x), int(body.position.y)
                pygame.draw.circle(self.screen, (0, 255, 0), pos, 15)
                pygame.draw.circle(self.screen, (0, 0, 0), pos, 15, 2)
        
        # 绘制连接线
        bodies_list = list(space.bodies)
        if len(bodies_list) >= 2:
            body1, body2 = bodies_list[0], bodies_list[1]
            distance = body1.position.get_distance(body2.position)
            
            # 根据距离选择颜色
            if distance < 70:
                color = (0, 255, 0)  # 绿色 - 正常
            elif distance < 100:
                color = (255, 255, 0)  # 黄色 - 拉伸
            else:
                color = (255, 0, 0)  # 红色 - 断开
            
            pygame.draw.line(self.screen, color, 
                           (int(body1.position.x), int(body1.position.y)),
                           (int(body2.position.x), int(body2.position.y)), 3)
        
        # 绘制标题
        title_surface = self.font.render(scenario['title'], True, (0, 0, 0))
        self.screen.blit(title_surface, (x_offset - 50, y_offset - 40))
        
        # 绘制约束力信息
        max_force_text = f"Max Force: {scenario['max_force']}"
        if scenario['max_force'] == float('inf'):
            max_force_text = "Max Force: ∞"
        force_surface = self.font.render(max_force_text, True, (100, 100, 100))
        self.screen.blit(force_surface, (x_offset - 50, y_offset - 20))
        
        # 绘制距离信息
        bodies_list = list(space.bodies)
        if len(bodies_list) >= 2:
            distance = bodies_list[0].position.get_distance(bodies_list[1].position)
            distance_text = f"距离: {distance:.1f}px"
            distance_surface = self.font.render(distance_text, True, (0, 0, 0))
            self.screen.blit(distance_surface, (x_offset - 50, y_offset + 120))
            
            # 状态
            status = "❌ 断开" if scenario['broken'] else "✅ 连接"
            status_color = (255, 0, 0) if scenario['broken'] else (0, 128, 0)
            status_surface = self.font.render(status, True, status_color)
            self.screen.blit(status_surface, (x_offset - 50, y_offset + 140))
    
    def draw_explanation(self):
        """绘制原理解释"""
        explanations = [
            "🔧 PyMunk约束力机制原理:",
            "",
            "1. 约束求解器每步迭代计算修正冲量",
            "2. max_force限制单次修正的最大冲量",
            "3. 约束力不足 → 无法维持连接 → Joint断开",
            "4. 约束力过大 → 数值不稳定 → 系统爆炸",
            "5. 无限约束力 → 理论刚性，但可能不稳定",
            "",
            "💡 观察不同约束力下的表现:",
            "• 绿线 = 连接正常",
            "• 黄线 = 连接拉伸",  
            "• 红线 = 连接断开",
            "",
            "按ESC退出"
        ]
        
        start_y = 400
        for i, text in enumerate(explanations):
            color = (0, 0, 0) if not text.startswith(('🔧', '💡')) else (0, 0, 128)
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (50, start_y + i * 25))
    
    def run(self):
        """运行演示"""
        running = True
        dt = 1/60.0
        
        print("🚀 PyMunk约束力机制演示开始")
        print("   观察不同约束力设置下的Joint行为")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 更新物理
            self.update_scenarios(dt)
            
            # 绘制
            self.screen.fill((255, 255, 255))
            
            # 绘制所有场景
            for scenario in self.scenarios:
                self.draw_scenario(scenario)
            
            # 绘制解释
            self.draw_explanation()
            
            # 绘制标题
            title = self.big_font.render("PyMunk约束力机制演示", True, (0, 0, 128))
            self.screen.blit(title, (400, 20))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        
        # 总结结果
        print("\n📊 演示结果总结:")
        for i, scenario in enumerate(self.scenarios):
            status = "断开" if scenario['broken'] else "稳定"
            print(f"   场景{i+1} ({scenario['title']}): {status}")

if __name__ == "__main__":
    demo = ConstraintForceDemo()
    demo.run()

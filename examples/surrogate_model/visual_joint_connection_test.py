#!/usr/bin/env python3
"""
可视化Joint连接线测试 - 直观显示Joint连接状态
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

class VisualJointConnectionTest:
    def __init__(self):
        self.env = Reacher2DEnv(
            num_links=4,
            link_lengths=[80, 60, 50, 40],
            render_mode="human",
            config_path="configs/reacher_with_zigzag_obstacles.yaml",
            debug_level='WARNING'
        )
        
        pygame.init()
        self.screen = pygame.display.set_mode((900, 900))
        pygame.display.set_caption("Joint连接可视化测试 - 绿线表示Joint连接")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
    def draw_joint_connections(self, surface):
        """绘制Joint连接线和状态信息"""
        
        # 绘制基座连接线（锚点到Link0中心）
        anchor_pos = self.env.anchor_point
        base_pos = self.env.bodies[0].position
        
        # 计算基座连接距离
        base_distance = math.sqrt((base_pos.x - anchor_pos[0])**2 + 
                                 (base_pos.y - anchor_pos[1])**2)
        
        # 绘制基座连接线
        color = (0, 255, 0) if base_distance < 5.0 else (255, 0, 0)  # 绿色=稳定，红色=断开
        pygame.draw.line(surface, color, 
                        (int(anchor_pos[0]), int(anchor_pos[1])),
                        (int(base_pos.x), int(base_pos.y)), 3)
        
        # 绘制基座锚点
        pygame.draw.circle(surface, (255, 255, 0), 
                          (int(anchor_pos[0]), int(anchor_pos[1])), 8)
        
        joint_info = [f"基座Joint: {base_distance:.2f}px"]
        
        # 绘制Link间连接线
        for i in range(1, len(self.env.bodies)):
            body_a = self.env.bodies[i-1]
            body_b = self.env.bodies[i]
            expected_distance = self.env.link_lengths[i-1]
            
            # 计算实际距离
            actual_distance = math.sqrt((body_a.position.x - body_b.position.x)**2 + 
                                      (body_a.position.y - body_b.position.y)**2)
            
            # 计算偏差
            deviation = abs(actual_distance - expected_distance)
            deviation_percentage = deviation / expected_distance * 100
            
            # 确定颜色（绿色=稳定，黄色=轻微偏差，红色=严重偏差）
            if deviation_percentage < 10:
                color = (0, 255, 0)  # 绿色
                status = "✅"
            elif deviation_percentage < 50:
                color = (255, 255, 0)  # 黄色
                status = "⚠️"
            else:
                color = (255, 0, 0)  # 红色
                status = "❌"
            
            # 绘制连接线
            pygame.draw.line(surface, color,
                           (int(body_a.position.x), int(body_a.position.y)),
                           (int(body_b.position.x), int(body_b.position.y)), 3)
            
            # 在连接线中点绘制Joint编号
            mid_x = (body_a.position.x + body_b.position.x) / 2
            mid_y = (body_a.position.y + body_b.position.y) / 2
            
            text = self.font.render(f"J{i}", True, (255, 255, 255))
            surface.blit(text, (int(mid_x) - 10, int(mid_y) - 10))
            
            joint_info.append(f"Joint{i}: {actual_distance:.2f}px (预期{expected_distance}px) {status}")
        
        # 绘制Joint状态信息
        y_offset = 10
        for info in joint_info:
            text = self.font.render(info, True, (255, 255, 255))
            surface.blit(text, (10, y_offset))
            y_offset += 25
        
        # 绘制图例
        legend_y = 200
        legend_items = [
            ("绿线: Joint连接稳定", (0, 255, 0)),
            ("黄线: Joint轻微偏差", (255, 255, 0)),
            ("红线: Joint严重偏差", (255, 0, 0)),
            ("黄圆: 基座锚点", (255, 255, 0))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            pygame.draw.line(surface, color, (10, legend_y + i*20), (30, legend_y + i*20), 3)
            text_surface = self.font.render(text, True, (255, 255, 255))
            surface.blit(text_surface, (35, legend_y + i*20 - 8))
    
    def run_test(self):
        """运行可视化测试"""
        print("🎮 Joint连接可视化测试")
        print("=" * 50)
        print("控制说明:")
        print("  WASD: 控制基座关节")
        print("  QE: 控制关节1")
        print("  ZC: 控制关节2")
        print("  RF: 控制关节3")
        print("  1: 强力模式切换")
        print("  ESC: 退出")
        print("=" * 50)
        
        self.env.reset()
        
        running = True
        step_count = 0
        joint_issue_count = 0
        
        while running and step_count < 1000:
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 获取按键状态进行手动控制
            keys = pygame.key.get_pressed()
            action = [0.0, 0.0, 0.0, 0.0]
            
            # 检查强力模式
            power_mode = keys[pygame.K_1]
            force = 10.0 if power_mode else 3.0
            
            # 控制映射
            if keys[pygame.K_w]:
                action[0] = -force
            elif keys[pygame.K_s]:
                action[0] = force
            if keys[pygame.K_a]:
                action[1] = -force
            elif keys[pygame.K_d]:
                action[1] = force
            if keys[pygame.K_q]:
                action[2] = -force
            elif keys[pygame.K_e]:
                action[2] = force
            if keys[pygame.K_z]:
                action[3] = -force
            elif keys[pygame.K_c]:
                action[3] = force
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            step_count += 1
            
            # 检查Joint连接状态
            current_issues = 0
            
            # 检查基座连接
            base_pos = self.env.bodies[0].position
            anchor_pos = self.env.anchor_point
            base_distance = math.sqrt((base_pos.x - anchor_pos[0])**2 + 
                                    (base_pos.y - anchor_pos[1])**2)
            if base_distance > 5.0:
                current_issues += 1
            
            # 检查Link间连接
            for i in range(1, len(self.env.bodies)):
                body_a = self.env.bodies[i-1]
                body_b = self.env.bodies[i]
                expected_distance = self.env.link_lengths[i-1]
                
                actual_distance = math.sqrt((body_a.position.x - body_b.position.x)**2 + 
                                          (body_a.position.y - body_b.position.y)**2)
                
                deviation = abs(actual_distance - expected_distance)
                if deviation > expected_distance * 0.5:  # 偏差超过50%
                    current_issues += 1
            
            joint_issue_count += current_issues
            
            # 清屏
            self.screen.fill((0, 0, 0))
            
            # 渲染环境
            self.env.render()
            
            # 绘制Joint连接线
            self.draw_joint_connections(self.screen)
            
            # 绘制控制信息
            control_info = [
                f"步数: {step_count}",
                f"Joint问题计数: {joint_issue_count}",
                f"强力模式: {'开启' if power_mode else '关闭'}",
                "",
                "控制: WASD(基座) QE(J1) ZC(J2) RF(J3)",
                "1键: 强力模式切换"
            ]
            
            for i, info in enumerate(control_info):
                text = self.font.render(info, True, (255, 255, 255))
                self.screen.blit(text, (650, 10 + i*25))
            
            # 更新显示
            pygame.display.flip()
            self.clock.tick(60)
        
        print(f"\n📊 测试结果:")
        print(f"   总步数: {step_count}")
        print(f"   Joint问题总计数: {joint_issue_count}")
        print(f"   Joint连接状态: {'❌ 有问题' if joint_issue_count > 0 else '✅ 完全稳定'}")
        
        self.env.close()
        pygame.quit()

if __name__ == "__main__":
    test = VisualJointConnectionTest()
    test.run_test()


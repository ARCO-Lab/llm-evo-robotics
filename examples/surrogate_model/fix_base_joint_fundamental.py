#!/usr/bin/env python3
"""
彻底修复基座关节穿透问题
通过重新设计基座关节的连接方式来解决PyMunk的static_body连接问题
"""

import sys
import os
import numpy as np
import pygame
import pymunk
import math
import yaml

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

class FixedBaseJointReacher2DEnv:
    """
    修复基座关节穿透问题的Reacher2D环境
    核心思路：创建一个固定的虚拟基座Body，而不是直接连接到static_body
    """
    
    def __init__(self, num_links=4, link_lengths=None, render_mode=None, config_path=None):
        self.num_links = num_links
        self.link_lengths = link_lengths or [80, 60, 50, 40]
        self.render_mode = render_mode
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.anchor_point = self.config["start"]["position"]
        self.goal_pos = np.array(self.config["goal"]["position"])
        self.goal_radius = self.config["goal"]["radius"]
        
        # 创建物理空间
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)
        
        # 碰撞统计
        self.base_collision_count = 0
        self.other_collision_count = 0
        
        # 创建机器人和障碍物
        self._create_fixed_base_robot()
        self._create_obstacles()
        self._setup_collision_handlers()
        
        # 渲染设置
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((800, 800))
            pygame.display.set_caption("修复基座关节穿透问题")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if not config_path:
            # 默认配置
            return {
                "start": {"position": [500, 620]},
                "goal": {"position": [600, 550], "radius": 10},
                "obstacles": [
                    {"shape": "segment", "points": [[500, 487], [550, 537]]},
                    {"shape": "segment", "points": [[550, 537], [600, 487]]},
                    {"shape": "segment", "points": [[600, 487], [650, 537]]},
                    {"shape": "segment", "points": [[650, 537], [700, 487]]},
                    {"shape": "segment", "points": [[500, 612], [550, 662]]},
                    {"shape": "segment", "points": [[550, 662], [600, 612]]},
                    {"shape": "segment", "points": [[600, 612], [650, 662]]},
                    {"shape": "segment", "points": [[650, 662], [700, 612]]}
                ]
            }
        
        try:
            config_full_path = os.path.join("examples/2d_reacher", config_path)
            with open(config_full_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            print(f"❌ 无法加载配置文件: {config_path}")
            return self._load_config(None)
    
    def _create_fixed_base_robot(self):
        """
        创建修复基座关节的机器人
        核心修复：使用虚拟固定基座Body代替static_body连接
        """
        print("🔧 创建修复版基座关节机器人...")
        
        self.bodies = []
        self.joints = []
        self.motors = []
        self.obstacles = []
        
        # 🎯 关键修复1：创建虚拟固定基座Body
        # 这个基座Body是固定的，但不是static_body，因此可以参与碰撞检测
        base_anchor_mass = 1000000  # 极大质量，实际上固定不动
        base_anchor_moment = pymunk.moment_for_circle(base_anchor_mass, 0, 5)
        self.base_anchor_body = pymunk.Body(base_anchor_mass, base_anchor_moment)
        self.base_anchor_body.position = self.anchor_point
        
        # 创建基座锚点形状（不可见，只用于物理）
        base_anchor_shape = pymunk.Circle(self.base_anchor_body, 5)
        base_anchor_shape.collision_type = 999  # 特殊碰撞类型，不与任何东西碰撞
        base_anchor_shape.sensor = True  # 设为传感器，不产生物理碰撞
        
        self.space.add(self.base_anchor_body, base_anchor_shape)
        
        # 🎯 关键修复2：将基座锚点固定到世界
        # 使用PinJoint将基座锚点完全固定
        anchor_pin = pymunk.PinJoint(self.space.static_body, self.base_anchor_body, 
                                   self.anchor_point, (0, 0))
        anchor_pin.stiffness = 1e10  # 极高刚度
        anchor_pin.damping = 1e8     # 极高阻尼
        self.space.add(anchor_pin)
        
        print(f"✅ 创建虚拟基座锚点: {self.anchor_point}")
        
        # 创建机器人Link
        density = 0.8
        current_pos = list(self.anchor_point)
        
        for i in range(self.num_links):
            length = self.link_lengths[i]
            mass = density * length * 10
            moment = pymunk.moment_for_segment(mass, (0, 0), (length, 0), 8)
            body = pymunk.Body(mass, moment)
            
            # 设置初始位置（垂直下垂）
            body.position = current_pos
            body.angle = math.pi/2
            
            # 创建形状
            shape = pymunk.Segment(body, (0, 0), (length, 0), 8)
            shape.friction = 0.8
            shape.collision_type = i + 1  # Link0 = 1, Link1 = 2, ...
            shape.collision_slop = 0.01
            
            self.space.add(body, shape)
            self.bodies.append(body)
            
            # 🎯 关键修复3：基座关节连接到虚拟基座Body而不是static_body
            if i == 0:
                # 基座关节连接到虚拟基座锚点
                joint = pymunk.PivotJoint(self.base_anchor_body, body, (0, 0), (0, 0))
                joint.collide_bodies = False  # 不让基座锚点与基座关节碰撞
                self.space.add(joint)
                self.joints.append(joint)
                
                # 基座关节的Motor
                motor = pymunk.SimpleMotor(self.base_anchor_body, body, 0.0)
                motor.max_force = 50000
                self.space.add(motor)
                self.motors.append(motor)
                
                print(f"✅ 基座关节连接到虚拟基座锚点")
                
            else:
                # 其他关节正常连接
                prev_body = self.bodies[i-1]
                joint = pymunk.PivotJoint(prev_body, body, (length, 0), (0, 0))
                joint.collide_bodies = False
                self.space.add(joint)
                self.joints.append(joint)
                
                # 其他关节的Motor
                motor = pymunk.SimpleMotor(prev_body, body, 0.0)
                motor.max_force = 50000
                self.space.add(motor)
                self.motors.append(motor)
                
                print(f"✅ Link{i}连接到Link{i-1}")
            
            # 更新下一个Link的起始位置
            if i < self.num_links - 1:
                end_x = current_pos[0] + length * math.cos(math.pi/2)
                end_y = current_pos[1] + length * math.sin(math.pi/2)
                current_pos = [end_x, end_y]
        
        print(f"✅ 创建了{self.num_links}个Link的修复版机器人")
    
    def _create_obstacles(self):
        """创建障碍物"""
        OBSTACLE_COLLISION_TYPE = 100
        
        for i, obs in enumerate(self.config["obstacles"]):
            if obs["shape"] == "segment":
                p1 = tuple(obs["points"][0])
                p2 = tuple(obs["points"][1])
                shape = pymunk.Segment(self.space.static_body, p1, p2, radius=5.0)
                shape.friction = 1.0
                shape.collision_type = OBSTACLE_COLLISION_TYPE
                shape.collision_slop = 0.01
                shape.color = (0, 0, 0, 255)
                
                self.space.add(shape)
                self.obstacles.append(shape)
                
                print(f"✅ 创建障碍物{i}: {p1} -> {p2}")
    
    def _setup_collision_handlers(self):
        """设置碰撞处理器"""
        OBSTACLE_COLLISION_TYPE = 100
        
        def base_joint_collision_handler(arbiter, space, data):
            """基座关节专用碰撞处理器"""
            self.base_collision_count += 1
            print(f"🎯 [修复版] 基座关节碰撞障碍物! 计数: {self.base_collision_count}")
            
            # 设置碰撞响应
            arbiter.restitution = 0.2
            arbiter.friction = 1.5
            return True
        
        def other_link_collision_handler(link_id):
            """其他Link的碰撞处理器"""
            def handler(arbiter, space, data):
                self.other_collision_count += 1
                print(f"🚨 Link{link_id}碰撞障碍物! 总计: {self.other_collision_count}")
                return True
            return handler
        
        # 为每个Link设置碰撞检测
        for i in range(self.num_links):
            link_collision_type = i + 1
            
            try:
                if i == 0:  # 基座关节特殊处理
                    self.space.on_collision(
                        collision_type_a=link_collision_type,
                        collision_type_b=OBSTACLE_COLLISION_TYPE,
                        begin=base_joint_collision_handler,
                        pre_solve=base_joint_collision_handler
                    )
                    print(f"✅ [修复版] 设置基座关节碰撞检测")
                else:
                    self.space.on_collision(
                        collision_type_a=link_collision_type,
                        collision_type_b=OBSTACLE_COLLISION_TYPE,
                        begin=other_link_collision_handler(i)
                    )
                    print(f"✅ 设置Link{i}碰撞检测")
                    
            except Exception as e:
                print(f"❌ 设置Link{i}碰撞检测失败: {e}")
    
    def step(self, actions):
        """执行一步仿真"""
        # 应用动作到Motors
        for i, action in enumerate(actions[:len(self.motors)]):
            self.motors[i].rate = action * 0.01  # 动作缩放
        
        # 物理仿真
        self.space.step(1/60.0)
        
        # 计算状态
        end_effector_pos = self._get_end_effector_position()
        distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
        # 简单奖励
        reward = -distance_to_goal * 0.01
        done = distance_to_goal < self.goal_radius
        
        return None, reward, done, {"distance": distance_to_goal}
    
    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        if not self.bodies:
            return np.array(self.anchor_point)
        
        last_body = self.bodies[-1]
        last_length = self.link_lengths[-1]
        
        end_x = last_body.position[0] + last_length * math.cos(last_body.angle)
        end_y = last_body.position[1] + last_length * math.sin(last_body.angle)
        
        return np.array([end_x, end_y])
    
    def render(self):
        """渲染环境"""
        if self.render_mode != "human":
            return
        
        self.screen.fill((255, 255, 255))
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            p1 = obstacle.a
            p2 = obstacle.b
            pygame.draw.line(self.screen, (0, 0, 0), p1, p2, 10)
        
        # 绘制机器人
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, body in enumerate(self.bodies):
            start_pos = body.position
            length = self.link_lengths[i]
            end_x = start_pos[0] + length * math.cos(body.angle)
            end_y = start_pos[1] + length * math.sin(body.angle)
            end_pos = (end_x, end_y)
            
            color = colors[i % len(colors)]
            pygame.draw.line(self.screen, color, start_pos, end_pos, 16)
            
            # 绘制关节
            pygame.draw.circle(self.screen, (50, 50, 50), (int(start_pos[0]), int(start_pos[1])), 8)
        
        # 绘制基座锚点
        pygame.draw.circle(self.screen, (100, 100, 100), 
                         (int(self.anchor_point[0]), int(self.anchor_point[1])), 12)
        
        # 绘制目标
        pygame.draw.circle(self.screen, (0, 255, 0), 
                         (int(self.goal_pos[0]), int(self.goal_pos[1])), self.goal_radius)
        
        # 绘制末端执行器
        end_pos = self._get_end_effector_position()
        pygame.draw.circle(self.screen, (0, 0, 255), 
                         (int(end_pos[0]), int(end_pos[1])), 5)
        
        pygame.display.flip()
    
    def reset(self):
        """重置环境"""
        # 重置机器人姿态
        current_pos = list(self.anchor_point)
        for i, body in enumerate(self.bodies):
            body.position = current_pos
            body.angle = math.pi/2
            body.velocity = (0, 0)
            body.angular_velocity = 0
            
            # 更新位置
            if i < self.num_links - 1:
                length = self.link_lengths[i]
                end_x = current_pos[0] + length * math.cos(math.pi/2)
                end_y = current_pos[1] + length * math.sin(math.pi/2)
                current_pos = [end_x, end_y]
        
        return None
    
    def close(self):
        """关闭环境"""
        if self.render_mode == "human":
            pygame.quit()

def test_fixed_base_joint():
    """测试修复版基座关节"""
    print("🛠️ 测试修复版基座关节")
    print("=" * 50)
    
    # 创建修复版环境
    env = FixedBaseJointReacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    env.reset()
    
    print(f"\n🎮 开始测试:")
    print("  自动执行D+W组合动作")
    print("  期望: 基座关节能够正确与障碍物碰撞")
    print("  Q: 退出")
    
    running = True
    step_count = 0
    
    while running and step_count < 1000:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # 自动执行强力测试动作
        actions = np.array([100, -80, 0, 0])  # 基座关节大力转动
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 渲染
        env.render()
        
        # 显示统计信息
        info_texts = [
            f"步数: {step_count}",
            f"修复版测试中...",
            "",
            "🚨 碰撞统计:",
            f"基座关节碰撞: {env.base_collision_count}",
            f"其他Link碰撞: {env.other_collision_count}",
            f"总碰撞: {env.base_collision_count + env.other_collision_count}",
            "",
            f"🔍 修复状态:",
            f"{'✅ 成功!' if env.base_collision_count > 0 else '❌ 仍有问题'}",
            "",
            "Q: 退出"
        ]
        
        # 显示信息
        info_surface = pygame.Surface((300, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "碰撞统计" in text:
                    color = (255, 200, 100)
                elif f"基座关节碰撞: {env.base_collision_count}" in text and env.base_collision_count > 0:
                    color = (100, 255, 100)  # 绿色表示成功
                elif "✅ 成功!" in text:
                    color = (100, 255, 100)
                elif "❌ 仍有问题" in text:
                    color = (255, 100, 100)
                
                surface = env.font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每200步输出统计
        if step_count % 200 == 0:
            print(f"\n📊 步数{step_count}统计:")
            print(f"   基座关节碰撞: {env.base_collision_count}")
            print(f"   其他Link碰撞: {env.other_collision_count}")
            
            if env.base_collision_count > 0:
                print("✅ 修复成功! 基座关节可以正确碰撞!")
                break
        
        if done:
            env.reset()
        
        env.clock.tick(60)
    
    # 最终结果
    print(f"\n🎯 最终测试结果:")
    print("=" * 40)
    print(f"测试步数: {step_count}")
    print(f"基座关节碰撞: {env.base_collision_count}")
    print(f"其他Link碰撞: {env.other_collision_count}")
    print(f"总碰撞: {env.base_collision_count + env.other_collision_count}")
    
    if env.base_collision_count > 0:
        print(f"\n🎉 修复成功!")
        print("   基座关节现在可以正确与障碍物碰撞")
        print("   可以将此修复应用到主环境文件")
    else:
        print(f"\n😔 修复仍未成功")
        print("   需要进一步调查PyMunk的行为")
    
    env.close()

if __name__ == "__main__":
    test_fixed_base_joint()


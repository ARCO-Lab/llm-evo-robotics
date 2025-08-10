# import gymnasium as gym
# from gymnasium import Env
# from gymnasium.spaces import Box

import gym

from gym import Env

from gym.spaces import Box

from pymunk import Segment
import pymunk
import pymunk.pygame_util  # 明确导入pygame_util
import numpy as np
import pygame
import math
import yaml

import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/configs'))
print(sys.path)

class Reacher2DEnv(Env):

    
    def __init__(self, num_links=3, link_lengths=None, render_mode=None, config_path=None):

        super().__init__()
        self.config = self._load_config(config_path)
        print(f"self.config: {self.config}")
        self.anchor_point = self.config["start"]["position"]
        self.gym_api_version = "old" # old or new. new is gymnasium, old is gym

        self.num_links = num_links  # 修复：使用传入的参数
        if link_lengths is None:

            self.link_lengths = [60] * num_links

        else:
            assert len(link_lengths) == num_links
            self.link_lengths = link_lengths
        
        self.render_mode = render_mode
        # self.goal_pos = np.array([250.0, 250.0])
        self.dt = 1/60.0  # 增加时间步长精度
        self.max_torque = 500  # 增加最大扭矩

        # 定义Gymnasium必需的action_space和observation_space
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(self.num_links,), dtype=np.float32)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 7,), dtype=np.float32)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 981.0)
        # 减少全局阻尼
        self.space.damping = 0.9  # 减少阻尼，让运动更明显
        self.obstacles = []
        self.bodies = []
        self.joints = []

        self._create_robot()  # 修复：方法名改为_create_robot
        self._create_obstacle()

        # 初始化渲染相关变量
        self.screen = None
        self.clock = None
        self.draw_options = None

        if self.render_mode:
            self._init_rendering()

    def _init_rendering(self):
        """初始化渲染相关组件"""
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 1200))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        

    def _create_robot(self):
        # anchor_point = (300, 300)

        prev_body = None
        density = 0.02  # 大幅减少密度，让link更轻
        
        for i in range(self.num_links):
            length = self.link_lengths[i]
            mass = density * length
            moment = pymunk.moment_for_segment(mass, (0, 0), (length, 0), 5)
            body = pymunk.Body(mass, moment)

            # 设置每个link的初始位置
            if i == 0:
                # 第一个link从anchor点开始
                body.position = self.anchor_point
            else:
                # 后续link连接到前一个link的末端
                prev_end = (prev_body.position[0] + self.link_lengths[i-1], prev_body.position[1])
                body.position = prev_end

            # 创建link的形状
            shape = pymunk.Segment(body, (0, 0), (length, 0), 5)
            shape.friction = 0.5

            self.space.add(body, shape)
            self.bodies.append(body)

            # 创建关节连接 - 使用PivotJoint而不是PinJoint
            if i == 0:
                # 第一个link固定到静态锚点
                joint = pymunk.PivotJoint(self.space.static_body, body, self.anchor_point, (0, 0))
                joint.collide_bodies = False  # 防止碰撞
                self.space.add(joint)
                self.joints.append(joint)
            else:
                # 后续link连接到前一个link的末端
                # 计算世界坐标中的连接点
                connection_point_world = (prev_body.position[0] + self.link_lengths[i-1], prev_body.position[1])
                joint = pymunk.PivotJoint(prev_body, body, (self.link_lengths[i-1], 0), (0, 0))
                joint.collide_bodies = False  # 防止碰撞
                self.space.add(joint)
                self.joints.append(joint)

            prev_body = body

        # 减少阻尼效果
        for body in self.bodies:
            body.velocity_func = self._apply_damping

    def _apply_damping(self, body, gravity, damping, dt):
        """应用轻微的阻尼力"""
        # 减少阻尼系数，让运动更明显
        body.velocity = body.velocity * 0.99  # 极小的线性阻尼
        body.angular_velocity = body.angular_velocity * 0.998  # 极小的角速度阻尼
        # 应用重力
        pymunk.Body.update_velocity(body, gravity, damping, dt)

    def reset(self, seed=None, options=None):  # 修复：添加正确的reset方法
        super().reset(seed=seed)
        self.space.remove(*self.space.bodies, *self.space.shapes, *self.space.constraints)
        self.bodies.clear()
        self.joints.clear()
        self.obstacles.clear()

        self._create_robot()
        self._create_obstacle()
        observation = self._get_observation()
        info = {}
        if self.gym_api_version == "old":
            return observation
        else:
            return observation, info

    def _get_observation(self):
        """获取当前状态观察值"""
        obs = []
        for body in self.bodies:
            obs.extend([body.angle, body.angular_velocity])
        
        # 计算末端执行器位置
        end_effector_pos = self._get_end_effector_position()
        obs.extend(end_effector_pos)
        
        # 🔧 添加目标信息
        obs.extend(self.goal_pos)  # 目标位置
        
        # 🔧 添加相对位置信息  
        relative_pos = np.array(self.goal_pos) - np.array(end_effector_pos)
        obs.extend(relative_pos)  # 到目标的相对位置
        
        # 🔧 添加距离信息
        distance = np.linalg.norm(relative_pos)
        obs.append(distance)  # 到目标的距离
        
        return np.array(obs, dtype=np.float32)

    def _get_end_effector_position(self):
        """计算末端执行器的位置"""
        if not self.bodies:
            return [0.0, 0.0]
        
        # 从第一个link的位置开始
        pos = np.array(self.bodies[0].position)
        current_angle = 0.0
        
        for i, body in enumerate(self.bodies):
            # 累积角度
            current_angle += body.angle
            length = self.link_lengths[i]
            
            # 计算这个link末端的位置
            if i == 0:
                # 第一个link从其起始位置延伸
                pos = np.array(self.bodies[0].position) + np.array([
                    length * np.cos(current_angle), 
                    length * np.sin(current_angle)
                ])
            else:
                # 后续link从前一个link的末端延伸
                pos += np.array([
                    length * np.cos(current_angle), 
                    length * np.sin(current_angle)
                ])
        
        return pos.tolist()
    
    def step(self, actions):
        """actions 是一个数组，控制每个关节的力矩"""
        actions = np.clip(actions, -self.max_torque, self.max_torque)
        
        for i, torque in enumerate(actions):
            if i < len(self.bodies):
                self.bodies[i].torque = torque

        self.space.step(self.dt)
        
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = False
        truncated = False
        info = {}

        if self.gym_api_version == "old":
            done = terminated or truncated
            return observation, reward, done, info
        else:
            return observation, reward, terminated, truncated, info

    def _compute_reward(self):
        end_effector_pos = np.array(self._get_end_effector_position())
        distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
        # 基础距离奖励
        distance_reward = -distance_to_goal / 100.0
        
        # 🔧 添加进步奖励
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = distance_to_goal
        
        progress = self.prev_distance - distance_to_goal  # 正值表示靠近
        progress_reward = progress * 10.0  # 放大进步奖励
        
        self.prev_distance = distance_to_goal
        
        return distance_reward + progress_reward
    
    def _load_config(self, config_path):
        if config_path is None:
            return {}
        
        # 如果是相对路径，将其转换为相对于当前脚本文件的路径
        if not os.path.isabs(config_path):
            # 获取当前脚本文件所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 将相对路径转换为基于脚本目录的绝对路径
            config_path = os.path.normpath(os.path.join(script_dir, "..", config_path))
        
        print(f"尝试加载配置文件: {config_path}")  # 调试用
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"错误：配置文件 {config_path} 不存在")
            return {}
        except Exception as e:
            print(f"错误：加载配置文件失败: {e}")
            return {}

    def _create_obstacle(self):
        if "obstacles" not in self.config:
            return
        
        for obs in self.config["obstacles"]:
            if obs["shape"] == "segment":
                p1 = tuple(obs["points"][0])
                p2 = tuple(obs["points"][1])
                shape = pymunk.Segment(self.space.static_body, p1, p2, 3.0)
                shape.friction = 1.0
                shape.color = (0,0,0,255)
                self.space.add(shape)
                self.obstacles.append(shape)

        if "goal" in self.config:
            self.goal_pos = np.array(self.config["goal"]["position"])
            self.goal_radius = self.config["goal"]["radius"]

   

    def render(self):
        if not self.render_mode:
            return
            
        self.screen.fill((255, 255, 255))
        
        # 绘制目标点
        pygame.draw.circle(self.screen, (255, 0, 0), self.goal_pos.astype(int), 10)
        
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)  # 控制渲染帧率

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()


    
if __name__ == "__main__":
    env = Reacher2DEnv(num_links=5, 
                       link_lengths=[80, 50, 30, 20, 50], 
                       render_mode="human",
                       config_path = "configs/reacher_with_zigzag_obstacles.yaml"
                       )

    running = True
    obs, info = env.reset()  # 修复：使用正确的reset调用
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
# 
        # 示例动作：三个关节施加不同大小的随机力矩
        actions = np.random.uniform(-5, 5, size=env.num_links)
        obs, reward, terminated, truncated, info = env.step(actions)  # 修复：接收step的返回值
        # print(obs, reward, terminated, truncated, info)
        env.render()

    env.close()
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
import logging

class Reacher2DEnv(Env):

    
    def __init__(self, num_links=3, link_lengths=None, render_mode=None, config_path=None, curriculum_stage=0, debug_level='SILENT'):

        super().__init__()
        self._set_logging(debug_level)
        self.config = self._load_config(config_path)
        self.logger.info(f"self.config: {self.config}")
        self.anchor_point = self.config["start"]["position"]
        self.gym_api_version = "old" # old or new. new is gymnasium, old is gym
        
        # 🎯 课程学习参数
        self.curriculum_stage = curriculum_stage
        self.base_goal_pos = np.array(self.config["goal"]["position"]) if "goal" in self.config else np.array([600, 575])
        print(f"🔍 [__init__] base_goal_pos from config: {self.base_goal_pos}")
        print(f"🔍 [__init__] anchor_point: {self.anchor_point}")
        print(f"🔍 [__init__] curriculum_stage: {curriculum_stage}")

        self.num_links = num_links  # 修复：使用传入的参数
        if link_lengths is None:

            self.link_lengths = [60] * num_links

        else:
            assert len(link_lengths) == num_links
            self.link_lengths = link_lengths
        
        self.render_mode = render_mode
        # self.goal_pos = np.array([250.0, 250.0])
        self.dt = 1/120.0  # 增加时间步长精度
        self.max_torque = 100  # 增加最大扭矩

        # 定义Gymnasium必需的action_space和observation_space
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(self.num_links,), dtype=np.float32)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 7,), dtype=np.float32)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 981.0)
        # 减少全局阻尼
        self.space.damping = 0.999  # 🔧 增加阻尼让角度限制更有效
        self.space.collision_slop = 0.1  # 🔧 减少全局碰撞容差，提高稳定性
        self.space.collision_bias = (1-0.1) ** 60
        self.space.sleep_time_threshold = 0.5
        
        # 🔧 增加物理系统稳定性设置 - 防止穿透和关节分离
        self.space.iterations = 30        # 🚨 大幅增加求解器迭代次数，确保约束收敛
        self.space.collision_persistence = 5  # 🚨 增加碰撞持续性，防止穿透
        
        # 🚨 关键设置：防止穿透的额外参数
        self.space.collision_bias = pow(1.0 - 0.01, 60)  # 更强的碰撞偏差修正
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

    def _set_logging(self, debug_level='INFO'):

        self.logger = logging.getLogger(f"Reacher2DEnv_{id(self)}")

        if not self.logger.handlers:

            level_map = {

                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
                 'SILENT': logging.CRITICAL + 10
            }

            env_level = os.getenv('REACHER_LOG_LEVEL',  debug_level).upper()
            log_level = level_map.get(env_level, logging.INFO)

            self.logger.setLevel(log_level)

            if env_level != 'SILENT' and log_level <= logging.CRITICAL:

                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)

                formatter = logging.Formatter('%(levelname)s [Reacher2D]: %(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            self.log_level = self.logger.level
            self.is_debug = self.log_level <= logging.DEBUG
            self.is_info = self.log_level <= logging.INFO
            self.is_warning = self.log_level <= logging.WARNING

            self.is_silent = env_level == 'SILENT'


    def _init_rendering(self):
        """初始化渲染相关组件"""
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 1200))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        

    def _create_robot(self):
        prev_body = None
        density = 0.8  # 🔧 增加密度，让约束更稳定
        self.joint_limits = []  # 🔧 存储角度限制约束
        self.motors = []  # 🔧 存储Motor控制器
        
        # 🔧 定义关节角度限制范围（弧度）
        # 基座关节无限制，其他关节有适当限制
        self.joint_angle_limits = [
            None,                                      # 第1个关节（基座）：无角度限制，可360°旋转
            (-math.pi * 2/3, math.pi * 2/3),          # 第2个关节：±120°
            (-math.pi * 2/3, math.pi * 2/3),          # 第3个关节：±120°
            (-math.pi * 2/3, math.pi * 2/3),          # 第4个关节：±120°
            (-math.pi * 2/3, math.pi * 2/3),          # 第5个关节：±120°
        ]
        
        # 🔧 从锚点开始构建机器人，每个link都有明确的位置
        current_pos = list(self.anchor_point)  # [x, y]
        
        for i in range(self.num_links):
            length = self.link_lengths[i]
            mass = density * length * 5  # 🔧 增加质量
            moment = pymunk.moment_for_segment(mass, (0, 0), (length, 0), 8)  # 🔧 增加半径
            body = pymunk.Body(mass, moment)
            
            # 🔧 设置初始位置（让机器人自然垂直下垂）
            body.position = current_pos
            body.angle = math.pi/2  # 改为45°避开边界
            
            # 🔧 创建形状 - 增加半径让碰撞更明显
            shape = pymunk.Segment(body, (0, 0), (length, 0), 10)  # 🚨 进一步增加半径到10，确保碰撞检测
            shape.friction = 1.0  # 🚨 最大摩擦力，防止滑动穿透
            shape.collision_type = i + 1  # 🔧 为每个link设置不同的碰撞类型
            shape.collision_slop = 0.01   # 🚨 减少碰撞容差，提高精度
            # 🚨 关键属性：防止穿透
            shape.elasticity = 0.0  # 无弹性，避免反弹导致的不稳定
            
            self.space.add(body, shape)
            self.bodies.append(body)
            
            # 创建关节连接和Motor
            if i == 0:
                # 🔧 基座关节：连接到世界 - 使用PivotJoint实现revolute关节
                joint = pymunk.PivotJoint(self.space.static_body, body, self.anchor_point, (0, 0))
                joint.collide_bodies = False
                # 🚨 关键修复：增强关节约束力，防止碰撞时分离
                joint.max_force = 1000000  # 非常大的约束力，确保关节不会分离
                self.space.add(joint)
                self.joints.append(joint)
                
                # 🔧 添加Motor控制器 - 控制关节运动
                motor = pymunk.SimpleMotor(self.space.static_body, body, 0.0)
                self.space.add(motor)
                self.motors.append(motor)
                
                # 🔧 添加角度限制约束 - 基座关节跳过角度限制
                if i < len(self.joint_angle_limits) and self.joint_angle_limits[i] is not None:
                    min_angle, max_angle = self.joint_angle_limits[i]
                    limit_joint = pymunk.RotaryLimitJoint(
                        self.space.static_body, body, 
                        min_angle, max_angle
                    )
                    # 🔧 增加约束的刚度
                    limit_joint.max_force = 50000   # 降低约束力，提高稳定性
                    self.space.add(limit_joint)
                    self.joint_limits.append(limit_joint)
                else:
                    # 基座关节无角度限制，添加None占位符
                    self.joint_limits.append(None)
                
            else:
                # 🔧 连接到前一个link的末端 - 使用PivotJoint实现revolute关节
                joint = pymunk.PivotJoint(prev_body, body, (self.link_lengths[i-1], 0), (0, 0))
                joint.collide_bodies = False
                # 🚨 关键修复：增强关节约束力，防止碰撞时分离
                joint.max_force = 1000000  # 非常大的约束力，确保关节不会分离
                self.space.add(joint)
                self.joints.append(joint)
                
                # 🔧 添加Motor控制器 - 控制关节运动
                motor = pymunk.SimpleMotor(prev_body, body, 0.0)
                self.space.add(motor)
                self.motors.append(motor)
                
                # 🔧 添加相对角度限制约束 - 物理约束防止过度旋转
                if i < len(self.joint_angle_limits) and self.joint_angle_limits[i] is not None:
                    min_angle, max_angle = self.joint_angle_limits[i]
                    limit_joint = pymunk.RotaryLimitJoint(
                        prev_body, body, 
                        min_angle, max_angle
                    )
                    # 🔧 增加约束的刚度
                    limit_joint.max_force = 50000   # 降低约束力，提高稳定性
                    self.space.add(limit_joint)
                    self.joint_limits.append(limit_joint)
                else:
                    # 无角度限制的关节，添加None占位符
                    self.joint_limits.append(None)
            
            # 🔧 计算下一个link的起始位置（用于初始化）
            if i < self.num_links - 1:
                end_x = current_pos[0] + length * math.cos(math.pi/2)  # 垂直向下
                end_y = current_pos[1] + length * math.sin(math.pi/2)
                current_pos = [end_x, end_y]
            
            prev_body = body
        
        # 🔧 添加关节间碰撞检测（可选 - 防止严重自碰撞）
        self._setup_collision_handlers()

    def _setup_collision_handlers(self):
        """设置碰撞处理器 - 改进版"""
        try:
            # 🎯 1. 改进的关节间碰撞处理 - 防止穿透但允许轻微接触
            def improved_joint_collision_handler(arbiter, space, data):
                """改进的关节间碰撞处理 - 防止穿透和关节分离"""
                # 🚨 强制设置碰撞属性，确保物理阻挡
                arbiter.restitution = 0.0   # 无弹性碰撞
                arbiter.friction = 1.0      # 🚨 最大摩擦力，防止滑动穿透
                
                # 获取碰撞深度 - 修复API错误
                contact_set = arbiter.contact_point_set
                if len(contact_set.points) > 0:
                    # 如果穿透太深，记录自碰撞并强制分离
                    max_depth = max(abs(p.distance) for p in contact_set.points)
                    if max_depth > 2.0:  # 🚨 降低阈值，更早检测穿透
                        if not hasattr(self, 'self_collision_count'):
                            self.self_collision_count = 0
                        self.self_collision_count += 1
                        self.logger.debug(f"🔴 严重自碰撞! 深度: {max_depth:.1f}px")
                        
                        # 🚨 强制修正穿透：增加分离冲量
                        for point in contact_set.points:
                            if abs(point.distance) > 2.0:
                                # 计算分离方向
                                normal = contact_set.normal
                                # 应用分离冲量
                                separation_impulse = normal * min(abs(point.distance) * 100, 1000)
                                arbiter.shapes[0].body.apply_impulse_at_world_point(-separation_impulse, point.point_a)
                                arbiter.shapes[1].body.apply_impulse_at_world_point(separation_impulse, point.point_b)
                
                return True  # 允许物理处理，但强化了碰撞响应
            
            # 关节间碰撞（改进逻辑）
            for i in range(self.num_links):
                for j in range(i + 2, self.num_links):  # 跳过相邻关节
                    try:
                        self.space.on_collision(
                            collision_type_a=i + 1, 
                            collision_type_b=j + 1,
                            pre_solve=improved_joint_collision_handler  # 使用pre_solve获得更多控制
                        )
                        self.logger.debug(f"✅ 设置改进关节碰撞: Link{i+1} vs Link{j+1}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 设置关节碰撞处理器失败: {e}")
            
            # 🎯 2. 新增：机器人与障碍物碰撞处理 - 使用正确API和begin回调
            def robot_obstacle_collision_handler(arbiter, space, data):
                """处理机器人与障碍物的碰撞 - 防止穿透"""
                # 🚨 强制设置碰撞属性，确保物理阻挡
                arbiter.restitution = 0.0   # 无弹性碰撞，避免反弹
                arbiter.friction = 1.0      # 🚨 最大摩擦力，防止滑动穿透
                
                # 记录碰撞信息
                if not hasattr(self, 'collision_count'):
                    self.collision_count = 0
                self.collision_count += 1
                
                self.logger.debug(f"🚨 检测到机器人-障碍物碰撞! 总计: {self.collision_count}")
                
                # 🚨 检查穿透深度并强制修正
                contact_set = arbiter.contact_point_set
                if len(contact_set.points) > 0:
                    max_depth = max(abs(p.distance) for p in contact_set.points)
                    if max_depth > 1.0:  # 检测到穿透
                        self.logger.debug(f"🔴 障碍物穿透! 深度: {max_depth:.1f}px")
                        
                        # 强制分离：应用分离冲量
                        for point in contact_set.points:
                            if abs(point.distance) > 1.0:
                                normal = contact_set.normal
                                separation_impulse = normal * min(abs(point.distance) * 200, 2000)
                                # 只对机器人施加分离冲量（障碍物是静态的）
                                if arbiter.shapes[0].body.body_type == pymunk.Body.DYNAMIC:
                                    arbiter.shapes[0].body.apply_impulse_at_world_point(-separation_impulse, point.point_a)
                                if arbiter.shapes[1].body.body_type == pymunk.Body.DYNAMIC:
                                    arbiter.shapes[1].body.apply_impulse_at_world_point(separation_impulse, point.point_b)
                
                return True  # 允许物理碰撞处理
            
            # 为每个机器人链接设置与障碍物的碰撞检测
            OBSTACLE_COLLISION_TYPE = 100
            for i in range(self.num_links):
                robot_link_type = i + 1
                try:
                    # 🔧 使用PyMunk 7.1.0的正确API和begin回调
                    self.space.on_collision(
                        collision_type_a=robot_link_type, 
                        collision_type_b=OBSTACLE_COLLISION_TYPE,
                        begin=robot_obstacle_collision_handler  # 改为begin回调
                    )
                    self.logger.debug(f"✅ 设置机器人链接{i+1}与障碍物的碰撞检测")
                except Exception as e:
                    self.logger.warning(f"⚠️ 设置机器人-障碍物碰撞处理器失败: {e}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 碰撞处理器设置跳过: {e}")

    def _apply_damping(self, body, gravity, damping, dt):
        """应用轻微的阻尼力"""
        # 🔧 增加阻尼，特别是角速度阻尼
        body.velocity = body.velocity * 0.995  # 增加线性阻尼
        body.angular_velocity = body.angular_velocity * 0.99  # 增加角速度阻尼
        # 应用重力
        pymunk.Body.update_velocity(body, gravity, damping, dt)

    def reset(self, seed=None, options=None):  # 修复：添加正确的reset方法
        super().reset(seed=seed)
        self.space.remove(*self.space.bodies, *self.space.shapes, *self.space.constraints)
        self.bodies.clear()
        self.joints.clear()
        self.obstacles.clear()
        
        # 🔧 清理角度限制约束
        if hasattr(self, 'joint_limits'):
            self.joint_limits.clear()
        if hasattr(self, 'motors'):
            self.motors.clear()

        self._create_robot()
        self._create_obstacle()

        print(f"\n🔍 [reset] 开始 - 检查goal设置:")
        print(f"  config中的goal: {self.config.get('goal', {}).get('position', 'NOT FOUND')}")
        print(f"  base_goal_pos: {self.base_goal_pos}")
        print(f"  当前goal_pos: {getattr(self, 'goal_pos', 'NOT SET YET')}")

        if "goal" in self.config:
            self.goal_pos = np.array(self.config["goal"]["position"])
            print(f"🎯 [reset] 设置goal_pos from config: {self.goal_pos}")
        else:
            self.goal_pos = np.array([150, 575])  # 后备目标
            print(f"🎯 [reset] 设置goal_pos 后备默认值: {self.goal_pos}")

        print(f"🔍 [reset] 最终goal_pos: {self.goal_pos}")

        # 🔍 添加机器人状态调试
        if self.bodies:
            print(f"🤖 [reset] 机器人状态调试:")
            print(f"  机器人link数量: {len(self.bodies)}")
            print(f"  Link长度配置: {self.link_lengths}")
            for i, body in enumerate(self.bodies):
                print(f"  Link {i}: position={body.position}, angle={math.degrees(body.angle):.1f}°")
            
            end_pos = self._get_end_effector_position()
            print(f"  末端执行器位置: {end_pos}")
            print(f"  起始位置 -> 末端位置: {self.anchor_point} -> {end_pos}")
            
            distance = np.linalg.norm(np.array(end_pos) - self.goal_pos)
            print(f"  到目标距离: {distance:.1f} pixels")
            print(f"  需要移动方向: {np.array(self.goal_pos) - np.array(end_pos)}")
                
        # 初始化计数器
        self.step_counter = 0
        if not hasattr(self, 'collision_count'):
            self.collision_count = 0
        if not hasattr(self, 'episode_start_collisions'):
            self.episode_start_collisions = self.collision_count
        if not hasattr(self, 'prev_collision_count'):
            self.prev_collision_count = 0

        observation = self._get_observation()
        info = self._build_info_dict()
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
        """计算末端执行器位置"""
        if not self.bodies:
            return [0.0, 0.0]
        
        # 从anchor_point逐步构建
        pos = np.array(self.anchor_point, dtype=float)
        for i, body in enumerate(self.bodies):
            length = self.link_lengths[i]
            link_vector = np.array([
                length * np.cos(body.angle),
                length * np.sin(body.angle)
            ])
            pos += link_vector
        
        return pos.tolist()
    
    def step(self, actions):
        """使用Motor控制 + 物理约束，结合真实性和安全性"""
        actions = np.clip(actions, -self.max_torque, self.max_torque)
        
        # 🔧 将扭矩转换为角速度目标，通过Motor控制
        # 简单的比例控制：扭矩 → 角速度
        torque_to_speed_ratio = 0.05  # 🚀 增加响应性：从0.01提升到0.05
        
        for i, torque in enumerate(actions):
            if i < len(self.motors):
                motor = self.motors[i]
                # 将扭矩转换为目标角速度
                target_angular_velocity = torque * torque_to_speed_ratio
                motor.rate = float(target_angular_velocity)

        # 🔧 让物理约束自动处理角度限制，无需手动干预
        self.space.step(self.dt)
        
        # 🧪 减少输出频率
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        self.step_counter += 1

        end_effector_pos = self._get_end_effector_position()
        distance_to_goal = np.linalg.norm(np.array(end_effector_pos) - self.goal_pos)
        terminated = distance_to_goal <= 35.0

        observation = self._get_observation()
        reward = self._compute_reward()
        truncated = False
        info = self._build_info_dict()

        if self.gym_api_version == "old":
            done = terminated or truncated
            return observation, reward, done, info
        else:
            return observation, reward, terminated, truncated, info
    
    def _build_info_dict(self):
        """构建包含丰富信息的info字典"""
        info = {}
        
        # 🎯 碰撞相关信息
        info['collisions'] = {
            'total_count': getattr(self, 'collision_count', 0),
        }
        
        # 🎯 目标相关信息
        end_effector_pos = self._get_end_effector_position()
        distance_to_goal = np.linalg.norm(np.array(end_effector_pos) - self.goal_pos)
        
        info['goal'] = {
            'distance_to_goal': float(distance_to_goal),
            'end_effector_position': end_effector_pos,
            'goal_position': self.goal_pos.tolist(),
            'goal_reached': distance_to_goal <= 50.0
        }
        
        return info

    def _compute_reward(self):
        """基础奖励函数 - 简单稳定的奖励机制"""
        end_effector_pos = np.array(self._get_end_effector_position())
        distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
        # 基础距离奖励
        max_distance = 300.0  
        distance_reward = -distance_to_goal / max_distance * 2.0
        
        # 成功奖励
        if distance_to_goal <= 35.0:
            success_reward = 5.0
        else:
            success_reward = 0.0
        
        # 进步奖励
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = distance_to_goal
        
        progress = self.prev_distance - distance_to_goal
        progress_reward = np.clip(progress * 3.0, -1.0, 1.0)
        self.prev_distance = distance_to_goal
        
        # 碰撞惩罚
        collision_penalty = 0.0
        if hasattr(self, 'collision_count'):
            if not hasattr(self, 'prev_collision_count'):
                self.prev_collision_count = 0
            new_collisions = self.collision_count - self.prev_collision_count
            if new_collisions > 0:
                collision_penalty = -0.5 * new_collisions
            self.prev_collision_count = self.collision_count
        
        total_reward = distance_reward + success_reward + progress_reward + collision_penalty
        total_reward = np.clip(total_reward, -5.0, 8.0)
        
        return total_reward

    def _load_config(self, config_path):
        if config_path is None:
            return {}
        
        # 如果是相对路径，将其转换为相对于当前脚本文件的路径
        if not os.path.isabs(config_path):
            # 获取当前脚本文件所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 将相对路径转换为基于脚本目录的绝对路径
            config_path = os.path.normpath(os.path.join(script_dir, "..", config_path))
        
        self.logger.debug(f"尝试加载配置文件: {config_path}")  # 调试用

        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"错误：配置文件 {config_path} 不存在")
            return {}
        except Exception as e:
            self.logger.error(f"错误：加载配置文件失败: {e}")
            return {}

    def _create_obstacle(self):
        if "obstacles" not in self.config:
            return
        
        # 🎯 定义障碍物的collision_type
        OBSTACLE_COLLISION_TYPE = 100  # 使用大数字避免与机器人冲突
        
        for obs in self.config["obstacles"]:
            if obs["shape"] == "segment":
                p1 = tuple(obs["points"][0])
                p2 = tuple(obs["points"][1])
                shape = pymunk.Segment(self.space.static_body, p1, p2, radius=5.0)
                shape.friction = 1.0
                shape.color = (0,0,0,255)
                shape.density = 1000
                
                # 🎯 关键添加：设置障碍物碰撞类型
                shape.collision_type = OBSTACLE_COLLISION_TYPE
                shape.collision_slop = 0.01  # 🔧 设置障碍物碰撞容差，与links一致
                
                self.space.add(shape)
                self.obstacles.append(shape)

        if "goal" in self.config:
            print(f"🔍 [_create_obstacle] 准备设置goal_pos: {self.config['goal']['position']}")
            self.goal_pos = np.array(self.config["goal"]["position"])
            self.goal_radius = self.config["goal"]["radius"]
            print(f"🎯 [_create_obstacle] 已设置goal_pos: {self.goal_pos}")
        else:
            print(f"❌ [_create_obstacle] config中没有goal配置")

    def render(self):
        if not self.render_mode:
            return
            
        self.screen.fill((255, 255, 255))
        
        # 绘制原始目标点 - 绿色大圆让它更明显
        goal_pos_int = self.goal_pos.astype(int)
        pygame.draw.circle(self.screen, (0, 255, 0), goal_pos_int, 15)  # 绿色大圆
        pygame.draw.circle(self.screen, (0, 0, 0), goal_pos_int, 15, 3)  # 黑色边框
        print(f"🎯 [render] 绘制目标点在: {goal_pos_int}")

        end_effector_pos = self._get_end_effector_position()
        print(f"🔍 [render] end_effector_pos: {end_effector_pos}")
        if end_effector_pos:
            # 绘制蓝色圆点标记end_effector位置
            pos_int = (int(end_effector_pos[0]), int(end_effector_pos[1]))
            pygame.draw.circle(self.screen, (0, 0, 255), pos_int, 8)  # 蓝色圆点，半径8
            
            # 绘制一个白色边框让蓝点更显眼
            pygame.draw.circle(self.screen, (255, 255, 255), pos_int, 8, 2)  # 白色边框
            print(f"🤖 [render] 绘制末端执行器在: {pos_int}")
        
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)

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
    obs= env.reset()  # 修复：使用正确的reset调用
    step_count = 0
    
    while running and step_count < 3000:  # 增加到300步测试更严格的限制
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 🧪 使用极大的力矩测试新的角度限制系统
        actions = np.random.uniform(-500, 500, size=env.num_links)  # 使用更大的力矩测试
        obs, reward, terminated, truncated= env.step(actions)
        env.render()
        step_count += 1

    env.close()

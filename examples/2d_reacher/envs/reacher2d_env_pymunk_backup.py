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
        
        # 🎯 课程学习参数 - 恢复原始简洁设计
        
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
        self.dt = 1/120.0  # 恢复原始时间步长
        self.max_torque = 100  # 增加最大扭矩

        # 定义Gymnasium必需的action_space和observation_space
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(self.num_links,), dtype=np.float32)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 7,), dtype=np.float32)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 981.0)  # 恢复原始重力
        # 减少全局阻尼
        self.space.damping = 0.999  # 恢复原始阻尼
        self.space.collision_slop = 0.5  # 恢复原始碰撞容差
        self.space.collision_bias = (1-0.1) ** 60  # 恢复原始碰撞偏差
        self.space.sleep_time_threshold = 0.5
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
        density = 0.8
        
        # 🎯 关键修复：创建虚拟固定基座Body代替static_body连接
        # 这解决了基座关节无法与障碍物碰撞的问题
        # 🔧 使用STATIC body类型，确保完全固定
        self.base_anchor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.base_anchor_body.position = self.anchor_point
        
        # 创建基座锚点形状（不可见，只用于物理）
        base_anchor_shape = pymunk.Circle(self.base_anchor_body, 5)
        base_anchor_shape.collision_type = 999  # 特殊碰撞类型，不与任何东西碰撞
        base_anchor_shape.sensor = True  # 设为传感器，不产生物理碰撞
        
        self.space.add(self.base_anchor_body, base_anchor_shape)
        
        # 🎯 静态body不需要额外的约束，它本身就是固定的
        
        self.logger.debug(f"✅ [修复] 创建虚拟基座锚点: {self.anchor_point}")  # 🔧 增加密度，让约束更稳定
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
            
            # 🔧 **关键修复**：为每个Body增加阻尼，提高数值稳定性
            body.velocity_func = lambda body, gravity, damping, dt: (
                body.velocity * 0.98,  # 线性速度阻尼
                body.angular_velocity * 0.98  # 角速度阻尼
            )
            
            # 🔧 设置初始位置（让机器人自然垂直下垂）
            body.position = current_pos
            body.angle = math.pi/2  # 改为45°避开边界
            
            # 🔧 创建形状 - 增加半径让碰撞更明显
            shape = pymunk.Segment(body, (0, 0), (length, 0), 8)  # 半径从5增加到8
            shape.friction = 0.8  # 🔧 增加摩擦力
            shape.collision_type = i + 1  # 🔧 为每个link设置不同的碰撞类型
            shape.collision_slop = 0.1   # 🔧 增加碰撞容差，避免穿透导致的约束冲突
            
            # 🔧 添加身体级别的阻尼，进一步稳定系统
            body.velocity_func = self._apply_body_damping
            
            self.space.add(body, shape)
            self.bodies.append(body)
            
            # 创建关节连接和Motor
            if i == 0:
                # 🔧 基座关节：直接连接到static_body，确保绝对固定
                joint = pymunk.PivotJoint(self.space.static_body, body, self.anchor_point, (0, 0))
                joint.collide_bodies = False  # 不让基座关节与static_body碰撞
                # 🔧 **关键修复**：设置合理的约束力，避免过度约束导致系统不稳定
                joint.max_force = 50000   # 🔧 大幅降低约束力：从1000000降到50000
                self.space.add(joint)
                self.joints.append(joint)
                
                # 🔧 添加Motor控制器 - 连接到static_body
                motor = pymunk.SimpleMotor(self.space.static_body, body, 0.0)
                motor.max_force = 5000   # 🔧 大幅降低Motor力度，避免手动控制时产生巨大冲击
                self.space.add(motor)
                self.motors.append(motor)
                
                # 🔧 **实验性修复**：完全移除角度限制约束，避免约束冲突
                # 让RL算法自己学会合理的关节角度
                print(f"   🚫 跳过Link{i}的角度限制约束")
                self.joint_limits.append(None)
                
            else:
                # 🔧 连接到前一个link的末端 - 使用PivotJoint实现revolute关节
                joint = pymunk.PivotJoint(prev_body, body, (self.link_lengths[i-1], 0), (0, 0))
                joint.collide_bodies = False
                # 🔧 **关键修复**：设置合理的约束力，避免过度约束导致系统不稳定
                joint.max_force = 50000   # 🔧 大幅降低约束力：从1000000降到50000
                self.space.add(joint)
                self.joints.append(joint)
                
                # 🔧 添加Motor控制器 - 控制关节运动
                motor = pymunk.SimpleMotor(prev_body, body, 0.0)
                motor.max_force = 3000   # 🔧 设置较小的Motor力度，避免手动控制时产生冲击
                self.space.add(motor)
                self.motors.append(motor)
                
                # 🔧 **实验性修复**：完全移除角度限制约束，避免约束冲突
                # 让RL算法自己学会合理的关节角度
                print(f"   🚫 跳过Link{i}的相对角度限制约束")
                self.joint_limits.append(None)
            
            # 🔧 计算下一个link的起始位置（用于初始化）
            if i < self.num_links - 1:
                end_x = current_pos[0] + length * math.cos(math.pi/4)  # 45度角
                end_y = current_pos[1] + length * math.sin(math.pi/4)
                current_pos = [end_x, end_y]
            
            prev_body = body
        
        # 🔧 添加关节间碰撞检测（可选 - 防止严重自碰撞）
        self._setup_collision_handlers()

    def _setup_collision_handlers(self):
        """设置碰撞处理器 - 防炸开版本"""
        try:
            # 🛡️ 1. 简化的Link间碰撞处理 - 仅设置物理参数
            def simple_link_collision_handler(arbiter, space, data):
                """简化的Link间碰撞处理 - 避免与约束求解器冲突"""
                
                # 🔧 只设置物理参数，不手动施加冲量
                arbiter.restitution = 0.01  # 几乎无弹性
                arbiter.friction = 0.9      # 高摩擦力
                
                return True  # 让PyMunk内部求解器处理
            
            # 🎯 2. 为所有非相邻Link对设置温和碰撞处理
            # 相邻Link通过joint.collide_bodies = False已经禁止碰撞
            for i in range(self.num_links):
                for j in range(i + 2, self.num_links):  # 跳过相邻Link，它们不应该碰撞
                    try:
                        self.space.on_collision(
                            collision_type_a=i + 1,
                            collision_type_b=j + 1,
                            begin=simple_link_collision_handler
                        )
                        self.logger.debug(f"✅ 设置温和碰撞处理: Link{i+1} vs Link{j+1}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 设置温和碰撞处理器失败: {e}")
            
            # 保持原有的机器人与障碍物碰撞处理
            def joint_collision_handler(arbiter, space, data):
                return True  # 允许关节间碰撞处理
            
            # 关节间碰撞（现有逻辑）
            for i in range(self.num_links):
                for j in range(i + 2, self.num_links):  # 跳过相邻关节
                    try:
                        # 🔧 使用PyMunk 7.1.0的正确API和begin回调
                        self.space.on_collision(
                            collision_type_a=i + 1, 
                            collision_type_b=j + 1,
                            begin=joint_collision_handler  # 改为begin回调
                        )
                        self.logger.debug(f"✅ 设置关节{i+1}与关节{j+1}的碰撞检测")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 设置关节碰撞处理器失败: {e}")
            
            # 🎯 2. 新增：机器人与障碍物碰撞处理 - 使用正确API和begin回调
            def robot_obstacle_collision_handler(arbiter, space, data):
                """处理机器人与障碍物的碰撞"""
                # 记录碰撞信息
                if not hasattr(self, 'collision_count'):
                    self.collision_count = 0
                self.collision_count += 1
                
                # 获取碰撞的形状
                shape_a, shape_b = arbiter.shapes
                link_collision_type = shape_a.collision_type if shape_a.collision_type != 100 else shape_b.collision_type
                
                self.logger.debug(f"🚨 检测到机器人Link{link_collision_type}-障碍物碰撞! 总计: {self.collision_count}")
                
                # 🔧 特殊处理基座关节碰撞
                if link_collision_type == 1:  # 基座关节
                    self.logger.info(f"🎯 基座关节碰撞障碍物!")
                    # 设置更强的碰撞响应
                    arbiter.restitution = 0.2  # 适中弹性
                    arbiter.friction = 1.5     # 高摩擦
                else:
                    # 其他Link的正常碰撞处理
                    arbiter.restitution = 0.1  # 低弹性
                    arbiter.friction = 0.9     # 正常摩擦
                
                return True  # 允许物理碰撞，提供真实反馈
            
            # 🔧 简化的基座关节碰撞处理器
            def simple_base_collision_handler(arbiter, space, data):
                """简化的基座关节与障碍物碰撞处理"""
                # 设置强碰撞响应，但不手动施加冲量
                arbiter.restitution = 0.0  # 无弹性
                arbiter.friction = 1.0     # 高摩擦，防止滑动
                
                return True
            
            # 为每个机器人链接设置与障碍物的碰撞检测
            OBSTACLE_COLLISION_TYPE = 100
            for i in range(self.num_links):
                robot_link_type = i + 1
                try:
                    if i == 0:  # 基座关节特殊处理
                        # 🎯 基座关节使用专用处理器
                        self.space.on_collision(
                            collision_type_a=robot_link_type, 
                            collision_type_b=OBSTACLE_COLLISION_TYPE,
                            begin=simple_base_collision_handler
                        )
                        self.logger.debug(f"✅ [专用] 设置基座关节与障碍物的碰撞检测")
                    else:
                        # 其他Link使用通用处理器
                        self.space.on_collision(
                            collision_type_a=robot_link_type, 
                            collision_type_b=OBSTACLE_COLLISION_TYPE,
                            begin=robot_obstacle_collision_handler
                        )
                        self.logger.debug(f"✅ 设置机器人链接{i+1}与障碍物的碰撞检测")
                except Exception as e:
                    self.logger.warning(f"⚠️ 设置机器人-障碍物碰撞处理器失败: {e}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 碰撞处理器设置跳过: {e}")

    def _apply_body_damping(self, body, gravity, damping, dt):
        """应用身体级别的阻尼力 - 防止速度爆炸"""
        # 🔧 更强的阻尼，防止手动控制时的速度爆炸
        current_vel = np.array(body.velocity)
        current_ang_vel = body.angular_velocity
        
        # 限制最大速度
        max_vel = 100.0
        max_ang_vel = 5.0
        
        if np.linalg.norm(current_vel) > max_vel:
            vel_direction = current_vel / (np.linalg.norm(current_vel) + 1e-6)
            body.velocity = (vel_direction * max_vel).tolist()
        
        if abs(current_ang_vel) > max_ang_vel:
            body.angular_velocity = np.sign(current_ang_vel) * max_ang_vel
        
        # 应用阻尼
        body.velocity = (np.array(body.velocity) * 0.98).tolist()  # 更强的线性阻尼
        body.angular_velocity = body.angular_velocity * 0.95      # 更强的角速度阻尼
        
        # 调用原始的速度更新
        pymunk.Body.update_velocity(body, gravity, damping, dt)

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
        
        # 🎯 课程学习：根据阶段调整目标位置
        # if hasattr(self, 'curriculum_stage'):
        #     if self.curriculum_stage == 0:
        #         # 阶段0：目标很近，容易达到
        #         self.goal_pos = self.base_goal_pos * 0.7 + np.array(self.anchor_point) * 0.3
        #     elif self.curriculum_stage == 1:
        #         # 阶段1：中等距离
        #         self.goal_pos = self.base_goal_pos * 0.85 + np.array(self.anchor_point) * 0.15
        #     else:
        #         # 阶段2+：完整难度
        #         self.goal_pos = self.base_goal_pos

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

        # 🔄 禁用路标点系统
        # self._reset_waypoint_system()

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
        """调试版本 - 对比多种计算方法"""
        if not self.bodies:
            return [0.0, 0.0]
        
        # 方法A：从anchor_point逐步构建
        pos_A = np.array(self.anchor_point, dtype=float)
        for i, body in enumerate(self.bodies):
            length = self.link_lengths[i]
            link_vector = np.array([
                length * np.cos(body.angle),
                length * np.sin(body.angle)
            ])
            pos_A += link_vector
        
        # 方法B：基于最后一个body的末端
        last_body = self.bodies[-1]
        last_length = self.link_lengths[-1]
        body_center = np.array(last_body.position)
        body_angle = last_body.angle
        end_offset = np.array([
            last_length/2 * np.cos(body_angle),
            last_length/2 * np.sin(body_angle)
        ])
        pos_B = body_center + end_offset
        
        # 方法C：你的原始方法
        pos_C = np.array(self.bodies[0].position)
        current_angle = 0.0
        for i, body in enumerate(self.bodies):
            current_angle += body.angle
            length = self.link_lengths[i]
            if i == 0:
                pos_C = np.array(self.bodies[0].position) + np.array([
                    length * np.cos(current_angle), 
                    length * np.sin(current_angle)
                ])
            else:
                pos_C += np.array([
                    length * np.cos(current_angle), 
                    length * np.sin(current_angle)
                ])
        
        # 打印对比（只在step 0, 50, 100...时打印）
        step_count = getattr(self, '_debug_step_count', 0)
        # if step_count % 50 == 0:
        #     print(f"🔍 End Effector 位置对比 (Step {step_count}):")
        #     print(f"  方法A (anchor+逐步): {pos_A}")
        #     print(f"  方法B (最后body末端): {pos_B}")
        #     print(f"  方法C (原始累积): {pos_C}")
        #     print(f"  A-B差异: {np.linalg.norm(pos_A - pos_B):.1f}")
        #     print(f"  A-C差异: {np.linalg.norm(pos_A - pos_C):.1f}")
        #     print(f"  B-C差异: {np.linalg.norm(pos_B - pos_C):.1f}")
        
        self._debug_step_count = step_count + 1
        
        # 返回最可能正确的方法A
        return pos_A.tolist()
    
    
    def step(self, actions):

        # 在step方法开始添加
        # if hasattr(self, 'step_counter') and self.step_counter % 50 == 0:
        #     print(f"🎯 [step] Step {self.step_counter}:")
        #     print(f"  输入动作: {actions}")
        #     print(f"  最大扭矩限制: {self.max_torque}")
        #     print(f"  动作空间: {self.action_space}")
        
        """使用Motor控制 + 物理约束，结合真实性和安全性 + 防炸开"""
        actions = np.clip(actions, -self.max_torque, self.max_torque)
        
        # 🛡️ 在step前记录速度
        pre_step_velocities = []
        if self.explosion_detection:
            for body in self.bodies:
                pre_step_velocities.append({
                    'velocity': np.array(body.velocity),
                    'angular_velocity': body.angular_velocity
                })
        
        # 🔧 将扭矩转换为角速度目标，通过Motor控制
        # 简单的比例控制：扭矩 → 角速度
        torque_to_speed_ratio = 0.5   # 🔧 大幅提高响应性：从0.01增加到0.5
        
        for i, torque in enumerate(actions):
            if i < len(self.motors):
                motor = self.motors[i]
                # 将扭矩转换为目标角速度
                target_angular_velocity = torque * torque_to_speed_ratio
                motor.rate = float(target_angular_velocity)

        # 🔧 让物理约束自动处理角度限制，无需手动干预
        self.space.step(self.dt)
        
        # 🛡️ 炸开检测和修正
        if self.explosion_detection and pre_step_velocities:
            self._detect_and_fix_explosion(pre_step_velocities)
        
        # 🧪 减少输出频率
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        self.step_counter += 1
        
        if self.step_counter % 20 == 0:  # 每20步打印一次
            self._print_motor_status()


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
        
    def _get_collision_rate(self):
        """计算碰撞率"""
        if hasattr(self, 'collision_count') and hasattr(self, 'step_counter'):
            if self.step_counter > 0:
                return float(self.collision_count) / float(self.step_counter)
        return 0.0
    
    def _get_episode_collisions(self):
        """获取本episode的碰撞次数"""
        if not hasattr(self, 'episode_start_collisions'):
            self.episode_start_collisions = getattr(self, 'collision_count', 0)
        
        current_total = getattr(self, 'collision_count', 0)
        return current_total - self.episode_start_collisions
    
    def _get_collision_penalty(self):
        """获取当前的碰撞惩罚值"""
        if hasattr(self, 'collision_count'):
            current_collisions = self.collision_count
            if not hasattr(self, 'prev_collision_count'):
                self.prev_collision_count = 0
            
            new_collisions = current_collisions - self.prev_collision_count
            return -new_collisions * 10.0  # 每次新碰撞扣10分
        return 0.0
    
    def _get_joint_distance_penalty(self):
        """计算关节间距离惩罚 - 防止关节过度靠近"""
        if not hasattr(self, 'bodies') or len(self.bodies) < 2:
            return 0.0
        
        penalty = 0.0
        min_safe_distance = 25.0  # 关节间最小安全距离（像素）
        max_penalty_per_pair = 0.2  # 每对关节的最大惩罚
        
        # 检查所有关节对
        for i in range(len(self.bodies)):
            for j in range(i + 2, len(self.bodies)):  # 跳过相邻关节，只检查间隔关节
                pos_i = np.array(self.bodies[i].position)
                pos_j = np.array(self.bodies[j].position)
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < min_safe_distance:
                    # 距离越近，惩罚越大
                    violation = min_safe_distance - distance
                    pair_penalty = (violation / min_safe_distance) * max_penalty_per_pair
                    penalty += pair_penalty
                    
                    # 调试信息（仅在严重违规时输出）
                    if hasattr(self, 'step_counter') and self.step_counter % 200 == 0 and violation > 10:
                        print(f"⚠️ 关节{i}-{j}过近: {distance:.1f}px (安全距离:{min_safe_distance}px), 惩罚:{pair_penalty:.3f}")
        
        # 限制总惩罚范围
        penalty = np.clip(penalty, 0.0, 1.0)
        return -penalty  # 返回负值作为惩罚
    
    def _build_info_dict(self):
        """构建包含丰富信息的info字典"""
        info = {}
        
        # 🎯 碰撞相关信息
        info['collisions'] = {
            'total_count': getattr(self, 'collision_count', 0),
            'collision_rate': self._get_collision_rate(),
            'collisions_this_episode': self._get_episode_collisions(),
            'collision_penalty': self._get_collision_penalty()
        }
        
        # 🎯 目标相关信息
        end_effector_pos = self._get_end_effector_position()
        distance_to_goal = np.linalg.norm(np.array(end_effector_pos) - self.goal_pos)
        
        info['goal'] = {
            'distance_to_goal': float(distance_to_goal),
            'end_effector_position': end_effector_pos,
            'goal_position': self.goal_pos.tolist(),
            'goal_reached': distance_to_goal <= 50.0  # 使用相同的阈值
        }
        
        # 🎯 机器人状态信息
        info['robot'] = {
            'joint_angles_deg': [math.degrees(body.angle) for body in self.bodies],
            'joint_velocities': [body.angular_velocity for body in self.bodies],
            'step_count': self.step_counter
        }
        
        # 🎯 奖励分解信息（用于调试）
        if hasattr(self, 'prev_distance'):
            progress = self.prev_distance - distance_to_goal
            info['reward_breakdown'] = {
                'distance_reward': -distance_to_goal / 30.0,
                'progress_reward': progress * 50.0,
                'success_bonus': 500.0 if distance_to_goal <= 10.0 else (300.0 if distance_to_goal <= 25.0 else (150.0 if distance_to_goal <= 50.0 else (50.0 if distance_to_goal <= 100.0 else 0.0))),
                'collision_penalty': self._get_collision_penalty(),
                'obstacle_avoidance': self._compute_obstacle_avoidance_reward()
            }
        
        return info
    def _print_motor_status(self):
        """打印Motor和物理约束状态信息"""
        # 计算绝对角度和相对角度
        absolute_angles = [math.degrees(body.angle) for body in self.bodies]
        relative_angles = []
        
        for i, body in enumerate(self.bodies):
            if i == 0:
                # 基座关节：相对角度 = 绝对角度
                relative_angles.append(absolute_angles[i])
            else:
                # 其他关节：相对角度 = 当前body角度 - 前一个body角度
                relative_angle = absolute_angles[i] - absolute_angles[i-1]
                # 标准化到[-180°, 180°]
                while relative_angle > 180:
                    relative_angle -= 360
                while relative_angle < -180:
                    relative_angle += 360
                relative_angles.append(relative_angle)
        
        self.logger.debug(f"步骤 {self.step_counter:4d} - 绝对角度: {[f'{a:7.1f}°' for a in absolute_angles]}")
        self.logger.debug(f"              相对角度: {[f'{a:7.1f}°' for a in relative_angles]}")
        
        # 打印Motor状态
        motor_rates = [motor.rate for motor in self.motors]
        self.logger.debug(f"    Motor角速度: {[f'{r:6.2f}' for r in motor_rates]} rad/s")
        
        # 检查约束是否还存在
        active_constraints = [c for c in self.joint_limits if c is not None]
        constraints_count = len([c for c in self.space.constraints if hasattr(c, 'min')])
        motors_count = len([c for c in self.space.constraints if isinstance(c, pymunk.SimpleMotor)])
        self.logger.debug(f"    约束数量: {constraints_count}/{len(active_constraints)} 角度限制, {motors_count}/{len(self.motors)} Motors")
        
        # 检查相对角度是否超出限制
        limit_degrees = [None, (-120, 120), (-120, 120), (-120, 120), (-120, 120)]  # 基座无限制
        violations = []
        for i, (rel_angle, limits) in enumerate(zip(relative_angles, limit_degrees)):
            if limits is not None:  # 跳过无限制的关节
                min_limit, max_limit = limits
                if rel_angle < min_limit or rel_angle > max_limit:
                    violations.append(f"关节{i+1}相对角度超限: {rel_angle:.1f}°")
        
        if violations:
            self.logger.warning(f"    ⚠️  角度超限: {', '.join(violations)} (物理约束应该防止这种情况)")
        else:
            if len(active_constraints) > 0:
                self.logger.debug(f"    ✅ 所有受限关节相对角度在范围内 (基座关节无限制)")
            else:   
                self.logger.debug(f"    ✅ 所有关节正常运行")

    def get_joint_angles(self):
        """获取所有关节的当前角度（度数）"""
        return [math.degrees(body.angle) for body in self.bodies]

    # def _compute_reward(self):
    #     """超稳定奖励函数 - 防止数值爆炸"""
    #     end_effector_pos = np.array(self._get_end_effector_position())
    #     distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
    #     # === 1. 距离奖励 - 使用tanh防止极值 ===
    #     distance_reward = -np.tanh(distance_to_goal / 100.0) * 2.0  # 范围: -2.0 到 0
        
    #     # === 2. 进步奖励 - 严格限制范围 ===
    #     if not hasattr(self, 'prev_distance'):
    #         self.prev_distance = distance_to_goal
        
    #     progress = self.prev_distance - distance_to_goal
    #     progress_reward = np.clip(progress * 5.0, -1.0, 1.0)  # 严格限制在[-1,1]
        
    #     # === 3. 成功奖励 - 使用连续函数而非阶跃 ===
    #     if distance_to_goal <= 50.0:
    #         # 使用平滑的指数衰减
    #         success_bonus = 2.0 * np.exp(-distance_to_goal / 25.0)  # 范围: 0 到 2.0
    #     else:
    #         success_bonus = 0.0
        
    #     # === 4. 碰撞惩罚 - 严格限制 ===
    #     collision_penalty = 0.0
    #     current_collisions = getattr(self, 'collision_count', 0)
        
    #     if not hasattr(self, 'prev_collision_count'):
    #         self.prev_collision_count = 0
        
    #     new_collisions = current_collisions - self.prev_collision_count
    #     if new_collisions > 0:
    #         collision_penalty = -np.clip(new_collisions * 0.5, 0, 1.0)  # 最大-1.0
        
    #     if current_collisions > 0:
    #         collision_penalty += -0.1  # 轻微持续惩罚
        
    #     self.prev_collision_count = current_collisions
        
    #     # === 5. 移动方向奖励 - 新增，鼓励有效移动 ===
    #     direction_reward = 0.0
    #     if hasattr(self, 'prev_end_effector_pos'):
    #         movement = np.array(end_effector_pos) - np.array(self.prev_end_effector_pos)
    #         movement_norm = np.linalg.norm(movement)
            
    #         if movement_norm > 1e-6 and distance_to_goal > 1e-6:
    #             goal_direction = np.array(self.goal_pos) - np.array(end_effector_pos)
    #             goal_direction_norm = np.linalg.norm(goal_direction)
                
    #             if goal_direction_norm > 1e-6:
    #                 # 计算移动与目标方向的相似度
    #                 cosine_sim = np.dot(movement, goal_direction) / (movement_norm * goal_direction_norm)
    #                 direction_reward = np.clip(cosine_sim * 0.5, -0.5, 0.5)
        
    #     self.prev_end_effector_pos = end_effector_pos.copy()
        
    #     # === 6. 停滞惩罚 - 温和版本 ===
    #     stagnation_penalty = 0.0
    #     if distance_to_goal > 300:
    #         stagnation_penalty = -np.tanh((distance_to_goal - 300) / 100.0) * 0.5
        
    #     self.prev_distance = distance_to_goal
        
    #     # === 7. 总奖励计算 - 每个组件都有明确的边界 ===
    #     total_reward = (distance_reward +      # [-2.0, 0]
    #                 progress_reward +       # [-1.0, 1.0] 
    #                 success_bonus +         # [0, 2.0]
    #                 collision_penalty +     # [-1.1, 0]
    #                 direction_reward +      # [-0.5, 0.5]
    #                 stagnation_penalty)     # [-0.5, 0]
        
    #     # 总范围: 约 [-5.1, 3.5]，非常安全
        
    #     # === 8. 最终安全检查 ===
    #     final_reward = np.clip(total_reward, -5.0, 5.0)
        
    #     # 调试输出 - 监控异常值
    #     if abs(final_reward) > 3.0:
    #         self.logger.warning(f"⚠️ 大奖励值: {final_reward:.3f} (distance: {distance_to_goal:.1f})")
        
    #     return final_reward


    # def _compute_reward(self):
    #     """修复版奖励函数 - 适度的奖励幅度"""
    #     end_effector_pos = np.array(self._get_end_effector_position())
    #     distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
    #     # === 1. 适度的距离奖励 - 线性但范围控制 ===
    #     max_distance = 400.0  # 预期最大距离
    #     distance_reward = -distance_to_goal / max_distance * 3.0  # 范围: -3.0 到 0 (降低了)
        
    #     # === 2. 适度的分级成功奖励 ===
    #     success_bonus = 0.0
    #     if distance_to_goal <= 35.0:  # 完全成功
    #         success_bonus = 5.0  # 从50.0降低到5.0
    #     elif distance_to_goal <= 70.0:  # 接近成功
    #         success_bonus = 2.0  # 从20.0降低到2.0
    #     elif distance_to_goal <= 100.0:  # 部分成功
    #         success_bonus = 1.0  # 从10.0降低到1.0
    #     elif distance_to_goal <= 150.0:  # 有进展
    #         success_bonus = 0.5  # 从5.0降低到0.5
        
    #     # === 3. 适度的进步奖励 ===
    #     if not hasattr(self, 'prev_distance'):
    #         self.prev_distance = distance_to_goal
        
    #     progress = self.prev_distance - distance_to_goal
    #     progress_reward = progress * 5.0  # 从20.0降低到5.0
    #     progress_reward = np.clip(progress_reward, -2.0, 2.0)  # 更严格的限制
        
    #     # === 4. 适度的方向奖励 ===
    #     direction_reward = 0.0
    #     if hasattr(self, 'prev_end_effector_pos'):
    #         movement = np.array(end_effector_pos) - np.array(self.prev_end_effector_pos)
    #         movement_norm = np.linalg.norm(movement)
            
    #         if movement_norm > 1e-6:
    #             goal_direction = np.array(self.goal_pos) - np.array(end_effector_pos)
    #             goal_direction_norm = np.linalg.norm(goal_direction)
                
    #             if goal_direction_norm > 1e-6:
    #                 cosine_sim = np.dot(movement, goal_direction) / (movement_norm * goal_direction_norm)
    #                 direction_reward = cosine_sim * 0.5  # 从2.0降低到0.5
        
    #     self.prev_end_effector_pos = end_effector_pos.copy()
        
    #     # === 5. 适度的停滞惩罚 ===
    #     stagnation_penalty = 0.0
    #     if distance_to_goal > 200:
    #         stagnation_penalty = -0.5  # 从-2.0降低到-0.5
        
    #     # === 6. 适度的碰撞惩罚 ===
    #     collision_penalty = 0.0
    #     current_collisions = getattr(self, 'collision_count', 0)
        
    #     if not hasattr(self, 'prev_collision_count'):
    #         self.prev_collision_count = 0
        
    #     new_collisions = current_collisions - self.prev_collision_count
    #     if new_collisions > 0:
    #         collision_penalty = -np.clip(new_collisions * 0.5, 0, 1.0)  # 降低惩罚
        
    #     if current_collisions > 0:
    #         collision_penalty += -0.1  # 持续接触惩罚
        
    #     self.prev_collision_count = current_collisions
        
    #     self.prev_distance = distance_to_goal
        
    #     # === 7. 总奖励 ===
    #     total_reward = (distance_reward +      # [-3.0, 0]
    #                 progress_reward +       # [-2.0, 2.0] 
    #                 success_bonus +         # [0, 5.0]
    #                 direction_reward +      # [-0.5, 0.5]
    #                 stagnation_penalty +    # [-0.5, 0]
    #                 collision_penalty)      # [-1.1, 0]
        
    #     # 新的总范围: [-7.1, 7.5] ← 比之前小很多
        
    #     # === 8. 最终缩放 ===
    #     final_reward = total_reward * 0.5  # 再整体缩放50%
    #     # 最终范围: [-3.55, 3.75] ← 非常安全的范围
        
    #     # === 9. 调试输出 - 每100步输出一次奖励分解 ===
    #     if hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
    #         self.logger.info(f"🎯 Step {self.step_counter} 奖励分解:")
    #         self.logger.info(f"   距离奖励: {distance_reward:.2f} (距离: {distance_to_goal:.1f})")
    #         self.logger.info(f"   进步奖励: {progress_reward:.2f}")
    #         self.logger.info(f"   成功奖励: {success_bonus:.2f}")
    #         self.logger.info(f"   方向奖励: {direction_reward:.2f}")
    #         self.logger.info(f"   停滞惩罚: {stagnation_penalty:.2f}")
    #         self.logger.info(f"   碰撞惩罚: {collision_penalty:.2f}")
    #         self.logger.info(f"   最终奖励: {final_reward:.2f}")

    #     # 在_compute_reward的最后添加
    #     if hasattr(self, 'step_counter') and self.step_counter % 50 == 0:
    #         print(f"💰 [reward] Step {self.step_counter}: 奖励={final_reward:.3f}")
    #         print(f"  距离: {distance_to_goal:.1f}, 距离奖励: {distance_reward:.3f}")
    #         print(f"  进步: {progress:.1f}, 进步奖励: {progress_reward:.3f}")
    #         print(f"  成功奖励: {success_bonus:.3f}")
        
    #     return final_reward

    def _compute_reward(self):
        """基础奖励函数 - 简单稳定的奖励机制"""
        # 🔄 禁用路标点系统，使用基础奖励保证训练稳定性
        return self._compute_reward_basic()
    
    def _compute_reward_with_waypoints(self):
        """带路标点的奖励函数 - 稳定版"""
        end_effector_pos = np.array(self._get_end_effector_position())
        
        # === 1. 路标点导航奖励（平滑化）===
        waypoint_reward, waypoint_info = self.waypoint_navigator.update(end_effector_pos)
        
        # 🛡️ 平滑路标点奖励 - 避免突然跳跃
        if waypoint_reward > 5.0:  # 如果是大的即时奖励
            waypoint_reward = np.clip(waypoint_reward * 0.2, 0, 3.0)  # 降低到合理范围
        
        # === 2. 基础距离奖励（到当前目标的距离）===
        current_target = self.waypoint_navigator.get_current_target()
        distance_to_target = np.linalg.norm(end_effector_pos - current_target)
        
        # 使用较小的距离权重
        max_distance = 200.0
        distance_weight = 0.5  # 固定较小权重，保持稳定
        distance_reward = -distance_to_target / max_distance * distance_weight
        
        # === 3. 进度奖励 ===
        if not hasattr(self, 'prev_waypoint_distance'):
            self.prev_waypoint_distance = distance_to_target
        
        progress = self.prev_waypoint_distance - distance_to_target
        progress_reward = np.clip(progress * 1.0, -0.5, 0.5)  # 减小进度奖励幅度
        self.prev_waypoint_distance = distance_to_target
        
        # === 4. 完成度奖励 ===
        completion_progress = waypoint_info.get('completion_progress', 0.0)
        completion_bonus = completion_progress * 1.0  # 减小完成度奖励
        
        # === 5. 碰撞惩罚（保持原有） ===
        collision_penalty = self._get_collision_penalty()
        
        # === 6. 关节间距离惩罚 ===
        joint_distance_penalty = self._get_joint_distance_penalty()
        
        # === 7. 总奖励计算 (稳定版) ===
        total_reward = (
            waypoint_reward +       # [0, 3] 路标点奖励 (平滑后)
            distance_reward +       # [-0.65, 0] 距离惩罚
            progress_reward +       # [-0.5, 0.5] 进度奖励
            completion_bonus +      # [0, 1] 完成度奖励
            collision_penalty +     # [-2, 0] 碰撞惩罚
            joint_distance_penalty  # [-1, 0] 关节间距离惩罚
        )
        
        # 🛡️ 最终奖励稳定性保证
        total_reward = np.clip(total_reward, -5.0, 5.0)
        
        # === 8. 调试信息 ===
        if hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
            print(f"💰 [waypoint_reward] Step {self.step_counter}:")
            print(f"   路标奖励: {waypoint_reward:.2f}")
            print(f"   距离奖励: {distance_reward:.2f} (距离: {distance_to_target:.1f}, 权重: {distance_weight:.2f})")
            print(f"   进度奖励: {progress_reward:.2f}")
            print(f"   完成奖励: {completion_bonus:.2f}")
            print(f"   碰撞惩罚: {collision_penalty:.2f}")
            print(f"   关节惩罚: {joint_distance_penalty:.2f}")
            print(f"   总奖励: {total_reward:.2f}")
            print(f"   当前目标: {current_target}")
            print(f"   完成进度: {completion_progress*100:.1f}%")
        
        return total_reward
    
    def _compute_reward_basic(self):
        """基础奖励函数 - 包含碰撞惩罚"""
        end_effector_pos = np.array(self._get_end_effector_position())
        distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
        # 🔧 强化的距离奖励设计 (无waypoint时更重要)
        # 1. 分段式距离奖励（主要信号）
        max_distance = 300.0  
        
        # 分段式奖励：近距离给更高权重
        if distance_to_goal <= 50.0:
            # 很近：高权重，鼓励精确到达
            distance_weight = 4.0
        elif distance_to_goal <= 150.0:
            # 中等距离：中等权重
            distance_weight = 3.0
        else:
            # 远距离：基础权重
            distance_weight = 2.0
        
        distance_reward = -distance_to_goal / max_distance * distance_weight  # 范围: [-2.67, 0]
        
        # 2. 成功奖励（明确的目标）
        if distance_to_goal <= 35.0:
            success_reward = 5.0  # 简单的+1奖励
        else:
            success_reward = 0.0
        
        # 3. 增强的进度奖励
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = distance_to_goal
        
        progress = self.prev_distance - distance_to_goal
        
        # 根据当前距离调整进度奖励权重
        if distance_to_goal <= 50.0:
            progress_weight = 5.0  # 近距离时进步更重要
        elif distance_to_goal <= 150.0:
            progress_weight = 3.0  # 中距离时正常权重
        else:
            progress_weight = 2.0  # 远距离时较低权重
            
        progress_reward = np.clip(progress * progress_weight, -1.0, 1.0)
        self.prev_distance = distance_to_goal
        
        # 🚨 4. 添加碰撞惩罚
        collision_penalty = self._get_collision_penalty()
        # 但要确保更新prev_collision_count
        if hasattr(self, 'collision_count'):
            if not hasattr(self, 'prev_collision_count'):
                self.prev_collision_count = 0
            self.prev_collision_count = self.collision_count
        
        # 🔧 增强版总奖励范围计算
        # 距离奖励: [-2.67, 0] (分段权重)
        # 成功奖励: [0, 5.0] 
        # 进度奖励: [-1.0, 1.0] (分段权重)
        # 碰撞惩罚: [-2.0, 0] (限制范围)
        
        # 先限制碰撞惩罚范围
        collision_penalty = np.clip(collision_penalty, -2.0, 0.0)
        
        total_reward = distance_reward + success_reward + progress_reward + collision_penalty
        
        # 🛡️ 总奖励稳定性保护
        total_reward = np.clip(total_reward, -6.0, 8.0)  # 适应新的奖励范围
        
        # 调试输出
        # if hasattr(self, 'step_counter') and self.step_counter % 50 == 0:
        #     print(f"💰 [reward] Step {self.step_counter}: 奖励={total_reward:.3f}")
        #     print(f"  距离: {distance_to_goal:.1f}, 距离奖励: {distance_reward:.3f}")
        #     print(f"  进步奖励: {progress_reward:.3f}, 成功奖励: {success_reward:.3f}")
        #     if collision_penalty != 0:
        #         print(f"  🚨 碰撞惩罚: {collision_penalty:.3f} (碰撞次数: {getattr(self, 'collision_count', 0)})")
        
        return total_reward


    # def _compute_reward(self, debug_mode=True):    
    #     """
    #     距离为最大权重；在此基础上额外提升“横向距离”的权重：
    #     - distance_term: 以欧氏距离为主导（负值，越近越接近0）
    #     - x_term: 对 |dx| 施加额外线性惩罚（负值，越靠近目标x越接近0）
    #     其它项（方向/通道/避障/碰撞/时间）仅做轻量调味，不盖过距离。
    #     """
    #     import math
    #     import numpy as np
    #     eps = 1e-6

    #     # --- 末端与目标 ---
    #     ee = np.array(self._get_end_effector_position(), dtype=float)
    #     goal = np.array(getattr(self, "goal_pos", [600.0, 575.0]), dtype=float)
    #     ee_x, ee_y = float(ee[0]), float(ee[1])
    #     gx, gy = float(goal[0]), float(goal[1])

    #     dx = abs(ee_x - gx)
    #     dy = abs(ee_y - gy)
    #     d = float(math.hypot(dx, dy) + eps)

    #     # --- 参考尺度 & 主导项（欧氏距离）---
    #     reach = float(sum(getattr(self, "link_lengths", [60]*self.num_links)))
    #     dist_ref = max(300.0, reach)                     # 欧氏距离归一化基准
    #     w_dist = 4.0                                     # 主导权重
    #     distance_term = - w_dist * (d / dist_ref)        # d→0 时 → 0；d大 → 负

    #     # --- 横向距离额外权重（新） ---
    #     x_ref = max(200.0, 0.5 * reach)                  # 横向归一化基准
    #     w_x = 0.5                                        # ⬅️ 提高/降低横向权重就调这里
    #     # x_term = - w_x * (dx / x_ref)                    # dx→0 时 → 0；dx大 → 负
    #     x_term = 0                  # dx→0 时 → 0；dx大 → 负


    #     # --- 方向轻量奖励（避免就地抖动，极小权重）---
    #     if not hasattr(self, "prev_end_effector_pos"):
    #         self.prev_end_effector_pos = ee.copy()
    #     v = ee - self.prev_end_effector_pos
    #     v_norm = float(np.linalg.norm(v) + eps)
    #     g_vec = goal - ee
    #     g_norm = float(np.linalg.norm(g_vec) + eps)
    #     cos_theta = float(np.dot(v, g_vec) / (v_norm * g_norm))  # [-1,1]
    #     speed_gate = min(v_norm / 8.0, 1.0)                      # 限制单步贡献
    #     direction_term = 0.15 * (cos_theta * speed_gate)         # 很小

    #     # --- 通道惩罚（只罚越界，温和）---
    #     tunnel_center_y = 575.0
    #     half_width = 90.0
    #     # outside = max(0.0, abs(ee_y - tunnel_center_y) - half_width)
    #     # tunnel_penalty = -0.1 * (outside / (half_width + 1e-6))
    #     tunnel_penalty = -0.01 * (ee_y - tunnel_center_y)

    #     # --- 避障（仅在过近时给轻微负值）---
    #     if hasattr(self, "_get_min_obstacle_distance"):
    #         min_obs = float(self._get_min_obstacle_distance())
    #     else:
    #         min_obs = float("inf")
    #     safe_r = 45.0
    #     if math.isfinite(min_obs) and min_obs < safe_r:
    #         avoidance_term = -0.10 * (1.0 - (min_obs / safe_r))  # (0,-0.1]
    #     else:
    #         avoidance_term = 0.0

    #     # --- 碰撞（只罚新碰撞，幅度很小）---
    #     if not hasattr(self, "prev_collision_count"):
    #         self.prev_collision_count = int(getattr(self, "collision_count", 0))
    #     if not hasattr(self, "prev_self_collision_count"):
    #         self.prev_self_collision_count = int(getattr(self, "self_collision_count", 0))

    #     current_coll = int(getattr(self, "collision_count", 0))
    #     current_self = int(getattr(self, "self_collision_count", 0))
    #     new_coll = max(0, current_coll - int(self.prev_collision_count))
    #     new_self = max(0, current_self - int(self.prev_self_collision_count))

    #     collision_term = -0.50 * float(new_coll)
    #     self_collision_term = -0.50 * float(new_self)

    #     self.prev_collision_count = current_coll
    #     self.prev_self_collision_count = current_self

    #     # --- 微小时间惩罚 & 微成功加成（不盖过距离项）---
    #     time_term = -0.005
    #     goal_threshold = 35.0
    #     success_bonus = 0.20 if d <= goal_threshold else 0.0

    #     # --- 汇总（横向项生效，仍以总距离为王）---
    #     total_reward = (
    #         distance_term      # 主导
    #         + x_term           # ⬅️ 横向加权（新）
    #         + direction_term
    #         + tunnel_penalty
    #         + avoidance_term
    #         + collision_term
    #         + self_collision_term
    #         + time_term
    #         + success_bonus
    #     )

    #     # 更新缓存
    #     self.prev_end_effector_pos = ee.copy()

    #     # if debug_mode:
    #     #     print("\n🔍 REWARD DEBUG (x-weighted):")
    #     #     print(f"  End: ({ee_x:.1f},{ee_y:.1f})  Goal: ({gx:.1f},{gy:.1f})")
    #     #     print(f"  dx={dx:.2f} dy={dy:.2f}  d={d:.2f}")
    #     #     print(f"  distance_term={distance_term:+.3f} (ref={dist_ref:.1f}, w={w_dist})")
    #     #     print(f"  x_term      ={x_term:+.3f} (x_ref={x_ref:.1f}, w_x={w_x})")
    #     #     print(f"  direction   ={direction_term:+.3f}  tunnel={tunnel_penalty:+.3f}  avoid={avoidance_term:+.3f}")
    #     #     print(f"  coll(new={new_coll})={collision_term:+.3f}  self(new={new_self})={self_collision_term:+.3f}")
    #     #     print(f"  time={time_term:+.3f}  success={success_bonus:+.3f}")
    #     #     print(f"  ✅ TOTAL={total_reward:+.3f}")

    #     # # 渲染信息
    #     # self.current_reward_info = {
    #     #     'total_reward': float(total_reward),
    #     #     'distance_to_goal': float(d),
    #     #     'distance_term': float(distance_term),
    #     #     'x_term': float(x_term),
    #     #     'dx': float(dx),
    #     #     'dy': float(dy),
    #     #     'direction_term': float(direction_term),
    #     #     'tunnel_penalty': float(tunnel_penalty),
    #     #     'avoidance_term': float(avoidance_term),
    #     #     'collision_term': float(collision_term),
    #     #     'self_collision_term': float(self_collision_term),
    #     #     'time_term': float(time_term),
    #     #     'success_bonus': float(success_bonus),
    #     #     'y_deviation': float(abs(ee_y - tunnel_center_y)),
    #     #     'is_success': d <= goal_threshold
    #     # }
    #     return float(total_reward)
    
    def _compute_obstacle_avoidance_reward(self):
        """计算障碍物避让奖励 - 鼓励机器人保持与障碍物的安全距离"""
        if not hasattr(self, 'obstacles') or len(self.obstacles) == 0:
            return 0.0
        
        # 获取所有机器人关节的位置
        robot_positions = []
        for body in self.bodies:
            robot_positions.append(body.position)
        
        # 计算与所有障碍物的最短距离
        min_distance_to_obstacles = float('inf')
        
        for obstacle in self.obstacles:
            # 对于Segment障碍物，计算点到线段的距离
            if hasattr(obstacle, 'a') and hasattr(obstacle, 'b'):
                # 障碍物线段的两个端点
                seg_start = np.array(obstacle.a)
                seg_end = np.array(obstacle.b)
                
                # 计算每个机器人关节到这个线段的距离
                for robot_pos in robot_positions:
                    robot_pos = np.array(robot_pos)
                    # 计算点到线段的距离
                    dist = self._point_to_segment_distance(robot_pos, seg_start, seg_end)
                    min_distance_to_obstacles = min(min_distance_to_obstacles, dist)
        
        # 如果没有找到有效距离，返回0
        if min_distance_to_obstacles == float('inf'):
            return 0.0
        
        # 🎯 更温和的障碍物避让策略
        safe_distance = 40.0  # 从50.0降低到40.0像素，允许更近距离
        
        if min_distance_to_obstacles < safe_distance:
            # 距离太近，给予温和惩罚
            avoidance_reward = -(safe_distance - min_distance_to_obstacles) * 0.5  # 从1.0降低到0.5
        else:
            # 距离安全，给予小幅奖励
            avoidance_reward = min(15.0, (min_distance_to_obstacles - safe_distance) * 0.15)  # 减少奖励
        
        return avoidance_reward
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """计算点到线段的最短距离"""
        # 向量化计算
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        
        # 处理退化情况（线段长度为0）
        seg_length_sq = np.dot(seg_vec, seg_vec)
        if seg_length_sq == 0:
            return np.linalg.norm(point_vec)
        
        # 计算投影参数t
        t = np.dot(point_vec, seg_vec) / seg_length_sq
        t = max(0, min(1, t))  # 限制在[0,1]范围内
        
        # 计算最近点
        closest_point = seg_start + t * seg_vec
        
        # 返回距离
        return np.linalg.norm(point - closest_point)
    
    def _compute_path_efficiency_reward(self, end_effector_pos, distance_to_goal):
        """计算路径效率奖励 - 鼓励绕行而非后退"""
        if not hasattr(self, 'prev_end_effector_pos'):
            self.prev_end_effector_pos = end_effector_pos
            return 0.0
        
        # 计算末端执行器的移动方向
        movement_vector = np.array(end_effector_pos) - np.array(self.prev_end_effector_pos)
        movement_distance = np.linalg.norm(movement_vector)
        
        if movement_distance < 1e-6:  # 几乎无移动
            self.prev_end_effector_pos = end_effector_pos
            return 0.0
        
        # 目标方向向量
        goal_direction = np.array(self.goal_pos) - np.array(end_effector_pos)
        goal_distance = np.linalg.norm(goal_direction)
        
        if goal_distance < 1e-6:  # 已到达目标
            self.prev_end_effector_pos = end_effector_pos
            return 0.0
        
        goal_direction_normalized = goal_direction / goal_distance
        movement_direction_normalized = movement_vector / movement_distance
        
        # 计算移动方向与目标方向的角度相似度
        dot_product = np.dot(movement_direction_normalized, goal_direction_normalized)
        
        # 🎯 检查是否在避开障碍物的同时仍朝向目标的大致方向
        min_obstacle_distance = self._get_min_obstacle_distance()
        
        path_reward = 0.0
        
        if min_obstacle_distance < 50.0:  # 在障碍物附近
            # 如果正在远离障碍物且大致朝向目标，给予奖励
            if dot_product > 0.3:  # 至少30度以内朝向目标
                path_reward = movement_distance * 0.5  # 奖励有效移动
            elif dot_product > -0.5:  # 不是完全背离目标
                path_reward = movement_distance * 0.2  # 小幅奖励侧向移动
        else:  # 远离障碍物时
            # 鼓励直接朝向目标
            if dot_product > 0.7:  # 70度以内朝向目标
                path_reward = movement_distance * 1.0
        
        self.prev_end_effector_pos = end_effector_pos
        return path_reward
    
    def _get_min_obstacle_distance(self):
        """获取到最近障碍物的距离"""
        if not hasattr(self, 'bodies') or len(self.bodies) == 0:
            return float('inf')
        
        # 获取所有机器人关节的位置
        robot_positions = []
        for body in self.bodies:
            robot_positions.append(body.position)
        
        # 计算与所有障碍物的最短距离
        min_distance = float('inf')
        
        for obstacle in getattr(self, 'obstacles', []):
            if hasattr(obstacle, 'a') and hasattr(obstacle, 'b'):
                seg_start = np.array(obstacle.a)
                seg_end = np.array(obstacle.b)
                
                for robot_pos in robot_positions:
                    robot_pos = np.array(robot_pos)
                    dist = self._point_to_segment_distance(robot_pos, seg_start, seg_end)
                    min_distance = min(min_distance, dist)
        
        return min_distance if min_distance != float('inf') else 100.0


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
        
        # 绘制路标点系统（如果存在）
        if hasattr(self, 'waypoint_navigator'):
            self._render_waypoints()
        else:
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
            
            # 🔍 【调试】在红点旁边显示坐标
            if hasattr(pygame, 'font') and pygame.font.get_init():
                font = pygame.font.Font(None, 24)
                coord_text = f"End: ({end_effector_pos[0]:.0f},{end_effector_pos[1]:.0f})"
                text_surface = font.render(coord_text, True, (0, 0, 0))
                # 在红点上方显示坐标文字
                text_pos = (pos_int[0] - 40, pos_int[1] - 25)
                self.screen.blit(text_surface, text_pos)
            
        # 🎯 新增：绘制安全区域（可选调试）
        if hasattr(self, 'bodies') and len(self.bodies) > 0:
            # 绘制每个关节到障碍物的安全距离
            for body in self.bodies:
                pos = (int(body.position[0]), int(body.position[1]))
                # 绘制安全半径（浅蓝色圆圈）
                pygame.draw.circle(self.screen, (173, 216, 230), pos, 30, 1)
        
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(60)
    
    def _render_waypoints(self):
        """渲染路标点系统"""
        if not hasattr(self, 'waypoint_navigator'):
            return
        
        # 绘制所有路标点
        for i, waypoint in enumerate(self.waypoint_navigator.waypoints):
            pos_int = waypoint.position.astype(int)
            
            if waypoint.visited:
                # 已访问的路标点 - 绿色
                color = (0, 255, 0)
                border_color = (0, 150, 0)
                text_color = (255, 255, 255)
            elif i == self.waypoint_navigator.current_waypoint_idx:
                # 当前目标路标点 - 黄色闪烁
                brightness = int(200 + 55 * abs(pygame.time.get_ticks() % 1000 - 500) / 500)
                color = (brightness, brightness, 0)
                border_color = (180, 180, 0)
                text_color = (0, 0, 0)
            else:
                # 未访问的路标点 - 蓝色
                color = (100, 150, 255)
                border_color = (50, 100, 200)
                text_color = (255, 255, 255)
            
            # 绘制路标点圆圈
            radius = int(waypoint.radius * 0.8)  # 略小于判定半径
            pygame.draw.circle(self.screen, color, pos_int, radius)
            pygame.draw.circle(self.screen, border_color, pos_int, radius, 3)
            
            # 绘制路标点编号
            if hasattr(pygame, 'font') and pygame.font.get_init():
                font = pygame.font.Font(None, 24)
                text = font.render(str(i), True, text_color)
                text_rect = text.get_rect(center=pos_int)
                self.screen.blit(text, text_rect)
            
            # 绘制到达半径（当前目标的虚线圆）
            if i == self.waypoint_navigator.current_waypoint_idx:
                self._draw_dashed_circle(pos_int, int(waypoint.radius), (255, 255, 0), 2)
        
        # 绘制路标点之间的连线
        if len(self.waypoint_navigator.waypoints) > 1:
            points = [wp.position.astype(int) for wp in self.waypoint_navigator.waypoints]
            
            for i in range(len(points) - 1):
                start_pos = points[i]
                end_pos = points[i + 1]
                
                # 根据完成状态选择线条颜色
                if i < self.waypoint_navigator.current_waypoint_idx:
                    # 已完成的路径段 - 绿色实线
                    pygame.draw.line(self.screen, (0, 200, 0), start_pos, end_pos, 3)
                elif i == self.waypoint_navigator.current_waypoint_idx:
                    # 当前路径段 - 黄色虚线
                    self._draw_dashed_line(start_pos, end_pos, (255, 200, 0), 3)
                else:
                    # 未来路径段 - 灰色虚线
                    self._draw_dashed_line(start_pos, end_pos, (150, 150, 150), 2)
        
        # 绘制进度信息
        self._render_waypoint_info()
    
    def _draw_dashed_circle(self, center, radius, color, width):
        """绘制虚线圆圈"""
        circumference = 2 * 3.14159 * radius
        dash_length = 8
        num_dashes = int(circumference / (dash_length * 2))
        
        for i in range(num_dashes):
            start_angle = (i * 2 * 3.14159) / num_dashes
            end_angle = ((i + 0.5) * 2 * 3.14159) / num_dashes
            
            start_x = center[0] + radius * np.cos(start_angle)
            start_y = center[1] + radius * np.sin(start_angle)
            end_x = center[0] + radius * np.cos(end_angle)
            end_y = center[1] + radius * np.sin(end_angle)
            
            pygame.draw.line(self.screen, color, 
                           (int(start_x), int(start_y)), 
                           (int(end_x), int(end_y)), width)
    
    def _draw_dashed_line(self, start_pos, end_pos, color, width):
        """绘制虚线"""
        distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        direction = (np.array(end_pos) - np.array(start_pos)) / distance
        
        dash_length = 10
        gap_length = 5
        current_pos = np.array(start_pos, dtype=float)
        
        while np.linalg.norm(current_pos - start_pos) < distance:
            # 绘制实线段
            next_pos = current_pos + direction * min(dash_length, 
                                                   distance - np.linalg.norm(current_pos - start_pos))
            
            if np.linalg.norm(next_pos - start_pos) <= distance:
                pygame.draw.line(self.screen, color, 
                               current_pos.astype(int), next_pos.astype(int), width)
            
            # 移动到下一个实线段起点
            current_pos = next_pos + direction * gap_length
    
    def _render_waypoint_info(self):
        """渲染路标点信息面板"""
        if not hasattr(pygame, 'font') or not pygame.font.get_init():
            return
        
        # 创建信息面板
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)
        
        progress = self.waypoint_navigator.get_progress_info()
        current_idx = self.waypoint_navigator.current_waypoint_idx
        total_waypoints = len(self.waypoint_navigator.waypoints)
        
        # 背景面板
        panel_width = 250
        panel_height = 120
        panel_x = 10
        panel_y = 10
        
        # 半透明背景
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(180)
        panel_surface.fill((50, 50, 50))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # 标题
        title_text = font.render("🗺️ Waypoint Navigation", True, (255, 255, 255))
        self.screen.blit(title_text, (panel_x + 10, panel_y + 10))
        
        # 进度信息
        progress_text = small_font.render(f"Progress: {progress['progress_percentage']:.1f}%", True, (255, 255, 255))
        self.screen.blit(progress_text, (panel_x + 10, panel_y + 35))
        
        waypoint_text = small_font.render(f"Waypoint: {current_idx}/{total_waypoints}", True, (255, 255, 255))
        self.screen.blit(waypoint_text, (panel_x + 10, panel_y + 55))
        
        reward_text = small_font.render(f"Reward: {progress['total_reward_earned']:.1f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (panel_x + 10, panel_y + 75))
        
        # 当前目标位置
        if current_idx < total_waypoints:
            target = progress['current_target']
            target_text = small_font.render(f"Target: ({target[0]:.0f}, {target[1]:.0f})", True, (255, 255, 0))
            self.screen.blit(target_text, (panel_x + 10, panel_y + 95))  # 控制渲染帧率

    def _init_waypoint_system(self):
        """初始化路标点系统"""
        if hasattr(self, 'waypoint_navigator'):
            return  # 已经初始化
            
        # 导入路标点系统
        import sys
        import os
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../')
        sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model'))
        from waypoint_navigator import WaypointNavigator
        
        # 创建路标点导航器
        start_pos = self.anchor_point
        goal_pos = self.goal_pos
        self.waypoint_navigator = WaypointNavigator(start_pos, goal_pos)
        
        print(f"🗺️ 路标点系统已初始化")
        print(f"   起点: {start_pos}")
        print(f"   终点: {goal_pos}")
        print(f"   路标数: {len(self.waypoint_navigator.waypoints)}")

    def _reset_waypoint_system(self):
        """重置路标点系统"""
        if hasattr(self, 'waypoint_navigator') and self.waypoint_navigator is not None:
            self.waypoint_navigator.reset()
            
            # 重置路标点相关的状态变量
            if hasattr(self, 'prev_waypoint_distance'):
                delattr(self, 'prev_waypoint_distance')
                
            print(f"🗺️ 路标点系统已重置")
        else:
            # 如果还没有路标点系统，则初始化它
            self._init_waypoint_system()

    def _detect_and_fix_explosion(self, pre_step_velocities):
        """检测和修正炸开现象"""
        explosion_detected = False
        
        for i, body in enumerate(self.bodies):
            if i < len(pre_step_velocities):
                pre_vel = pre_step_velocities[i]
                
                # 检查速度突变
                velocity_change = np.linalg.norm(np.array(body.velocity) - pre_vel['velocity'])
                angular_velocity_change = abs(body.angular_velocity - pre_vel['angular_velocity'])
                
                # 🚨 炸开检测：速度突然大幅增加
                if (velocity_change > 150.0 or 
                    angular_velocity_change > 8.0 or
                    np.linalg.norm(body.velocity) > self.max_safe_velocity or
                    abs(body.angular_velocity) > self.max_safe_angular_velocity):
                    
                    explosion_detected = True
                    
                    # 🔧 温和修正：不是直接设为0，而是渐进减少
                    if np.linalg.norm(body.velocity) > self.max_safe_velocity:
                        # 限制线速度
                        vel_direction = np.array(body.velocity) / (np.linalg.norm(body.velocity) + 1e-6)
                        body.velocity = (vel_direction * self.max_safe_velocity * 0.5).tolist()
                    
                    if abs(body.angular_velocity) > self.max_safe_angular_velocity:
                        # 限制角速度
                        body.angular_velocity = np.sign(body.angular_velocity) * self.max_safe_angular_velocity * 0.5
                    
                    self.logger.warning(f"🚨 检测到Link{i}炸开倾向，已修正速度")
        
        if explosion_detected:
            self.logger.warning("🔴 检测到炸开现象，已进行速度修正")

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
    
    # 📊 最终总结
    env.logger.info("\n" + "="*60)    
    env.logger.info("🎯 增强角度限制测试总结:")
    env.logger.info(f"✅ 测试步数: {step_count}")
    env.logger.info(f"✅ 约束数量: {len(env.joint_limits)}")
    env.logger.info(f"✅ 最终关节角度: {env.get_joint_angles()}")
    env.logger.info(f"✅ 改进的角度限制系统:")
    env.logger.info(f"   - 移除了SimpleMotor (避免冲突)")
    env.logger.info(f"   - 增强了RotaryLimitJoint约束力")
    env.logger.info(f"   - 添加了双重角度强制检查")
    env.logger.info(f"   - 增加了关节间碰撞检测")
    env.logger.info(f"   - 使用更严格的角度限制")
    env.logger.info("="*60)   
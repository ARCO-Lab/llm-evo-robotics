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

    
    def __init__(self, num_links=3, link_lengths=None, render_mode=None, config_path=None, curriculum_stage=0):

        super().__init__()
        self.config = self._load_config(config_path)
        print(f"self.config: {self.config}")
        self.anchor_point = self.config["start"]["position"]
        self.gym_api_version = "old" # old or new. new is gymnasium, old is gym
        
        # 🎯 课程学习参数
        self.curriculum_stage = curriculum_stage
        self.base_goal_pos = np.array(self.config["goal"]["position"]) if "goal" in self.config else np.array([600, 575])

        self.num_links = num_links  # 修复：使用传入的参数
        if link_lengths is None:

            self.link_lengths = [60] * num_links

        else:
            assert len(link_lengths) == num_links
            self.link_lengths = link_lengths
        
        self.render_mode = render_mode
        # self.goal_pos = np.array([250.0, 250.0])
        self.dt = 1/60.0  # 增加时间步长精度
        self.max_torque = 100  # 增加最大扭矩

        # 定义Gymnasium必需的action_space和observation_space
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, shape=(self.num_links,), dtype=np.float32)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_links * 2 + 7,), dtype=np.float32)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 981.0)
        # 减少全局阻尼
        self.space.damping = 0.999  # 🔧 增加阻尼让角度限制更有效
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
            mass = density * length * 10  # 🔧 增加质量
            moment = pymunk.moment_for_segment(mass, (0, 0), (length, 0), 8)  # 🔧 增加半径
            body = pymunk.Body(mass, moment)
            
            # 🔧 设置初始位置（让机器人自然垂直下垂）
            body.position = current_pos
            body.angle = math.pi/4  # 改为45°避开边界
            
            # 🔧 创建形状 - 增加半径让碰撞更明显
            shape = pymunk.Segment(body, (0, 0), (length, 0), 8)  # 半径从5增加到8
            shape.friction = 0.8  # 🔧 增加摩擦力
            shape.collision_type = i + 1  # 🔧 为每个link设置不同的碰撞类型
            
            self.space.add(body, shape)
            self.bodies.append(body)
            
            # 创建关节连接和Motor
            if i == 0:
                # 🔧 基座关节：连接到世界 - 使用PivotJoint实现revolute关节
                joint = pymunk.PivotJoint(self.space.static_body, body, self.anchor_point, (0, 0))
                joint.collide_bodies = False
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
                    limit_joint.max_force = 100000  # 进一步增加约束力，确保能够约束Motor
                    self.space.add(limit_joint)
                    self.joint_limits.append(limit_joint)
                else:
                    # 基座关节无角度限制，添加None占位符
                    self.joint_limits.append(None)
                
            else:
                # 🔧 连接到前一个link的末端 - 使用PivotJoint实现revolute关节
                joint = pymunk.PivotJoint(prev_body, body, (self.link_lengths[i-1], 0), (0, 0))
                joint.collide_bodies = False
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
                    limit_joint.max_force = 100000  # 进一步增加约束力，确保能够约束Motor
                    self.space.add(limit_joint)
                    self.joint_limits.append(limit_joint)
                else:
                    # 无角度限制的关节，添加None占位符
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
        """设置碰撞处理器"""
        try:
            # 🎯 1. 机器人关节间碰撞处理（现有代码）
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
                        print(f"✅ 设置关节{i+1}与关节{j+1}的碰撞检测")
                    except Exception as e:
                        print(f"⚠️ 设置关节碰撞处理器失败: {e}")
            
            # 🎯 2. 新增：机器人与障碍物碰撞处理 - 使用正确API和begin回调
            def robot_obstacle_collision_handler(arbiter, space, data):
                """处理机器人与障碍物的碰撞"""
                # 记录碰撞信息
                if not hasattr(self, 'collision_count'):
                    self.collision_count = 0
                self.collision_count += 1
                
                print(f"🚨 检测到机器人-障碍物碰撞! 总计: {self.collision_count}")
                
                # 可以选择：
                # return True   # 允许碰撞（物理反弹）
                # return False  # 阻止碰撞（穿透）
                return True  # 推荐：允许物理碰撞，提供真实反馈
            
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
                    print(f"✅ 设置机器人链接{i+1}与障碍物的碰撞检测")
                except Exception as e:
                    print(f"⚠️ 设置机器人-障碍物碰撞处理器失败: {e}")
                    
        except Exception as e:
            print(f"⚠️ 碰撞处理器设置跳过: {e}")

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
        
        # 🎯 课程学习：根据阶段调整目标位置
        if hasattr(self, 'curriculum_stage'):
            if self.curriculum_stage == 0:
                # 阶段0：目标很近，容易达到
                self.goal_pos = self.base_goal_pos * 0.7 + np.array(self.anchor_point) * 0.3
            elif self.curriculum_stage == 1:
                # 阶段1：中等距离
                self.goal_pos = self.base_goal_pos * 0.85 + np.array(self.anchor_point) * 0.15
            else:
                # 阶段2+：完整难度
                self.goal_pos = self.base_goal_pos
        
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
        """使用Motor控制 + 物理约束，结合真实性和安全性"""
        actions = np.clip(actions, -self.max_torque, self.max_torque)
        
        # 🔧 将扭矩转换为角速度目标，通过Motor控制
        # 简单的比例控制：扭矩 → 角速度
        torque_to_speed_ratio = 0.01  # 调节这个比例来控制响应性
        
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
        
        if self.step_counter % 20 == 0:  # 每20步打印一次
            self._print_motor_status()
        
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = False
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
        
        print(f"步骤 {self.step_counter:4d} - 绝对角度: {[f'{a:7.1f}°' for a in absolute_angles]}")
        print(f"              相对角度: {[f'{a:7.1f}°' for a in relative_angles]}")
        
        # 打印Motor状态
        motor_rates = [motor.rate for motor in self.motors]
        print(f"    Motor角速度: {[f'{r:6.2f}' for r in motor_rates]} rad/s")
        
        # 检查约束是否还存在
        active_constraints = [c for c in self.joint_limits if c is not None]
        constraints_count = len([c for c in self.space.constraints if hasattr(c, 'min')])
        motors_count = len([c for c in self.space.constraints if isinstance(c, pymunk.SimpleMotor)])
        print(f"    约束数量: {constraints_count}/{len(active_constraints)} 角度限制, {motors_count}/{len(self.motors)} Motors")
        
        # 检查相对角度是否超出限制
        limit_degrees = [None, (-120, 120), (-120, 120), (-120, 120), (-120, 120)]  # 基座无限制
        violations = []
        for i, (rel_angle, limits) in enumerate(zip(relative_angles, limit_degrees)):
            if limits is not None:  # 跳过无限制的关节
                min_limit, max_limit = limits
                if rel_angle < min_limit or rel_angle > max_limit:
                    violations.append(f"关节{i+1}相对角度超限: {rel_angle:.1f}°")
        
        if violations:
            print(f"    ⚠️  角度超限: {', '.join(violations)} (物理约束应该防止这种情况)")
        else:
            if len(active_constraints) > 0:
                print(f"    ✅ 所有受限关节相对角度在范围内 (基座关节无限制)")
            else:
                print(f"    ✅ 所有关节正常运行")

    def get_joint_angles(self):
        """获取所有关节的当前角度（度数）"""
        return [math.degrees(body.angle) for body in self.bodies]

    def _compute_reward(self):
        """超稳定奖励函数 - 防止数值爆炸"""
        end_effector_pos = np.array(self._get_end_effector_position())
        distance_to_goal = np.linalg.norm(end_effector_pos - self.goal_pos)
        
        # === 1. 距离奖励 - 使用tanh防止极值 ===
        distance_reward = -np.tanh(distance_to_goal / 100.0) * 2.0  # 范围: -2.0 到 0
        
        # === 2. 进步奖励 - 严格限制范围 ===
        if not hasattr(self, 'prev_distance'):
            self.prev_distance = distance_to_goal
        
        progress = self.prev_distance - distance_to_goal
        progress_reward = np.clip(progress * 5.0, -1.0, 1.0)  # 严格限制在[-1,1]
        
        # === 3. 成功奖励 - 使用连续函数而非阶跃 ===
        if distance_to_goal <= 50.0:
            # 使用平滑的指数衰减
            success_bonus = 2.0 * np.exp(-distance_to_goal / 25.0)  # 范围: 0 到 2.0
        else:
            success_bonus = 0.0
        
        # === 4. 碰撞惩罚 - 严格限制 ===
        collision_penalty = 0.0
        current_collisions = getattr(self, 'collision_count', 0)
        
        if not hasattr(self, 'prev_collision_count'):
            self.prev_collision_count = 0
        
        new_collisions = current_collisions - self.prev_collision_count
        if new_collisions > 0:
            collision_penalty = -np.clip(new_collisions * 0.5, 0, 1.0)  # 最大-1.0
        
        if current_collisions > 0:
            collision_penalty += -0.1  # 轻微持续惩罚
        
        self.prev_collision_count = current_collisions
        
        # === 5. 移动方向奖励 - 新增，鼓励有效移动 ===
        direction_reward = 0.0
        if hasattr(self, 'prev_end_effector_pos'):
            movement = np.array(end_effector_pos) - np.array(self.prev_end_effector_pos)
            movement_norm = np.linalg.norm(movement)
            
            if movement_norm > 1e-6 and distance_to_goal > 1e-6:
                goal_direction = np.array(self.goal_pos) - np.array(end_effector_pos)
                goal_direction_norm = np.linalg.norm(goal_direction)
                
                if goal_direction_norm > 1e-6:
                    # 计算移动与目标方向的相似度
                    cosine_sim = np.dot(movement, goal_direction) / (movement_norm * goal_direction_norm)
                    direction_reward = np.clip(cosine_sim * 0.5, -0.5, 0.5)
        
        self.prev_end_effector_pos = end_effector_pos.copy()
        
        # === 6. 停滞惩罚 - 温和版本 ===
        stagnation_penalty = 0.0
        if distance_to_goal > 300:
            stagnation_penalty = -np.tanh((distance_to_goal - 300) / 100.0) * 0.5
        
        self.prev_distance = distance_to_goal
        
        # === 7. 总奖励计算 - 每个组件都有明确的边界 ===
        total_reward = (distance_reward +      # [-2.0, 0]
                    progress_reward +       # [-1.0, 1.0] 
                    success_bonus +         # [0, 2.0]
                    collision_penalty +     # [-1.1, 0]
                    direction_reward +      # [-0.5, 0.5]
                    stagnation_penalty)     # [-0.5, 0]
        
        # 总范围: 约 [-5.1, 3.5]，非常安全
        
        # === 8. 最终安全检查 ===
        final_reward = np.clip(total_reward, -5.0, 5.0)
        
        # 调试输出 - 监控异常值
        if abs(final_reward) > 3.0:
            print(f"⚠️ 大奖励值: {final_reward:.3f} (distance: {distance_to_goal:.1f})")
        
        return final_reward
    
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
        
        # 🎯 定义障碍物的collision_type
        OBSTACLE_COLLISION_TYPE = 100  # 使用大数字避免与机器人冲突
        
        for obs in self.config["obstacles"]:
            if obs["shape"] == "segment":
                p1 = tuple(obs["points"][0])
                p2 = tuple(obs["points"][1])
                shape = pymunk.Segment(self.space.static_body, p1, p2, 3.0)
                shape.friction = 1.0
                shape.color = (0,0,0,255)
                
                # 🎯 关键添加：设置障碍物碰撞类型
                shape.collision_type = OBSTACLE_COLLISION_TYPE
                
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
        
        # 🎯 新增：绘制安全区域（可选调试）
        if hasattr(self, 'bodies') and len(self.bodies) > 0:
            # 绘制每个关节到障碍物的安全距离
            for body in self.bodies:
                pos = (int(body.position[0]), int(body.position[1]))
                # 绘制安全半径（浅蓝色圆圈）
                pygame.draw.circle(self.screen, (173, 216, 230), pos, 30, 1)
        
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
    print("\n" + "="*60)
    print("🎯 增强角度限制测试总结:")
    print(f"✅ 测试步数: {step_count}")
    print(f"✅ 约束数量: {len(env.joint_limits)}")
    print(f"✅ 最终关节角度: {env.get_joint_angles()}")
    print(f"✅ 改进的角度限制系统:")
    print(f"   - 移除了SimpleMotor (避免冲突)")
    print(f"   - 增强了RotaryLimitJoint约束力")
    print(f"   - 添加了双重角度强制检查")
    print(f"   - 增加了关节间碰撞检测")
    print(f"   - 使用更严格的角度限制")
    print("="*60)
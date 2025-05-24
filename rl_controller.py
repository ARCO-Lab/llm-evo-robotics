"""
强化学习机器人设计测试框架
用于测试基于NSGA-II算法进化出的机器人设计
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gym
from gym import spaces
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch


class RobotDesignEnv(gym.Env):
    """
    用于测试进化机器人设计的强化学习环境
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, urdf_path=None, max_steps=1000, render_mode="human", 
                 terrain_type="flat", reward_type="distance"):
        super(RobotDesignEnv, self).__init__()
        
        # 环境参数
        self.urdf_path = urdf_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.terrain_type = terrain_type
        self.reward_type = reward_type
        self.step_counter = 0
        
        # PyBullet连接
        self.physics_client_id = -1
        self.robot_id = -1
        self.initial_position = [0, 0, 0.1]
        self.prev_position = self.initial_position.copy()
        
        # 动作空间：关节控制
        self.action_space = None  # 将在reset时根据机器人关节数量设置
        
        # 观测空间：关节位置、速度、机器人姿态等
        self.observation_space = None  # 将在reset时设置
        
        # 性能指标
        self.total_distance = 0
        self.total_energy = 0
        self.path_points = []
        self.start_position = None
        
        # 初始化环境
        self.reset()
    
    def step(self, action):
        """执行一个动作并返回下一个状态、奖励、完成标志和信息"""
        self.step_counter += 1
        
        # 应用动作到关节
        self._apply_action(action)
        
        # 模拟物理步骤
        p.stepSimulation(physicsClientId=self.physics_client_id)
        
        # 获取新的观测
        observation = self._get_observation()
        
        # 计算奖励
        reward, reward_info = self._compute_reward()
        
        # 检查是否完成
        done = self._is_done()
        
        # 收集额外信息
        info = {
            'step': self.step_counter,
            'distance': self.total_distance,
            'energy': self.total_energy,
            'reward_info': reward_info
        }
        
        return observation, reward, done, info
    
    def reset(self):
        """重置环境状态"""
        # 如果已连接，则断开
        if self.physics_client_id >= 0:
            p.disconnect(physicsClientId=self.physics_client_id)
        
        # 设置PyBullet
        if self.render_mode == "human":
            self.physics_client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client_id)
        else:
            self.physics_client_id = p.connect(p.DIRECT)
        
        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client_id)
        
        # 设置相机
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0],
            physicsClientId=self.physics_client_id
        )
        
        # 加载地形
        self._load_terrain()
        
        # 加载机器人
        self._load_robot()
        
        # 重置计数器和性能指标
        self.step_counter = 0
        self.total_distance = 0
        self.total_energy = 0
        self.path_points = []
        
        # 记录初始位置
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, 
            physicsClientId=self.physics_client_id
        )
        self.prev_position = list(pos)
        self.start_position = list(pos)
        self.path_points.append(self.prev_position)
        
        # 获取观测
        observation = self._get_observation()
        
        return observation
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'rgb_array':
            # 获取相机图像
            width, height = 320, 240
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=1.5,
                yaw=0,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client_id
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.1,
                farVal=100,
                physicsClientId=self.physics_client_id
            )
            
            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.physics_client_id
            )
            
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            
            return rgb_array
        
        return None
    
    def close(self):
        """关闭环境"""
        if self.physics_client_id >= 0:
            p.disconnect(physicsClientId=self.physics_client_id)
            self.physics_client_id = -1
    
    def _load_terrain(self):
        """加载地形"""
        if self.terrain_type == "flat":
            p.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client_id
            )
        elif self.terrain_type == "hills":
            # 创建随机山丘地形
            heightfieldData = np.zeros((256, 256))
            for i in range(256):
                for j in range(256):
                    heightfieldData[i][j] = np.sin(i/20) * np.cos(j/20) * 0.5
            
            terrainShape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[0.05, 0.05, 0.5],
                heightfieldData=heightfieldData,
                numHeightfieldRows=256,
                numHeightfieldColumns=256,
                physicsClientId=self.physics_client_id
            )
            p.createMultiBody(0, terrainShape, physicsClientId=self.physics_client_id)
        elif self.terrain_type == "steps":
            # 创建台阶地形
            p.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client_id
            )
            
            # 添加一些障碍物
            for i in range(5):
                box_pos = [3 + i*1.0, 0, 0.1 + i*0.1]
                box_size = [0.5, 2.0, 0.2]
                box_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=box_size,
                    physicsClientId=self.physics_client_id
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=box_id,
                    basePosition=box_pos,
                    physicsClientId=self.physics_client_id
                )
    
    def _load_robot(self):
        """加载机器人模型"""
        if not self.urdf_path or not os.path.exists(self.urdf_path):
            raise ValueError(f"URDF文件不存在: {self.urdf_path}")
        
        # 加载机器人URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.initial_position,
            physicsClientId=self.physics_client_id
        )
        
        # 获取机器人关节信息
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id)
        self.active_joints = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            joint_type = joint_info[2]
            
            if joint_type != p.JOINT_FIXED:
                self.active_joints.append(i)
        
        # 设置动作空间
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.active_joints),),
            dtype=np.float32
        )
        
        # 设置观测空间 (关节位置、速度、机器人位置、方向)
        observation_dim = len(self.active_joints) * 2 + 7  # 位置+速度+基础位置+四元数
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )
    
    def _apply_action(self, action):
        """将动作应用到机器人关节"""
        # 将动作从[-1, 1]映射到合适的速度范围
        max_velocity = 5.0
        target_velocities = action * max_velocity
        
        # 设置关节速度
        max_force = 10.0
        for i, joint_idx in enumerate(self.active_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[i],
                force=max_force,
                physicsClientId=self.physics_client_id
            )
    
    def _get_observation(self):
        """获取环境观测"""
        # 获取机器人基础位置和方向
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client_id
        )
        
        # 获取关节状态
        joint_states = []
        for joint_idx in self.active_joints:
            joint_state = p.getJointState(
                self.robot_id,
                joint_idx,
                physicsClientId=self.physics_client_id
            )
            joint_states.append(joint_state[0])  # 位置
            joint_states.append(joint_state[1])  # 速度
        
        # 合并所有观测
        observation = np.concatenate([
            np.array(base_pos),
            np.array(base_orn),
            np.array(joint_states)
        ]).astype(np.float32)
        
        return observation
    
    def _compute_reward(self):
        """计算奖励"""
        # 获取当前位置
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client_id
        )
        curr_position = list(pos)
        
        # 计算移动距离
        step_distance_x = curr_position[0] - self.prev_position[0]
        step_distance_y = curr_position[1] - self.prev_position[1]
        step_distance = np.sqrt(step_distance_x**2 + step_distance_y**2)
        
        # 更新总距离
        self.total_distance += step_distance
        
        # 计算能量消耗 (简化模型)
        energy = 0
        for joint_idx in self.active_joints:
            joint_state = p.getJointState(
                self.robot_id,
                joint_idx,
                physicsClientId=self.physics_client_id
            )
            velocity = joint_state[1]
            force = p.getJointState(
                self.robot_id,
                joint_idx,
                physicsClientId=self.physics_client_id
            )[3]
            
            energy += abs(velocity * force) * 0.01
        
        self.total_energy += energy
        
        # 更新路径点
        self.path_points.append(curr_position)
        
        # 更新位置
        self.prev_position = curr_position
        
        # 计算奖励
        reward_info = {}
        
        if self.reward_type == "distance":
            # 主要奖励前向移动
            forward_reward = step_distance_x * 10.0 if step_distance_x > 0 else 0.0
            
            # 惩罚侧向移动
            lateral_penalty = abs(step_distance_y) * 5.0
            
            # 惩罚高能耗
            energy_penalty = energy * 0.1
            
            # 稳定性奖励
            height = curr_position[2]
            target_height = 0.2
            stability_reward = -abs(height - target_height) * 5.0
            
            # 总奖励
            reward = forward_reward - lateral_penalty - energy_penalty + stability_reward
            
            # 收集奖励信息
            reward_info = {
                'forward_reward': forward_reward,
                'lateral_penalty': lateral_penalty,
                'energy_penalty': energy_penalty,
                'stability_reward': stability_reward,
                'total_reward': reward
            }
        
        elif self.reward_type == "survival":
            # 生存奖励
            reward = 0.1
            
            # 如果机器人翻倒或离开地面太远，给予惩罚
            height = curr_position[2]
            if height < 0.05 or height > 0.5:
                reward -= 1.0
            
            # 前进奖励
            if step_distance_x > 0:
                reward += step_distance_x * 5.0
            
            reward_info = {
                'survival_reward': 0.1,
                'height_penalty': 0 if 0.05 <= height <= 0.5 else -1.0,
                'forward_reward': step_distance_x * 5.0 if step_distance_x > 0 else 0,
                'total_reward': reward
            }
        
        else:  # 默认简单奖励
            reward = step_distance_x
            reward_info = {'step_distance_x': step_distance_x}
        
        return reward, reward_info
    
    def _is_done(self):
        """检查是否完成"""
        # 检查步数是否达到最大步数
        if self.step_counter >= self.max_steps:
            return True
        
        # 获取机器人高度
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client_id
        )
        height = pos[2]
        
        # 检查机器人是否翻倒或离开地面太远
        if height < 0.02 or height > 1.0:
            return True
        
        return False
    
    def get_performance_metrics(self):
        """返回性能指标"""
        # 计算直线性
        if len(self.path_points) < 2:
            linearity = 0.0
        else:
            start = np.array(self.path_points[0])
            end = np.array(self.path_points[-1])
            path_length = 0.0
            
            for i in range(1, len(self.path_points)):
                p1 = np.array(self.path_points[i-1])
                p2 = np.array(self.path_points[i])
                path_length += np.linalg.norm(p2 - p1)
            
            direct_distance = np.linalg.norm(end - start)
            
            if path_length > 0:
                linearity = direct_distance / path_length
            else:
                linearity = 0.0
        
        return {
            'total_distance': self.total_distance,
            'path_linearity': linearity,
            'total_energy': self.total_energy
        }


class RLRobotTester:
    """
    强化学习机器人测试类
    用于测试进化出的机器人设计并优化其控制策略
    """
    def __init__(self, results_dir="rl_results"):
        """初始化测试器"""
        self.results_dir = results_dir
        
        # 创建结果目录
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 记录模型和性能
        self.trained_models = {}
        self.performance_history = {}
        
    def create_test_environment(self, urdf_path, render_mode="human", max_steps=1000, 
                              terrain_type="flat", reward_type="distance"):
        """创建测试环境"""
        env = RobotDesignEnv(
            urdf_path=urdf_path,
            render_mode=render_mode,
            max_steps=max_steps,
            terrain_type=terrain_type,
            reward_type=reward_type
        )
        
        # 包装环境以监控性能
        env = Monitor(env)
        
        # 向量化环境
        vec_env = DummyVecEnv([lambda: env])
        
        return vec_env
    
    def train_controller(self, env, robot_id, algorithm="PPO", total_timesteps=20000):
        """
        训练机器人控制器
        
        参数:
            env: 环境
            robot_id: 机器人ID
            algorithm: 算法 ("PPO", "SAC", "TD3")
            total_timesteps: 训练总步数
        
        返回:
            训练好的模型
        """
        # 创建模型保存路径
        model_dir = os.path.join(self.results_dir, f"model_{robot_id}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 定义评估回调
        eval_callback = EvalCallback(
            env,
            best_model_save_path=model_dir,
            log_path=model_dir,
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        # 定义检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=model_dir,
            name_prefix=f"rl_model_{algorithm}"
        )
        
        # 选择算法
        if algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=os.path.join(self.results_dir, "tb_logs"),
                device="auto"
            )
        elif algorithm == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=os.path.join(self.results_dir, "tb_logs"),
                device="auto"
            )
        elif algorithm == "TD3":
            model = TD3(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=os.path.join(self.results_dir, "tb_logs"),
                device="auto"
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        # 训练模型
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, f"final_model_{algorithm}.zip")
        model.save(final_model_path)
        
        # 记录训练好的模型
        self.trained_models[robot_id] = {
            'model': model,
            'algorithm': algorithm,
            'path': final_model_path
        }
        
        return model
    
    def evaluate_controller(self, env, model, n_eval_episodes=10):
        """
        评估控制器性能
        
        参数:
            env: 环境
            model: 模型
            n_eval_episodes: 评估轮数
        
        返回:
            评估结果
        """
        # 评估模型
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        # 收集更详细的性能指标
        episode_metrics = []
        
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
            
            # 获取性能指标
            env_metrics = env.get_env_method("get_performance_metrics")[0]
            episode_metrics.append(env_metrics)
        
        # 计算平均性能指标
        avg_metrics = {}
        for key in episode_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in episode_metrics])
        
        # 添加标准奖励
        avg_metrics['mean_reward'] = mean_reward
        avg_metrics['std_reward'] = std_reward
        
        return avg_metrics
    
    def test_evolved_robot(self, urdf_path, robot_id, algorithm="PPO", 
                          train_timesteps=20000, n_eval_episodes=5,
                          terrain_type="flat", render_eval=True):
        """
        测试进化出的机器人设计
        
        参数:
            urdf_path: URDF文件路径
            robot_id: 机器人ID
            algorithm: 强化学习算法
            train_timesteps: 训练步数
            n_eval_episodes: 评估轮数
            terrain_type: 地形类型
            render_eval: 是否可视化评估
        
        返回:
            评估结果
        """
        print(f"\n开始测试机器人 {robot_id} 使用算法 {algorithm}")
        
        # 创建训练环境 (不渲染)
        train_env = self.create_test_environment(
            urdf_path=urdf_path,
            render_mode="direct",
            terrain_type=terrain_type
        )
        
        # 训练控制器
        print(f"\n开始训练控制器...")
        model = self.train_controller(
            train_env,
            robot_id,
            algorithm=algorithm,
            total_timesteps=train_timesteps
        )
        
        # 创建评估环境 (可视化)
        eval_env = self.create_test_environment(
            urdf_path=urdf_path,
            render_mode="human" if render_eval else "direct",
            terrain_type=terrain_type
        )
        
        # 评估控制器
        print(f"\n开始评估控制器...")
        results = self.evaluate_controller(
            eval_env,
            model,
            n_eval_episodes=n_eval_episodes
        )
        
        # 记录性能
        self.performance_history[robot_id] = results
        
        # 保存性能结果
        results_path = os.path.join(self.results_dir, f"results_{robot_id}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n测试完成，结果已保存到 {results_path}")
        
        # 关闭环境
        train_env.close()
        eval_env.close()
        
        return results
    
    def compare_robots(self, robot_ids=None, metrics=None):
        """
        比较多个机器人设计的性能
        
        参数:
            robot_ids: 要比较的机器人ID列表
            metrics: 要比较的指标列表
        
        返回:
            比较结果
        """
        if not self.performance_history:
            print("没有可比较的性能数据")
            return
        
        # 如果未指定，使用所有机器人
        if robot_ids is None:
            robot_ids = list(self.performance_history.keys())
        
        # 如果未指定，使用所有指标
        if metrics is None:
            metrics = list(next(iter(self.performance_history.values())).keys())
        
        # 创建比较结果
        comparison = {}
        for metric in metrics:
            comparison[metric] = {}
            for robot_id in robot_ids:
                if robot_id in self.performance_history and metric in self.performance_history[robot_id]:
                    comparison[metric][robot_id] = self.performance_history[robot_id][metric]
        
        # 可视化比较
        self.visualize_comparison(comparison)
        
        return comparison
    
    def visualize_comparison(self, comparison):
        """
        可视化比较结果
        
        参数:
            comparison: 比较结果
        """
        n_metrics = len(comparison)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric, values) in enumerate(comparison.items()):
            robot_ids = list(values.keys())
            metric_values = list(values.values())
            
            # 创建条形图
            ax = axes[i]
            bars = ax.bar(robot_ids, metric_values, color='skyblue')
            ax.set_title(f'{metric}')
            ax.set_ylabel('值')
            ax.set_xlabel('机器人ID')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "robots_comparison.png"))
        plt.show()
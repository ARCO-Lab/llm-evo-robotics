#!/usr/bin/env python3
"""
真实多关节 Reacher 环境生成器
基于 GPT-5 建议：创建真实的 N 关节 MuJoCo 环境
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import tempfile

# ============================================================================
# 🧩 参数化 MuJoCo XML 生成器
# ============================================================================

def generate_multi_joint_reacher_xml(num_joints: int = 3, 
                                    link_lengths: list = None,
                                    link_masses: list = None) -> str:
    """
    生成 N 关节 Reacher 的 MuJoCo XML
    
    Args:
        num_joints: 关节数量
        link_lengths: 每个 link 的长度
        link_masses: 每个 link 的质量
    
    Returns:
        XML 字符串
    """
    if link_lengths is None:
        link_lengths = [0.1] * num_joints
    if link_masses is None:
        link_masses = [0.1] * num_joints
    
    # 确保长度匹配
    while len(link_lengths) < num_joints:
        link_lengths.append(0.1)
    while len(link_masses) < num_joints:
        link_masses.append(0.1)
    
    xml_template = f"""
<mujoco model="reacher_{num_joints}joint">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom contype="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="ground1" rgba="0.9 0.9 0.9 1" size=".02" type="capsule"/>
    <geom contype="0" fromto="-.3 .3 .01 .3 .3 .01" name="ground2" rgba="0.9 0.9 0.9 1" size=".02" type="capsule"/>
    <geom contype="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="ground3" rgba="0.9 0.9 0.9 1" size=".02" type="capsule"/>
    <geom contype="0" fromto=".3 -.3 .01 .3 .3 .01" name="ground4" rgba="0.9 0.9 0.9 1" size=".02" type="capsule"/>
    
    <!-- Base -->
    <body name="base" pos="0 0 0.01">
      <geom name="base_geom" pos="0 0 0" rgba="0.9 0.4 0.6 1" size=".02" type="sphere"/>
"""
    
    # 生成关节链
    current_body = "base"
    total_length = 0
    
    for i in range(num_joints):
        joint_name = f"joint{i+1}"
        link_name = f"link{i+1}"
        length = link_lengths[i]
        mass = link_masses[i]
        total_length += length
        
        xml_template += f"""
      <!-- Joint {i+1} -->
      <joint axis="0 0 1" limited="true" name="{joint_name}" pos="0 0 0" range="-3.14159 3.14159" type="hinge"/>
      <geom fromto="0 0 0 {length} 0 0" name="{link_name}_geom" rgba="0.7 0.7 0 1" size=".02" type="capsule"/>
      
      <body name="{link_name}" pos="{length} 0 0">
        <geom name="{link_name}_tip" pos="0 0 0" rgba="0.9 0.4 0.6 1" size=".01" type="sphere"/>
"""
        
        if i < num_joints - 1:
            # 不是最后一个关节，继续嵌套
            pass
        else:
            # 最后一个关节，添加末端执行器
            xml_template += f"""
        <!-- End Effector -->
        <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.3 0.9 0.3 1" size=".01" type="sphere"/>
"""
    
    # 关闭所有 body 标签
    for i in range(num_joints):
        xml_template += "      </body>\n"
    
    xml_template += """    </body>
    
    <!-- Target -->
    <body name="target" pos="0.1 0.1 0.01">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="target_x" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="target_y" pos="0 0 0" stiffness="0" type="slide"/>
      <geom contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".02" type="sphere"/>
    </body>
  </worldbody>
  
  <actuator>
"""
    
    # 生成执行器
    for i in range(num_joints):
        joint_name = f"joint{i+1}"
        xml_template += f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="{joint_name}" name="{joint_name}_motor"/>\n'
    
    xml_template += """  </actuator>
</mujoco>
"""
    
    return xml_template

# ============================================================================
# 🧩 真实多关节 Reacher 环境
# ============================================================================

class RealMultiJointReacherEnv(MujocoEnv):
    """
    真实的 N 关节 Reacher 环境
    基于 MuJoCo 物理引擎的真实多关节动力学
    """
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    
    def __init__(self, num_joints: int = 3, 
                 link_lengths: list = None,
                 link_masses: list = None,
                 render_mode: str = None,
                 **kwargs):
        
        self.num_joints = num_joints
        self.link_lengths = link_lengths or [0.1] * num_joints
        self.link_masses = link_masses or [0.1] * num_joints
        
        print(f"🌟 RealMultiJointReacherEnv 初始化:")
        print(f"   关节数: {num_joints}")
        print(f"   Link 长度: {self.link_lengths}")
        print(f"   Link 质量: {self.link_masses}")
        
        # 生成 XML
        xml_content = generate_multi_joint_reacher_xml(
            num_joints=num_joints,
            link_lengths=self.link_lengths,
            link_masses=self.link_masses
        )
        
        # 创建临时 XML 文件
        self.temp_xml_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False
        )
        self.temp_xml_file.write(xml_content)
        self.temp_xml_file.close()
        
        print(f"   XML 文件: {self.temp_xml_file.name}")
        
        # 设置观察空间和动作空间
        # 观察空间：[cos, sin] × num_joints + [vel] × num_joints + [ee_pos, target_pos, target_vec]
        # = 2*num_joints + num_joints + 2 + 2 + 2 = 3*num_joints + 6
        obs_dim = 3 * num_joints + 6
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        # 初始化 MujocoEnv
        MujocoEnv.__init__(
            self,
            model_path=self.temp_xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )
        
        # 动作空间：num_joints 维 (手动设置，因为 MujocoEnv 会从 XML 推断)
        self.action_space = Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        
        print(f"✅ RealMultiJointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def step(self, action):
        """执行动作"""
        # 确保动作维度正确
        if len(action) != self.num_joints:
            raise ValueError(f"Action dimension {len(action)} != num_joints {self.num_joints}")
        
        # 执行动作
        self.do_simulation(action, self.frame_skip)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._get_reward()
        
        # 检查是否结束
        # 成功条件：到达目标
        distance_to_target = self._get_distance_to_target()
        terminated = distance_to_target < 0.05  # 5cm 内算成功
        
        # 截断条件：超过最大步数 (类似标准 Reacher-v5 的 50 步)
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        max_episode_steps = 50  # 与标准 Reacher-v5 保持一致
        truncated = self.step_count >= max_episode_steps
        
        info = {
            'num_joints': self.num_joints,
            'link_lengths': self.link_lengths,
            'end_effector_pos': self._get_end_effector_pos(),
            'target_pos': self._get_target_pos(),
            'distance_to_target': self._get_distance_to_target()
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        """重置模型"""
        # 重置步数计数器
        self.step_count = 0
        
        # 随机初始化关节角度
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # 关节角度：小范围随机
        for i in range(self.num_joints):
            qpos[i] = self.np_random.uniform(low=-0.1, high=0.1)
            qvel[i] = self.np_random.uniform(low=-0.1, high=0.1)
        
        # 目标位置：在可达范围内随机
        total_reach = sum(self.link_lengths)
        target_distance = self.np_random.uniform(0.1, total_reach * 0.8)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        qpos[-2] = target_x  # target_x
        qpos[-1] = target_y  # target_y
        
        self.set_state(qpos, qvel)
        
        return self._get_obs()
    
    def _get_obs(self):
        """获取观察"""
        # 关节角度 (cos, sin)
        joint_angles = self.data.qpos[:self.num_joints]
        cos_angles = np.cos(joint_angles)
        sin_angles = np.sin(joint_angles)
        
        # 关节速度
        joint_velocities = self.data.qvel[:self.num_joints]
        
        # 末端执行器位置
        end_effector_pos = self._get_end_effector_pos()
        
        # 目标位置
        target_pos = self._get_target_pos()
        
        # 目标向量 (从末端执行器到目标)
        target_vec = target_pos - end_effector_pos
        
        # 组合观察
        obs = np.concatenate([
            cos_angles,           # num_joints
            sin_angles,           # num_joints  
            joint_velocities,     # num_joints
            end_effector_pos,     # 2
            target_pos,           # 2
            target_vec            # 2
        ])
        
        return obs
    
    def _get_end_effector_pos(self):
        """获取末端执行器位置"""
        # 获取最后一个 link 的末端位置
        fingertip_id = self.model.geom('fingertip').id
        return self.data.geom_xpos[fingertip_id][:2]  # 只要 x, y
    
    def _get_target_pos(self):
        """获取目标位置"""
        target_id = self.model.body('target').id
        return self.data.xpos[target_id][:2]  # 只要 x, y
    
    def _get_distance_to_target(self):
        """获取到目标的距离"""
        end_effector_pos = self._get_end_effector_pos()
        target_pos = self._get_target_pos()
        return np.linalg.norm(end_effector_pos - target_pos)
    
    def _get_reward(self):
        """计算奖励"""
        # 距离奖励
        distance = self._get_distance_to_target()
        distance_reward = -distance
        
        # 到达奖励
        reach_reward = 0
        if distance < 0.05:  # 5cm 内算成功
            reach_reward = 10
        
        # 控制惩罚
        action_penalty = -0.1 * np.sum(np.square(self.data.ctrl))
        
        # 速度惩罚 (避免过快运动)
        velocity_penalty = -0.01 * np.sum(np.square(self.data.qvel[:self.num_joints]))
        
        total_reward = distance_reward + reach_reward + action_penalty + velocity_penalty
        
        return total_reward
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'temp_xml_file') and os.path.exists(self.temp_xml_file.name):
            os.unlink(self.temp_xml_file.name)

# ============================================================================
# 🧩 真实多关节环境包装器
# ============================================================================

class RealMultiJointWrapper(gym.Wrapper):
    """
    真实多关节环境包装器
    只做编解码，不造数据
    """
    
    def __init__(self, num_joints: int = 3, 
                 link_lengths: list = None,
                 link_masses: list = None,
                 render_mode: str = None):
        
        # 创建真实的多关节环境
        env = RealMultiJointReacherEnv(
            num_joints=num_joints,
            link_lengths=link_lengths,
            link_masses=link_masses,
            render_mode=render_mode
        )
        
        super(RealMultiJointWrapper, self).__init__(env)
        
        self.num_joints = num_joints
        self.link_lengths = link_lengths or [0.1] * num_joints
        
        # 重新定义观察空间为统一格式
        # [joint_features, global_features]
        # joint_features: [cos, sin, vel, link_length] × num_joints
        # global_features: [ee_x, ee_y, target_x, target_y, target_vec_x, target_vec_y]
        obs_dim = num_joints * 4 + 6
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"🌐 RealMultiJointWrapper 初始化:")
        print(f"   关节数: {num_joints}")
        print(f"   Link 长度: {self.link_lengths}")
        print(f"   原始观察空间: {env.observation_space}")
        print(f"   包装后观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        将原始观察转换为统一格式
        
        原始格式: [cos×N, sin×N, vel×N, ee_pos×2, target_pos×2, target_vec×2]
        统一格式: [joint_features×N, global_features×6]
        joint_features: [cos, sin, vel, link_length]
        """
        # 解析原始观察
        cos_angles = obs[:self.num_joints]
        sin_angles = obs[self.num_joints:2*self.num_joints]
        velocities = obs[2*self.num_joints:3*self.num_joints]
        ee_pos = obs[3*self.num_joints:3*self.num_joints+2]
        target_pos = obs[3*self.num_joints+2:3*self.num_joints+4]
        target_vec = obs[3*self.num_joints+4:3*self.num_joints+6]
        
        # 构造关节特征
        joint_features = []
        for i in range(self.num_joints):
            joint_feature = [
                cos_angles[i],
                sin_angles[i], 
                velocities[i],
                self.link_lengths[i]
            ]
            joint_features.extend(joint_feature)
        
        # 全局特征
        global_features = np.concatenate([ee_pos, target_pos, target_vec])
        
        # 组合
        transformed_obs = np.array(joint_features + global_features.tolist(), dtype=np.float32)
        
        return transformed_obs
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        # 转换观察
        transformed_obs = self._transform_observation(obs)
        
        # 添加关节信息
        info['num_joints'] = self.num_joints
        info['link_lengths'] = self.link_lengths
        info['is_real_multi_joint'] = True
        
        return transformed_obs, info
    
    def step(self, action):
        """执行动作"""
        # 直接传递动作 (不做任何转换)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 转换观察
        transformed_obs = self._transform_observation(obs)
        
        # 添加关节信息
        info['num_joints'] = self.num_joints
        info['link_lengths'] = self.link_lengths
        info['is_real_multi_joint'] = True
        
        return transformed_obs, reward, terminated, truncated, info

# ============================================================================
# 🧩 测试函数
# ============================================================================

def test_real_multi_joint_env():
    """测试真实多关节环境"""
    print("🧪 测试真实多关节环境")
    
    # 测试不同关节数
    for num_joints in [2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"🔧 测试 {num_joints} 关节环境")
        print(f"{'='*50}")
        
        try:
            # 创建环境
            env = RealMultiJointWrapper(
                num_joints=num_joints,
                link_lengths=[0.1] * num_joints,
                render_mode=None
            )
            
            print(f"✅ {num_joints} 关节环境创建成功")
            print(f"   观察空间: {env.observation_space}")
            print(f"   动作空间: {env.action_space}")
            
            # 测试重置
            obs, info = env.reset()
            print(f"   重置观察维度: {obs.shape}")
            print(f"   是否真实多关节: {info.get('is_real_multi_joint', False)}")
            
            # 测试几步
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"   Step {step+1}: 动作维度={len(action)}, 奖励={reward:.3f}, 距离={info['distance_to_target']:.3f}")
            
            env.close()
            print(f"✅ {num_joints} 关节环境测试完成")
            
        except Exception as e:
            print(f"❌ {num_joints} 关节环境测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🌟 真实多关节 Reacher 环境测试")
    print("💡 基于 GPT-5 建议：真实的 N 关节 MuJoCo 动力学")
    print()
    
    test_real_multi_joint_env()

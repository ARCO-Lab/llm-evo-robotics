#!/usr/bin/env python3
"""
基于标准MuJoCo Reacher的自然3关节版本
只在原有2关节基础上添加一个关节和link
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
import tempfile

# ============================================================================
# 🧩 自然3关节Reacher XML生成器
# ============================================================================

def generate_natural_3joint_reacher_xml() -> str:
    """
    基于标准MuJoCo Reacher生成自然的3关节版本
    保持原有的场地大小和视觉效果，只添加一个关节
    """
    
    xml_content = """<mujoco model="reacher3joint">
    <compiler angle="radian" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
    </default>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
    <worldbody>
        <!-- Arena (与标准Reacher相同，但稍微大一点) -->
        <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1.5 1.5 10" type="plane"/>
        <geom conaffinity="0" fromto="-.4 -.4 .01 .4 -.4 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
        <geom conaffinity="0" fromto=" .4 -.4 .01 .4  .4 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
        <geom conaffinity="0" fromto="-.4  .4 .01 .4  .4 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
        <geom conaffinity="0" fromto="-.4 -.4 .01 -.4 .4 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
        
        <!-- Root -->
        <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
        
        <!-- 3关节机械臂 -->
        <body name="body0" pos="0 0 .01">
            <!-- 第一个link：稍微短一点以适应3关节 -->
            <geom fromto="0 0 0 0.08 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
            <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
            
            <body name="body1" pos="0.08 0 0">
                <!-- 第二个link -->
                <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
                <geom fromto="0 0 0 0.08 0 0" name="link1" rgba="0.0 0.6 0.4 1" size=".01" type="capsule"/>
                
                <body name="body2" pos="0.08 0 0">
                    <!-- 第三个link (新增) -->
                    <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
                    <geom fromto="0 0 0 0.08 0 0" name="link2" rgba="0.6 0.0 0.4 1" size=".01" type="capsule"/>
                    
                    <!-- 末端执行器 -->
                    <body name="fingertip" pos="0.09 0 0">
                        <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
                    </body>
                </body>
            </body>
        </body>
        
        <!-- Target (与标准Reacher相同，但范围稍大) -->
        <body name="target" pos=".15 -.15 .01">
            <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.35 .35" ref=".15" stiffness="0" type="slide"/>
            <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.35 .35" ref="-.15" stiffness="0" type="slide"/>
            <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
        </body>
    </worldbody>
    <actuator>
        <!-- 3个电机对应3个关节 -->
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
    </actuator>
</mujoco>"""
    
    return xml_content

# ============================================================================
# 🧩 自然3关节Reacher环境
# ============================================================================

class Natural3JointReacherEnv(MujocoEnv):
    """
    自然的3关节Reacher环境
    基于标准MuJoCo Reacher，保持相同的观察空间格式
    """
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    
    def __init__(self, render_mode: str = None, **kwargs):
        
        print(f"🌟 Natural3JointReacherEnv 初始化")
        
        # 生成3关节XML
        xml_content = generate_natural_3joint_reacher_xml()
        
        # 创建临时XML文件
        self.temp_xml_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', delete=False
        )
        self.temp_xml_file.write(xml_content)
        self.temp_xml_file.close()
        
        print(f"   XML文件: {self.temp_xml_file.name}")
        
        # 观察空间：与标准Reacher类似，但是3个关节
        # [cos0, cos1, cos2, sin0, sin1, sin2, vel0, vel1, vel2, ee_x, ee_y, target_x, target_y, vec_x, vec_y]
        # = 3 + 3 + 3 + 2 + 2 + 2 = 15维
        observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        
        # 初始化MujocoEnv
        MujocoEnv.__init__(
            self,
            model_path=self.temp_xml_file.name,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )
        
        # 动作空间：3维
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        print(f"✅ Natural3JointReacherEnv 创建完成")
        print(f"   观察空间: {self.observation_space}")
        print(f"   动作空间: {self.action_space}")
    
    def step(self, action):
        """执行动作"""
        # 确保动作维度正确
        if len(action) != 3:
            raise ValueError(f"Action dimension {len(action)} != 3")
        
        # 执行动作
        self.do_simulation(action, self.frame_skip)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励 (与标准Reacher相同的奖励函数)
        reward = self._get_reward()
        
        # 检查是否结束
        distance_to_target = self._get_distance_to_target()
        terminated = distance_to_target < 0.01  # 1cm内算成功
        
        # 截断条件：50步
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        truncated = self.step_count >= 50
        
        info = {
            'distance_to_target': distance_to_target,
            'end_effector_pos': self._get_end_effector_pos(),
            'target_pos': self._get_target_pos()
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        """重置模型"""
        # 重置步数计数器
        self.step_count = 0
        
        # 随机初始化关节角度 (小范围)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # 3个关节的随机初始角度
        qpos[0] = self.np_random.uniform(low=-0.1, high=0.1)  # joint0
        qpos[1] = self.np_random.uniform(low=-0.1, high=0.1)  # joint1  
        qpos[2] = self.np_random.uniform(low=-0.1, high=0.1)  # joint2
        
        # 关节速度
        qvel[0] = self.np_random.uniform(low=-0.005, high=0.005)
        qvel[1] = self.np_random.uniform(low=-0.005, high=0.005)
        qvel[2] = self.np_random.uniform(low=-0.005, high=0.005)
        
        # 随机目标位置 (在可达范围内)
        # 总臂长约为 0.08 + 0.08 + 0.08 = 0.24
        max_reach = 0.20  # 稍微保守一点
        target_distance = self.np_random.uniform(0.05, max_reach)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        qpos[-2] = target_x  # target_x
        qpos[-1] = target_y  # target_y
        
        self.set_state(qpos, qvel)
        
        return self._get_obs()
    
    def _get_obs(self):
        """获取观察 (与标准Reacher格式兼容)"""
        # 关节角度的cos和sin
        cos_angles = np.cos(self.data.qpos[:3])  # 3个关节
        sin_angles = np.sin(self.data.qpos[:3])
        
        # 关节速度
        joint_velocities = self.data.qvel[:3]
        
        # 末端执行器位置
        fingertip_pos = self.data.body("fingertip").xpos[:2]
        
        # 目标位置
        target_pos = self.data.body("target").xpos[:2]
        
        # 从末端到目标的向量
        target_vector = target_pos - fingertip_pos
        
        # 组合观察 [cos0,cos1,cos2, sin0,sin1,sin2, vel0,vel1,vel2, ee_x,ee_y, target_x,target_y, vec_x,vec_y]
        obs = np.concatenate([
            cos_angles,      # 3维
            sin_angles,      # 3维  
            joint_velocities, # 3维
            fingertip_pos,   # 2维
            target_pos,      # 2维
            target_vector    # 2维
        ])
        
        return obs
    
    def _get_reward(self):
        """计算奖励 (与标准Reacher相同)"""
        # 距离奖励 (主要奖励)
        distance = self._get_distance_to_target()
        distance_reward = -distance
        
        # 控制惩罚 (鼓励平滑控制)
        ctrl_cost = -np.square(self.data.ctrl).sum()
        
        # 总奖励
        reward = distance_reward + 0.1 * ctrl_cost
        
        return reward
    
    def _get_distance_to_target(self):
        """计算到目标的距离"""
        fingertip_pos = self.data.body("fingertip").xpos[:2]
        target_pos = self.data.body("target").xpos[:2]
        return np.linalg.norm(fingertip_pos - target_pos)
    
    def _get_end_effector_pos(self):
        """获取末端执行器位置"""
        return self.data.body("fingertip").xpos[:2]
    
    def _get_target_pos(self):
        """获取目标位置"""
        return self.data.body("target").xpos[:2]
    
    def __del__(self):
        """清理临时文件"""
        if hasattr(self, 'temp_xml_file') and os.path.exists(self.temp_xml_file.name):
            os.unlink(self.temp_xml_file.name)

# ============================================================================
# 🧩 测试函数
# ============================================================================

def test_natural_3joint_env():
    """测试自然3关节环境"""
    print("🧪 测试自然3关节Reacher环境")
    
    # 创建环境
    env = Natural3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建成功")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 测试几个episodes
    for episode in range(3):
        obs, info = env.reset()
        print(f"\n📍 Episode {episode + 1}")
        print(f"   初始观察形状: {obs.shape}")
        
        episode_reward = 0
        for step in range(50):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            distance = info['distance_to_target']
            
            if step % 10 == 0:
                print(f"   Step {step}: 距离={distance:.3f}, 奖励={reward:.3f}")
            
            # 稍微慢一点让您观察
            import time
            time.sleep(0.1)
            
            if terminated or truncated:
                break
        
        print(f"   Episode结束: 总奖励={episode_reward:.3f}, 最终距离={distance:.3f}")
    
    env.close()
    print("✅ 测试完成")

if __name__ == "__main__":
    test_natural_3joint_env()



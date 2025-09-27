#!/usr/bin/env python3
"""
深入调试MuJoCo渲染问题
找出为什么自定义环境的FPS异常高
"""

import gymnasium as gym
import numpy as np
import time
from simple_3joint_reacher import Simple3JointReacherEnv

def compare_environments():
    """对比标准Reacher和自定义3关节环境的内部配置"""
    print("🔍 对比环境内部配置")
    
    # 创建标准Reacher
    print("\n📊 标准Reacher-v5:")
    standard_env = gym.make('Reacher-v5', render_mode='human')
    obs, info = standard_env.reset()
    
    # 获取真正的MuJoCo环境（去掉包装器）
    unwrapped_env = standard_env.unwrapped
    
    print(f"   包装类型: {type(standard_env)}")
    print(f"   真实类型: {type(unwrapped_env)}")
    print(f"   frame_skip: {getattr(unwrapped_env, 'frame_skip', 'N/A')}")
    print(f"   dt: {getattr(unwrapped_env, 'dt', 'N/A')}")
    if hasattr(unwrapped_env, 'model'):
        print(f"   model.opt.timestep: {getattr(unwrapped_env.model.opt, 'timestep', 'N/A')}")
    print(f"   render_mode: {getattr(unwrapped_env, 'render_mode', 'N/A')}")
    
    if hasattr(unwrapped_env, 'viewer'):
        print(f"   viewer: {type(unwrapped_env.viewer) if unwrapped_env.viewer else None}")
    
    # 测试一步的时间
    start_time = time.time()
    for _ in range(10):
        action = standard_env.action_space.sample()
        obs, reward, terminated, truncated, info = standard_env.step(action)
        if terminated or truncated:
            obs, info = standard_env.reset()
    step_time = (time.time() - start_time) / 10
    print(f"   平均步时间: {step_time*1000:.1f}ms")
    
    standard_env.close()
    
    # 创建自定义3关节
    print("\n📊 自定义3关节:")
    custom_env = Simple3JointReacherEnv(render_mode='human')
    obs, info = custom_env.reset()
    
    print(f"   类型: {type(custom_env)}")
    print(f"   frame_skip: {getattr(custom_env, 'frame_skip', 'N/A')}")
    print(f"   dt: {getattr(custom_env, 'dt', 'N/A')}")
    print(f"   model.opt.timestep: {getattr(custom_env.model.opt, 'timestep', 'N/A')}")
    print(f"   render_mode: {getattr(custom_env, 'render_mode', 'N/A')}")
    
    if hasattr(custom_env, 'viewer'):
        print(f"   viewer: {type(custom_env.viewer) if custom_env.viewer else None}")
    
    # 测试一步的时间
    start_time = time.time()
    for _ in range(10):
        action = custom_env.action_space.sample()
        obs, reward, terminated, truncated, info = custom_env.step(action)
        if terminated or truncated:
            obs, info = custom_env.reset()
    step_time = (time.time() - start_time) / 10
    print(f"   平均步时间: {step_time*1000:.1f}ms")
    
    custom_env.close()

def test_render_methods():
    """测试不同的渲染方法"""
    print("\n🎮 测试渲染方法")
    
    # 测试标准Reacher的渲染
    print("\n📊 标准Reacher渲染:")
    standard_env = gym.make('Reacher-v5', render_mode='human')
    obs, info = standard_env.reset()
    
    # 手动调用render
    start_time = time.time()
    render_result = standard_env.render()
    render_time = time.time() - start_time
    print(f"   render()返回: {type(render_result)}")
    print(f"   render()时间: {render_time*1000:.1f}ms")
    
    # 检查viewer
    if hasattr(standard_env, 'mujoco_renderer'):
        print(f"   mujoco_renderer: {type(standard_env.mujoco_renderer)}")
        if hasattr(standard_env.mujoco_renderer, 'viewer'):
            print(f"   renderer.viewer: {type(standard_env.mujoco_renderer.viewer)}")
    
    standard_env.close()
    
    # 测试自定义3关节的渲染
    print("\n📊 自定义3关节渲染:")
    custom_env = Simple3JointReacherEnv(render_mode='human')
    obs, info = custom_env.reset()
    
    # 手动调用render
    start_time = time.time()
    render_result = custom_env.render()
    render_time = time.time() - start_time
    print(f"   render()返回: {type(render_result)}")
    print(f"   render()时间: {render_time*1000:.1f}ms")
    
    # 检查viewer
    if hasattr(custom_env, 'mujoco_renderer'):
        print(f"   mujoco_renderer: {type(custom_env.mujoco_renderer)}")
        if hasattr(custom_env.mujoco_renderer, 'viewer'):
            print(f"   renderer.viewer: {type(custom_env.mujoco_renderer.viewer)}")
    
    custom_env.close()

def create_fixed_3joint_reacher():
    """创建修复的3关节Reacher，强制使用与标准Reacher相同的配置"""
    print("\n🔧 创建修复的3关节Reacher")
    
    # 直接继承标准Reacher环境
    from gymnasium.envs.mujoco.reacher import ReacherEnv
    
    class Fixed3JointReacherEnv(ReacherEnv):
        """修复的3关节Reacher，继承标准Reacher的所有配置"""
        
        def __init__(self, **kwargs):
            # 创建3关节XML
            xml_content = """
<mujoco model="reacher">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.2 0.2 10" type="plane"/>
    <geom conaffinity="0" fromto="-.2 -.2 .01 .2 -.2 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto=" .2 -.2 .01 .2  .2 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.2  .2 .01 .2  .2 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" fromto="-.2 -.2 .01 -.2  .2 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
    <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
    <body name="body0" pos="0 0 .01">
      <geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
      <joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
      <body name="body1" pos="0.1 0 0">
        <joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
        <geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
        <body name="body2" pos="0.1 0 0">
          <joint axis="0 0 1" limited="true" name="joint2" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
          <geom fromto="0 0 0 0.1 0 0" name="link2" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
          <body name="fingertip" pos="0.1 0 0">
            <geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" pos=".1 -.1 .01">
      <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".1" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.1" stiffness="0" type="slide"/>
      <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint2"/>
  </actuator>
</mujoco>
"""
            
            # 保存到临时文件
            import tempfile
            import os
            self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
            self.xml_file.write(xml_content)
            self.xml_file.flush()
            
            # 调用父类初始化，但使用我们的XML
            # 这里需要修改model_path
            original_model_path = ReacherEnv.metadata['model_path']
            ReacherEnv.metadata['model_path'] = self.xml_file.name
            
            try:
                super().__init__(**kwargs)
            finally:
                # 恢复原始路径
                ReacherEnv.metadata['model_path'] = original_model_path
            
            # 修改动作空间为3维
            from gymnasium.spaces import Box
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        def _get_obs(self):
            """重写观察函数以支持3关节"""
            theta = self.data.qpos.flat[:3]  # 3个关节角度
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.data.qvel.flat[:3],  # 3个关节速度
                self.get_body_com("fingertip")[:2],  # 末端位置
                self.get_body_com("target")[:2],     # 目标位置
            ])
        
        def __del__(self):
            """清理临时文件"""
            if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
                os.unlink(self.xml_file.name)
    
    return Fixed3JointReacherEnv

def test_fixed_3joint():
    """测试修复的3关节环境"""
    print("\n🧪 测试修复的3关节环境")
    
    Fixed3JointReacherEnv = create_fixed_3joint_reacher()
    
    try:
        env = Fixed3JointReacherEnv(render_mode='human')
        print("✅ 修复的3关节环境创建成功")
        
        obs, info = env.reset()
        print(f"   观察空间: {obs.shape}")
        print(f"   动作空间: {env.action_space}")
        
        # 测试FPS
        print("🎯 测试FPS...")
        num_steps = 50
        start_time = time.time()
        
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        total_time = time.time() - start_time
        fps = num_steps / total_time
        
        print(f"   FPS: {fps:.1f}")
        print(f"   每步时间: {total_time/num_steps*1000:.1f}ms")
        
        if fps < 100:
            print("✅ FPS正常!")
        else:
            print("⚠️ FPS仍然异常")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🌟 深入调试MuJoCo渲染问题")
    print("💡 找出自定义环境FPS异常高的根本原因")
    print()
    
    try:
        # 1. 对比环境配置
        compare_environments()
        
        # 2. 测试渲染方法
        test_render_methods()
        
        # 3. 测试修复的3关节环境
        test_fixed_3joint()
        
        print(f"\n🎉 调试完成！")
        
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

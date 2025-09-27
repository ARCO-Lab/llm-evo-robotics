#!/usr/bin/env python3
"""
修复MuJoCo自定义环境渲染问题
基于网络搜索结果的解决方案
"""

import os
import tempfile
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import time

def test_rendering_backends():
    """测试不同的MuJoCo渲染后端"""
    print("🔧 测试不同的MuJoCo渲染后端")
    
    # 测试不同的渲染后端
    backends = ['glfw', 'egl', 'osmesa']
    
    # 简单的3关节XML
    simple_xml = """
<mujoco model="simple_3joint">
  <compiler angle="radian" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
  <worldbody>
    <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="0.2 0.2 10" type="plane"/>
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
    
    class TestEnv(MujocoEnv):
        def __init__(self, **kwargs):
            self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
            self.xml_file.write(simple_xml)
            self.xml_file.flush()
            
            observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
            
            super().__init__(
                self.xml_file.name,
                frame_skip=2,
                observation_space=observation_space,
                render_mode='human'
            )
            
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        def step(self, action):
            self.do_simulation(action, self.frame_skip)
            obs = self._get_obs()
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        def _get_obs(self):
            theta = self.data.qpos.flat[:3]
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.data.qvel.flat[:3],
                self.data.body("fingertip").xpos[:2],
                self.data.body("target").xpos[:2],
            ])
        
        def reset_model(self):
            qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
            self.set_state(qpos, qvel)
            return self._get_obs()
        
        def __del__(self):
            if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
                os.unlink(self.xml_file.name)
    
    results = {}
    
    for backend in backends:
        print(f"\n📊 测试渲染后端: {backend}")
        
        # 设置环境变量
        original_gl = os.environ.get('MUJOCO_GL', None)
        os.environ['MUJOCO_GL'] = backend
        
        try:
            env = TestEnv()
            print(f"   ✅ 环境创建成功")
            
            # 测试FPS
            obs, info = env.reset()
            
            start_time = time.time()
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
            
            total_time = time.time() - start_time
            fps = 20 / total_time
            
            print(f"   📈 FPS: {fps:.1f}")
            print(f"   每步时间: {total_time/20*1000:.1f}ms")
            
            if 20 <= fps <= 200:
                status = "✅ 正常"
            else:
                status = "⚠️ 异常"
            
            results[backend] = {
                'success': True,
                'fps': fps,
                'status': status
            }
            
            print(f"   状态: {status}")
            
            env.close()
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            results[backend] = {
                'success': False,
                'error': str(e)
            }
        
        # 恢复原始环境变量
        if original_gl is not None:
            os.environ['MUJOCO_GL'] = original_gl
        elif 'MUJOCO_GL' in os.environ:
            del os.environ['MUJOCO_GL']
    
    return results

def test_render_modes():
    """测试不同的渲染模式"""
    print("\n🎮 测试不同的渲染模式")
    
    # 使用最佳的渲染后端
    os.environ['MUJOCO_GL'] = 'glfw'
    
    simple_xml = """
<mujoco model="render_test">
  <worldbody>
    <geom name="floor" pos="0 0 -0.5" size="2 2 0.1" type="plane" rgba="0.8 0.9 0.8 1"/>
    <body name="ball" pos="0 0 1">
      <geom name="ball_geom" type="sphere" size="0.1" rgba="1 0 0 1"/>
      <joint name="ball_x" type="slide" axis="1 0 0"/>
      <joint name="ball_y" type="slide" axis="0 1 0"/>
      <joint name="ball_z" type="slide" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor_x" joint="ball_x" gear="1"/>
    <motor name="motor_y" joint="ball_y" gear="1"/>
  </actuator>
</mujoco>
"""
    
    class RenderTestEnv(MujocoEnv):
        def __init__(self, render_mode=None, **kwargs):
            self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
            self.xml_file.write(simple_xml)
            self.xml_file.flush()
            
            observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
            
            super().__init__(
                self.xml_file.name,
                frame_skip=2,
                observation_space=observation_space,
                render_mode=render_mode
            )
            
            self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        def step(self, action):
            self.do_simulation(action, self.frame_skip)
            obs = self._get_obs()
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        def _get_obs(self):
            return np.concatenate([
                self.data.qpos.flat,
                self.data.qvel.flat,
            ])[:6]
        
        def reset_model(self):
            qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
            self.set_state(qpos, qvel)
            return self._get_obs()
        
        def __del__(self):
            if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
                os.unlink(self.xml_file.name)
    
    render_modes = ['human', 'rgb_array', None]
    
    for mode in render_modes:
        print(f"\n📊 测试渲染模式: {mode}")
        
        try:
            env = RenderTestEnv(render_mode=mode)
            print(f"   ✅ 环境创建成功")
            
            obs, info = env.reset()
            
            # 测试渲染
            if mode == 'human':
                print("   🎯 测试human模式渲染...")
                start_time = time.time()
                for step in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    # 显式调用render
                    env.render()
                    if terminated or truncated:
                        obs, info = env.reset()
                
                total_time = time.time() - start_time
                fps = 10 / total_time
                print(f"   📈 FPS: {fps:.1f}")
                
            elif mode == 'rgb_array':
                print("   🎯 测试rgb_array模式...")
                start_time = time.time()
                for step in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    # 获取RGB图像
                    img = env.render()
                    if img is not None:
                        print(f"   图像尺寸: {img.shape}")
                    if terminated or truncated:
                        obs, info = env.reset()
                
                total_time = time.time() - start_time
                fps = 10 / total_time
                print(f"   📈 FPS: {fps:.1f}")
                
            else:
                print("   🎯 测试无渲染模式...")
                start_time = time.time()
                for step in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
                
                total_time = time.time() - start_time
                fps = 10 / total_time
                print(f"   📈 FPS: {fps:.1f}")
            
            env.close()
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")

def apply_best_solution():
    """应用最佳解决方案"""
    print("\n🚀 应用最佳解决方案")
    
    # 设置最佳渲染后端
    os.environ['MUJOCO_GL'] = 'glfw'
    os.environ['MUJOCO_RENDERER'] = 'glfw'
    
    print("✅ 设置环境变量:")
    print(f"   MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")
    print(f"   MUJOCO_RENDERER = {os.environ.get('MUJOCO_RENDERER')}")
    
    # 创建改进的3关节环境
    improved_xml = """
<mujoco model="improved_3joint">
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
    
    class ImprovedReacherEnv(MujocoEnv):
        def __init__(self, **kwargs):
            print("🌟 ImprovedReacherEnv 初始化")
            
            self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
            self.xml_file.write(improved_xml)
            self.xml_file.flush()
            
            print(f"   XML文件: {self.xml_file.name}")
            
            observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
            
            super().__init__(
                self.xml_file.name,
                frame_skip=2,
                observation_space=observation_space,
                render_mode='human'
            )
            
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            
            print("✅ ImprovedReacherEnv 创建完成")
            print(f"   观察空间: {self.observation_space}")
            print(f"   动作空间: {self.action_space}")
        
        def step(self, action):
            self.do_simulation(action, self.frame_skip)
            obs = self._get_obs()
            
            # 计算奖励
            vec = self.data.body("fingertip").xpos[:2] - self.data.body("target").xpos[:2]
            reward_dist = -np.linalg.norm(vec)
            reward_ctrl = -np.square(action).sum()
            reward = reward_dist + reward_ctrl
            
            terminated = False
            truncated = False
            info = {'reward_dist': reward_dist, 'reward_ctrl': reward_ctrl}
            
            return obs, reward, terminated, truncated, info
        
        def _get_obs(self):
            theta = self.data.qpos.flat[:3]
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.data.qvel.flat[:3],
                self.data.body("fingertip").xpos[:2],
                self.data.body("target").xpos[:2],
            ])
        
        def reset_model(self):
            qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
            self.set_state(qpos, qvel)
            return self._get_obs()
        
        def __del__(self):
            if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
                os.unlink(self.xml_file.name)
    
    # 测试改进的环境
    print("\n🧪 测试改进的3关节环境:")
    
    try:
        env = ImprovedReacherEnv()
        
        # 测试FPS
        obs, info = env.reset()
        
        print("🎯 测试FPS (30步)...")
        start_time = time.time()
        
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 显式调用render
            env.render()
            
            if step % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"   Step {step}: FPS = {current_fps:.1f}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        total_time = time.time() - start_time
        fps = 30 / total_time
        
        print(f"\n📈 最终结果:")
        print(f"   平均FPS: {fps:.1f}")
        print(f"   每步时间: {total_time/30*1000:.1f}ms")
        
        if 20 <= fps <= 200:
            print("   ✅ FPS正常!")
            success = True
        else:
            print("   ⚠️ FPS仍然异常")
            success = False
        
        env.close()
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🌟 修复MuJoCo自定义环境渲染问题")
    print("💡 基于网络搜索结果的解决方案")
    print()
    
    try:
        # 1. 测试不同的渲染后端
        backend_results = test_rendering_backends()
        
        # 2. 测试不同的渲染模式
        test_render_modes()
        
        # 3. 应用最佳解决方案
        success = apply_best_solution()
        
        # 4. 总结结果
        print("\n" + "="*60)
        print("📊 解决方案测试结果:")
        print("="*60)
        
        print("\n🔧 渲染后端测试:")
        for backend, result in backend_results.items():
            if result['success']:
                print(f"   {backend}: {result['status']} (FPS: {result['fps']:.1f})")
            else:
                print(f"   {backend}: ❌ 失败")
        
        print(f"\n🚀 最佳解决方案:")
        if success:
            print("   ✅ 成功! 3关节环境渲染正常")
        else:
            print("   ⚠️ 仍有问题，需要进一步调试")
        
        print(f"\n🎉 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



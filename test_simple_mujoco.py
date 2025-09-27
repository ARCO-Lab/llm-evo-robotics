#!/usr/bin/env python3
"""
测试最简单的MuJoCo环境
确认基础MuJoCo功能是否正常工作
"""

import gymnasium as gym
import time
import numpy as np

def test_basic_mujoco_envs():
    """测试基础MuJoCo环境"""
    print("🌟 测试基础MuJoCo环境")
    
    # 测试几个简单的MuJoCo环境
    envs_to_test = [
        'Pendulum-v1',      # 最简单的连续控制环境
        'CartPole-v1',      # 经典控制环境
        'Reacher-v5',       # 我们已知正常工作的
        'HalfCheetah-v5',   # 稍复杂的MuJoCo环境
        'Hopper-v5',        # 另一个MuJoCo环境
    ]
    
    results = {}
    
    for env_name in envs_to_test:
        print(f"\n📊 测试 {env_name}:")
        
        try:
            # 创建环境
            env = gym.make(env_name, render_mode='human')
            
            print(f"   ✅ 环境创建成功")
            print(f"   观察空间: {env.observation_space}")
            print(f"   动作空间: {env.action_space}")
            
            # 重置环境
            obs, info = env.reset()
            print(f"   初始观察: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            
            # 测试FPS
            print(f"   🎯 测试FPS (20步)...")
            
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
            
            # 判断FPS是否正常
            if 20 <= fps <= 200:
                fps_status = "✅ 正常"
            else:
                fps_status = "⚠️ 异常"
            
            results[env_name] = {
                'success': True,
                'fps': fps,
                'fps_status': fps_status,
                'obs_space': str(env.observation_space),
                'action_space': str(env.action_space)
            }
            
            print(f"   状态: {fps_status}")
            
            env.close()
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            results[env_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def test_simplest_custom_mujoco():
    """测试最简单的自定义MuJoCo环境"""
    print("\n🔧 测试最简单的自定义MuJoCo环境")
    
    from gymnasium.envs.mujoco import MujocoEnv
    from gymnasium.spaces import Box
    import tempfile
    import os
    
    # 最简单的MuJoCo XML - 只有一个球
    simple_xml = """
<mujoco model="simple">
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
    
    class SimpleBallEnv(MujocoEnv):
        def __init__(self, **kwargs):
            # 创建临时XML文件
            self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
            self.xml_file.write(simple_xml)
            self.xml_file.flush()
            
            observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
            
            super().__init__(
                self.xml_file.name,
                frame_skip=2,
                observation_space=observation_space,
                render_mode='human'
            )
            
            self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        def step(self, action):
            self.do_simulation(action, self.frame_skip)
            obs = self._get_obs()
            reward = 0.0  # 简单奖励
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
        
        def _get_obs(self):
            return np.concatenate([
                self.data.qpos.flat,  # 位置
                self.data.qvel.flat,  # 速度
            ])[:6]  # 确保是6维
        
        def reset_model(self):
            qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv)
            self.set_state(qpos, qvel)
            return self._get_obs()
        
        def __del__(self):
            if hasattr(self, 'xml_file') and os.path.exists(self.xml_file.name):
                os.unlink(self.xml_file.name)
    
    try:
        print("   🌟 创建最简单的自定义MuJoCo环境...")
        env = SimpleBallEnv()
        
        print("   ✅ 环境创建成功")
        print(f"   观察空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        print(f"   执行器数量: {env.model.nu}")
        print(f"   关节数量: {env.model.nq}")
        
        # 重置环境
        obs, info = env.reset()
        print(f"   初始观察: {obs.shape}")
        
        # 测试FPS
        print(f"   🎯 测试FPS (20步)...")
        
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
            print("   ✅ FPS正常")
            custom_result = True
        else:
            print("   ⚠️ FPS异常")
            custom_result = False
        
        env.close()
        
        return custom_result
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🌟 测试最简单的MuJoCo环境")
    print("💡 确认MuJoCo基础功能是否正常工作")
    print()
    
    try:
        # 1. 测试标准MuJoCo环境
        results = test_basic_mujoco_envs()
        
        # 2. 测试自定义环境
        custom_ok = test_simplest_custom_mujoco()
        
        # 3. 总结结果
        print("\n" + "="*60)
        print("📊 测试结果总结:")
        print("="*60)
        
        print("\n🏷️ 标准环境:")
        for env_name, result in results.items():
            if result['success']:
                print(f"   {env_name}: {result['fps_status']} (FPS: {result['fps']:.1f})")
            else:
                print(f"   {env_name}: ❌ 失败 - {result['error']}")
        
        print(f"\n🔧 自定义环境:")
        if custom_ok:
            print("   SimpleBallEnv: ✅ 正常")
        else:
            print("   SimpleBallEnv: ⚠️ 异常")
        
        # 4. 分析
        print(f"\n🔍 分析:")
        normal_count = sum(1 for r in results.values() if r.get('success') and '✅' in r.get('fps_status', ''))
        total_count = len([r for r in results.values() if r.get('success')])
        
        print(f"   标准环境正常率: {normal_count}/{total_count}")
        
        if custom_ok:
            print("   自定义环境: 正常")
        else:
            print("   自定义环境: 异常 - 可能是系统级MuJoCo配置问题")
        
        print(f"\n🎉 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



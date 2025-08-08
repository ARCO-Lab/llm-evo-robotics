#!/usr/bin/env python3
"""
快速修复并测试可视化
"""
import sys
import os
import numpy as np
import pygame

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

def quick_fix_and_test():
    """快速修复配置问题并测试"""
    print("🛠️ 快速修复环境配置...")
    
    # 动态修改Reacher2DEnv类
from reacher2d_env import Reacher2DEnv
    
    # 保存原始__init__方法
    original_init = Reacher2DEnv.__init__
    
    def patched_init(self, num_links=3, link_lengths=None, render_mode=None, config_path=None):
        """修复版的__init__方法"""
        # 调用gym.Env的__init__
        import gym
        gym.Env.__init__(self)
        
        # 加载配置
        self.config = self._load_config(config_path)
        print(f"self.config: {self.config}")
        
        # 确保配置有默认值
        if not isinstance(self.config, dict):
            self.config = {}
            
        if "start" not in self.config:
            self.config["start"] = {"position": [300, 300]}
            print("⚠️ 使用默认start配置: [300, 300]")
            
        if "goal" not in self.config:
            self.config["goal"] = {"position": [250, 250], "radius": 10}
            print("⚠️ 使用默认goal配置: [250, 250]")
        
        self.anchor_point = self.config["start"]["position"]
        self.gym_api_version = "old"
        
        self.num_links = num_links
        if link_lengths is None:
            self.link_lengths = [60] * num_links
        else:
            assert len(link_lengths) == num_links
            self.link_lengths = link_lengths
        
        self.render_mode = render_mode
        self.goal_pos = np.array(self.config["goal"]["position"])
        self.dt = 1/60.0
        self.max_torque = 50.0
        
        # 定义action_space和observation_space
        from gym.spaces import Box
        self.action_space = Box(low=-self.max_torque, high=self.max_torque, 
                               shape=(self.num_links,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, 
                                   shape=(self.num_links * 2 + 2,), dtype=np.float32)
        
        # 初始化物理世界
        import pymunk
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 981.0)
        self.space.damping = 0.99
        self.obstacles = []
        self.bodies = []
        self.joints = []
        
        self._create_robot()
        
        # 初始化渲染
        self.screen = None
        self.clock = None
        self.draw_options = None
        
        if self.render_mode:
            self._init_rendering()
    
    # 应用补丁
    Reacher2DEnv.__init__ = patched_init
    
    print("✅ 环境补丁应用成功")
    
    # 现在测试环境
    try:
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 40, 30],
            render_mode="human",
            config_path=None
        )
        
        print("✅ 修复后环境创建成功！")
        print(f"   关节数: {env.num_links}")
        print(f"   锚点: {env.anchor_point}")
        print(f"   目标: {env.goal_pos}")
        print(f"   bodies: {len(env.bodies)}")
        
        # 测试基本功能
        obs = env.reset()
        print(f"✅ 重置成功，观察形状: {obs.shape}")
        
        action = np.random.uniform(-5, 5, env.num_links)
        obs, reward, done, info = env.step(action)
        print(f"✅ 步进成功，奖励: {reward:.3f}")
        
        # 开始可视化
        print("\n🎥 开始可视化演示（按ESC退出）...")
        
    running = True
    step_count = 0
    
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        print("🔄 环境重置")
            
            # 生成动作 - 正弦波运动
            t = step_count * 0.03
            action = np.array([
                np.sin(t) * 20,
                np.sin(t + 1.5) * 15,
                np.sin(t + 3) * 10,
                np.sin(t + 4.5) * 8
            ])
            
            # 执行步骤
            obs, reward, done, info = env.step(action)
            
            # 渲染
            env.render()
            
            # 打印信息
            if step_count % 100 == 0:
                print(f"步骤 {step_count}: 奖励 {reward:.3f}")
            
            step_count += 1
            
            if done:
                obs = env.reset()
                print("🔄 环境自动重置")
                
        env.close()
        print("✅ 可视化演示完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 快速修复和可视化测试")
    print("=" * 40)
    quick_fix_and_test()
#!/usr/bin/env python3
"""
基于Gymnasium的2D Reacher环境演示
使用MuJoCo物理引擎，解决PyMunk的约束和穿透问题
"""

import gymnasium as gym
import numpy as np
import pygame
import time

def test_mujoco_reacher():
    """测试MuJoCo Reacher环境"""
    try:
        # 创建MuJoCo Reacher环境
        env = gym.make('Reacher-v4', render_mode='human')
        print("✅ 成功创建MuJoCo Reacher-v4环境")
        
        # 重置环境
        observation, info = env.reset()
        print(f"📊 观察空间维度: {observation.shape}")
        print(f"🎮 动作空间: {env.action_space}")
        
        # 手动控制测试
        print("🎮 手动控制测试开始...")
        print("按键说明:")
        print("  W/S: 控制第一个关节")
        print("  A/D: 控制第二个关节") 
        print("  ESC: 退出")
        
        running = True
        step_count = 0
        
        while running and step_count < 1000:
            # 获取键盘输入
            action = np.array([0.0, 0.0])  # 两个关节的扭矩
            
            # 检查pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 获取当前按键状态
            keys = pygame.key.get_pressed()
            
            # 根据按键设置动作
            if keys[pygame.K_w]:
                action[0] = 1.0   # 第一个关节正向扭矩
            elif keys[pygame.K_s]:
                action[0] = -1.0  # 第一个关节反向扭矩
                
            if keys[pygame.K_a]:
                action[1] = -1.0  # 第二个关节反向扭矩
            elif keys[pygame.K_d]:
                action[1] = 1.0   # 第二个关节正向扭矩
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 渲染环境
            env.render()
            
            step_count += 1
            
            # 检查是否需要重置
            if terminated or truncated:
                print(f"🔄 环境重置 (步数: {step_count})")
                observation, info = env.reset()
                step_count = 0
            
            time.sleep(0.01)  # 控制帧率
        
        env.close()
        print("✅ MuJoCo Reacher测试完成")
        return True
        
    except ImportError as e:
        print(f"❌ MuJoCo未安装: {e}")
        return False
    except Exception as e:
        print(f"❌ MuJoCo Reacher测试失败: {e}")
        return False

def test_classic_control():
    """测试经典控制环境（不需要MuJoCo）"""
    try:
        # 尝试一些不需要MuJoCo的环境
        envs_to_try = [
            'Pendulum-v1',
            'CartPole-v1', 
            'Acrobot-v1'
        ]
        
        for env_name in envs_to_try:
            try:
                print(f"\n🧪 测试 {env_name}...")
                env = gym.make(env_name, render_mode='human')
                observation, info = env.reset()
                
                print(f"📊 观察空间: {observation.shape}")
                print(f"🎮 动作空间: {env.action_space}")
                
                # 运行几步
                for _ in range(100):
                    action = env.action_space.sample()  # 随机动作
                    observation, reward, terminated, truncated, info = env.step(action)
                    env.render()
                    
                    if terminated or truncated:
                        observation, info = env.reset()
                    
                    time.sleep(0.02)
                
                env.close()
                print(f"✅ {env_name} 测试成功")
                break
                
            except Exception as e:
                print(f"❌ {env_name} 测试失败: {e}")
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ 经典控制环境测试失败: {e}")
        return False

def create_custom_reacher_env():
    """创建自定义的2D Reacher环境"""
    print("\n🔨 创建自定义2D Reacher环境...")
    
    # 这里可以创建一个基于Gymnasium的自定义环境
    # 使用更稳定的物理引擎（如Box2D）
    
    try:
        # 尝试Box2D环境
        env = gym.make('LunarLander-v2', render_mode='human')
        print("✅ Box2D环境可用，可以基于此创建自定义Reacher")
        
        observation, info = env.reset()
        print(f"📊 观察空间: {observation.shape}")
        
        # 运行几步演示
        for _ in range(200):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                observation, info = env.reset()
            
            time.sleep(0.02)
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Box2D环境测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🤖 Gymnasium 2D Reacher 环境测试")
    print("=" * 50)
    
    # 初始化pygame（某些环境需要）
    pygame.init()
    
    # 1. 尝试MuJoCo Reacher
    print("\n1️⃣ 尝试MuJoCo Reacher环境...")
    mujoco_success = test_mujoco_reacher()
    
    if not mujoco_success:
        # 2. 尝试经典控制环境
        print("\n2️⃣ 尝试经典控制环境...")
        classic_success = test_classic_control()
        
        if not classic_success:
            # 3. 创建自定义环境
            print("\n3️⃣ 尝试Box2D环境（用于自定义Reacher）...")
            custom_success = create_custom_reacher_env()
    
    pygame.quit()
    
    print("\n" + "=" * 50)
    print("🎯 建议:")
    print("1. 如果MuJoCo可用，使用Reacher-v4环境（最稳定）")
    print("2. 如果需要自定义，可以基于Box2D创建2D Reacher")
    print("3. 也可以使用PyBullet创建自定义机器人环境")
    
    print("\n📦 安装MuJoCo的命令:")
    print("pip install mujoco")
    print("pip install gymnasium[mujoco]")

if __name__ == "__main__":
    main()

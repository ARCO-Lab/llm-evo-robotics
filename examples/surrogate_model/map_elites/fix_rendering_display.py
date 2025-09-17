#!/usr/bin/env python3
"""
修复reacher2d环境的实时渲染显示问题
"""

import sys
import os
import time

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '2d_reacher', 'envs'))

def test_pygame_display():
    """测试pygame显示功能"""
    print("🎮 测试pygame显示功能")
    
    try:
        import pygame
        
        # 初始化pygame
        pygame.init()
        
        # 创建窗口
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Reacher2D 渲染测试")
        clock = pygame.time.Clock()
        
        print("✅ pygame窗口创建成功")
        print("🎨 应该看到一个测试窗口...")
        
        # 运行简单的渲染循环
        for i in range(100):
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True
            
            # 清屏
            screen.fill((100, 150, 200))  # 蓝色背景
            
            # 绘制一个移动的圆
            x = 400 + 200 * (i % 50) / 50 - 100
            y = 300 + 100 * (i % 30) / 30 - 50
            pygame.draw.circle(screen, (255, 255, 0), (int(x), int(y)), 20)
            
            # 显示文本
            font = pygame.font.Font(None, 36)
            text = font.render(f"Test Frame {i}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            
            # 更新显示
            pygame.display.flip()
            clock.tick(10)  # 10 FPS
        
        pygame.quit()
        print("✅ pygame显示测试完成")
        return True
        
    except Exception as e:
        print(f"❌ pygame显示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_reacher2d_rendering():
    """修复reacher2d环境的渲染显示"""
    print("\n🤖 修复reacher2d环境渲染")
    
    try:
        from reacher2d_env import Reacher2DEnv
        import pygame
        
        print("🎨 创建强制显示窗口的reacher2d环境...")
        
        # 强制初始化pygame显示
        pygame.init()
        pygame.display.init()
        
        env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 40, 30], 
            render_mode='human'
        )
        
        # 确保窗口在前台
        if hasattr(env, 'screen') and env.screen:
            pygame.display.set_caption("Reacher2D - MAP-Elites Training")
            print("✅ 设置窗口标题")
        
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        print("🏃 开始训练循环 (应该显示实时窗口)...")
        
        for step in range(30):
            # 处理pygame事件（重要！）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("用户关闭窗口")
                    env.close()
                    return True
            
            # 执行动作
            action = env.action_space.sample()
            result = env.step(action)
            
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result
                truncated = False
            
            # 渲染 - 这应该显示实时窗口
            env.render()
            
            # 强制刷新显示
            pygame.display.flip()
            
            if step % 5 == 0:
                print(f"步骤 {step}: reward={reward:.3f} (窗口应该在显示)")
            
            # 稍微慢一点，让用户能看清楚
            time.sleep(0.2)
            
            if done or truncated:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                print("Episode结束，重置")
        
        print("✅ reacher2d渲染修复测试完成")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ reacher2d渲染修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_map_elites_test():
    """创建最小的MAP-Elites渲染测试"""
    print("\n🧬 最小MAP-Elites渲染测试")
    
    try:
        # 简化的训练适配器测试
        print("🔧 创建简化的训练适配器...")
        
        import argparse
        from enhanced_train_interface import MAPElitesTrainingInterface
        
        # 创建训练接口
        interface = MAPElitesTrainingInterface(
            enable_rendering=True,
            silent_mode=False
        )
        
        print("✅ 训练接口创建成功")
        
        # 创建模拟的训练参数
        training_args = argparse.Namespace()
        training_args.num_joints = 3
        training_args.link_lengths = [60.0, 40.0, 30.0]
        training_args.lr = 3e-4
        training_args.gamma = 0.99
        training_args.alpha = 0.2
        training_args.batch_size = 64
        training_args.buffer_capacity = 10000
        training_args.warmup_steps = 1000
        training_args.total_steps = 100  # 很短的训练
        training_args.save_dir = './test_minimal_rendering'
        training_args.seed = 42
        
        print("🚀 开始训练 (应该显示渲染窗口)...")
        
        # 这应该调用enhanced_train.py的subprocess，并显示渲染
        result = interface.train_individual(training_args)
        
        print(f"✅ 训练完成: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 最小MAP-Elites测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始渲染显示修复测试")
    print("=" * 60)
    
    # 1. 测试基本pygame显示
    if test_pygame_display():
        print("✅ pygame基础显示正常")
    else:
        print("❌ pygame基础显示有问题，可能是显示环境问题")
        exit(1)
    
    # 2. 测试reacher2d环境渲染
    if fix_reacher2d_rendering():
        print("✅ reacher2d环境渲染正常")
    else:
        print("❌ reacher2d环境渲染有问题")
    
    # 3. 测试MAP-Elites集成渲染
    if create_minimal_map_elites_test():
        print("✅ MAP-Elites集成渲染正常")
    else:
        print("❌ MAP-Elites集成渲染有问题")
    
    print("\n🎉 渲染修复测试完成")


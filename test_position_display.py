#!/usr/bin/env python3
"""
测试实时位置信息显示功能
"""

from baseline_complete_sequential_training import create_env
import time

def test_position_display():
    """测试位置信息显示功能"""
    print("🎯 测试3关节Reacher的实时位置信息显示")
    print("=" * 60)
    
    # 创建带位置信息显示的3关节环境
    env = create_env(3, render_mode='human', show_position_info=True)
    
    print("✅ 环境创建完成，开始测试...")
    print("📍 每10步会显示一次end-effector位置信息")
    print("🎮 使用随机动作进行测试 (按Ctrl+C停止)")
    print()
    
    try:
        obs, info = env.reset()
        episode_count = 1
        
        for step in range(200):  # 测试200步
            # 使用小幅度随机动作
            action = env.action_space.sample() * 0.2
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 每50步显示episode信息
            if step % 50 == 0 and step > 0:
                print(f"\n🔄 Episode {episode_count}, 总步数: {step}")
                print(f"   当前距离: {info.get('distance_to_target', 'N/A'):.4f}")
                print(f"   成功状态: {'✅' if info.get('is_success', False) else '❌'}")
            
            # 重置episode
            if terminated or truncated:
                print(f"\n🏁 Episode {episode_count} 结束，重置环境...")
                obs, info = env.reset()
                episode_count += 1
            
            # 控制速度
            time.sleep(0.05)
        
        print(f"\n✅ 测试完成！共运行了 {episode_count} 个episodes")
        
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("✅ 环境已关闭")

if __name__ == "__main__":
    test_position_display()


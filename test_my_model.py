#!/usr/bin/env python3
"""
🎯 模型性能测试脚本
用途: 测试训练好的reacher2d SAC模型
作者: Assistant
日期: 2025-08-28

使用方法:
    python test_my_model.py
    python test_my_model.py --episodes 10
    python test_my_model.py --no-render
"""

import torch
import numpy as np
import sys
import os
import argparse
import time
from pathlib import Path

# 设置数据类型避免类型错误
torch.set_default_dtype(torch.float32)

# 添加路径
base_dir = Path(__file__).parent
sys.path.append(str(base_dir))
sys.path.insert(0, str(base_dir / 'examples/2d_reacher/envs'))
sys.path.insert(0, str(base_dir / 'examples/surrogate_model/attn_model'))
sys.path.insert(0, str(base_dir / 'examples/surrogate_model/sac'))
sys.path.insert(0, str(base_dir / 'examples/2d_reacher/utils'))

# 导入环境
try:
    sys.path.append('/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
    from reacher2d_env import Reacher2DEnv
    print("✅ 环境导入成功")
except ImportError as e:
    print(f"❌ 无法导入 Reacher2DEnv: {e}")
    sys.exit(1)

# 导入模型组件
try:
    # 直接添加到系统路径
    import sys
    sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/surrogate_model/attn_model')
    sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/surrogate_model/sac')
    sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/utils')
    
    # 直接导入文件中的类
    from attn_model import AttnModel
    from sac_model import AttentionSACWithBuffer
    from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
    print("✅ 模型组件导入成功")
except ImportError as e:
    print(f"❌ 导入模型组件失败: {e}")
    print("📍 请确保在正确的虚拟环境中运行")
    sys.exit(1)


class ModelTester:
    def __init__(self):
        self.model_path = None
        self.env = None
        self.sac = None
        self.gnn_embed = None
        
    def setup_environment(self, render=True):
        """设置测试环境"""
        env_params = {
            'num_links': 4,
            'link_lengths': [80, 80, 80, 60],
            'render_mode': 'human' if render else None,
            'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
            'debug_level': 'SILENT'
        }
        
        try:
            self.env = Reacher2DEnv(**env_params)
            num_joints = self.env.action_space.shape[0]
            
            print(f"✅ 环境创建成功")
            print(f"   关节数: {num_joints}")
            print(f"   Action space: {self.env.action_space}")
            print(f"   Observation space: {self.env.observation_space}")
            
            return num_joints
            
        except Exception as e:
            print(f"❌ 环境创建失败: {e}")
            return None
    
    def setup_gnn_encoder(self, num_joints, link_lengths):
        """设置GNN编码器"""
        try:
            reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
            self.gnn_embed = reacher2d_encoder.get_gnn_embeds(
                num_links=num_joints, 
                link_lengths=link_lengths
            )
            print(f"✅ GNN编码器设置成功，嵌入形状: {self.gnn_embed.shape}")
            return True
            
        except Exception as e:
            print(f"❌ GNN编码器设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_model(self, num_joints):
        """设置SAC模型"""
        try:
            attn_model = AttnModel(128, 130, 130, 4)
            self.sac = AttentionSACWithBuffer(
                attn_model, 
                num_joints, 
                buffer_capacity=10000, 
                batch_size=64,
                lr=1e-5, 
                env_type='reacher2d'
            )
            print(f"✅ SAC模型创建成功")
            return True
            
        except Exception as e:
            print(f"❌ SAC模型创建失败: {e}")
            return False
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        try:
            model_data = torch.load(model_path, map_location='cpu')
            
            # 加载Actor权重
            if 'actor_state_dict' in model_data:
                self.sac.actor.load_state_dict(model_data['actor_state_dict'], strict=False)
                print("✅ Actor权重加载成功")
            else:
                print("⚠️ 未找到actor_state_dict")
            
            # 可选：加载Critic权重（用于更完整的测试）
            if 'critic1_state_dict' in model_data:
                self.sac.critic1.load_state_dict(model_data['critic1_state_dict'], strict=False)
                print("✅ Critic1权重加载成功")
            
            if 'critic2_state_dict' in model_data:
                self.sac.critic2.load_state_dict(model_data['critic2_state_dict'], strict=False)
                print("✅ Critic2权重加载成功")
            
            # 显示模型信息
            print(f"\n📋 模型信息:")
            print(f"   训练步数: {model_data.get('step', 'N/A')}")
            print(f"   最终成功率: {model_data.get('final_success_rate', 'N/A')}")
            print(f"   最终最小距离: {model_data.get('final_min_distance', 'N/A')}")
            print(f"   训练完成: {model_data.get('training_completed', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_action(self, obs, num_joints, deterministic=True):
        """获取模型动作"""
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = self.sac.get_action(
                    obs_tensor, 
                    self.gnn_embed, 
                    num_joints=num_joints, 
                    deterministic=deterministic
                )
                return action.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"❌ 动作获取失败: {e}")
            # 返回随机动作作为fallback
            return self.env.action_space.sample()
    
    def test_single_episode(self, episode_num, max_steps=500, goal_threshold=35.0, verbose=True):
        """测试单个episode"""
        obs = self.env.reset()
        episode_reward = 0
        step_count = 0
        min_distance = float('inf')
        success = False
        num_joints = self.env.action_space.shape[0]
        
        if verbose:
            print(f"\n--- Episode {episode_num} ---")
        
        while step_count < max_steps:
            # 获取动作
            action = self.get_action(obs, num_joints, deterministic=True)
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            step_count += 1
            
            # 计算到目标的距离
            end_pos = self.env._get_end_effector_position()
            goal_pos = self.env.goal_pos
            distance = np.linalg.norm(np.array(goal_pos) - np.array(end_pos))
            min_distance = min(min_distance, distance)
            
            # 检查是否成功
            if distance <= goal_threshold:
                if verbose:
                    print(f"🎉 成功到达目标! 距离: {distance:.1f} pixels")
                success = True
                break
            
            # 显示进度（每100步）
            if verbose and step_count % 100 == 0:
                print(f"    Step {step_count}: 距离={distance:.1f}, 奖励={reward:.3f}")
            
            if done:
                break
        
        if verbose:
            print(f"  📊 结果:")
            print(f"    总奖励: {episode_reward:.2f}")
            print(f"    最小距离: {min_distance:.1f} pixels")
            print(f"    步骤数: {step_count}")
            print(f"    成功: {'✅ 是' if success else '❌ 否'}")
        
        return {
            'reward': episode_reward,
            'min_distance': min_distance,
            'steps': step_count,
            'success': success
        }
    
    def run_test(self, num_episodes=5, render=True, goal_threshold=35.0):
        """运行完整测试"""
        print(f"🎯 开始测试模型性能")
        print(f"   模型路径: {self.model_path}")
        print(f"   测试Episodes: {num_episodes}")
        print(f"   目标阈值: {goal_threshold} pixels")
        print(f"   渲染: {render}")
        
        # 设置环境
        num_joints = self.setup_environment(render)
        if num_joints is None:
            return None
        
        # 设置GNN编码器
        if not self.setup_gnn_encoder(num_joints, [80, 80, 80, 60]):
            return None
        
        # 设置模型
        if not self.setup_model(num_joints):
            return None
        
        # 加载模型权重
        if not self.load_model(self.model_path):
            return None
        
        # 运行测试episodes
        results = []
        success_count = 0
        
        print(f"\n🎮 开始测试 {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            try:
                result = self.test_single_episode(
                    episode + 1, 
                    max_steps=500, 
                    goal_threshold=goal_threshold,
                    verbose=True
                )
                
                results.append(result)
                if result['success']:
                    success_count += 1
                    
            except KeyboardInterrupt:
                print(f"\n⚠️ 用户中断测试")
                break
            except Exception as e:
                print(f"❌ Episode {episode + 1} 测试失败: {e}")
                continue
        
        # 计算统计数据
        if not results:
            print("❌ 没有成功完成的测试")
            return None
        
        success_rate = success_count / len(results)
        avg_reward = np.mean([r['reward'] for r in results])
        avg_min_distance = np.mean([r['min_distance'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        # 显示结果
        print(f"\n{'='*60}")
        print(f"🏆 测试结果总结:")
        print(f"  完成Episodes: {len(results)}")
        print(f"  成功次数: {success_count}")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  平均奖励: {avg_reward:.2f}")
        print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
        print(f"  平均Episode长度: {avg_steps:.1f} steps")
        print(f"  目标阈值: {goal_threshold:.1f} pixels")
        
        # 性能评价
        print(f"\n📋 性能评价:")
        if success_rate >= 0.8:
            print(f"  🏆 优秀! 成功率 >= 80%")
        elif success_rate >= 0.5:
            print(f"  👍 良好! 成功率 >= 50%")
        elif success_rate >= 0.2:
            print(f"  ⚠️  一般! 成功率 >= 20%")
        else:
            print(f"  ❌ 需要改进! 成功率 < 20%")
            
        if avg_min_distance <= goal_threshold:
            print(f"  ✅ 平均最小距离达到目标阈值")
        else:
            print(f"  ⚠️  平均最小距离超出目标阈值 {avg_min_distance - goal_threshold:.1f} pixels")
        
        print(f"{'='*60}")
        
        # 清理
        if self.env:
            self.env.close()
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_min_distance': avg_min_distance,
            'avg_steps': avg_steps,
            'success_count': success_count,
            'total_episodes': len(results),
            'results': results
        }


def main():
    parser = argparse.ArgumentParser(description='测试训练好的reacher2d模型')
    parser.add_argument('--model-path', type=str, 
                       default='trained_models/reacher2d/enhanced_test/08-28-2025-20-07-09/best_models/final_model_step_119999.pth',
                       help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=5,
                       help='测试episode数量')
    parser.add_argument('--no-render', action='store_true',
                       help='不显示渲染')
    parser.add_argument('--goal-threshold', type=float, default=35.0,
                       help='目标距离阈值')
    
    args = parser.parse_args()
    
    print(f"🚀 模型测试脚本启动")
    print(f"   模型路径: {args.model_path}")
    print(f"   Episodes: {args.episodes}")
    print(f"   渲染: {not args.no_render}")
    print(f"   目标阈值: {args.goal_threshold}")
    
    # 创建测试器
    tester = ModelTester()
    tester.model_path = args.model_path
    
    # 运行测试
    try:
        result = tester.run_test(
            num_episodes=args.episodes,
            render=not args.no_render,
            goal_threshold=args.goal_threshold
        )
        
        if result:
            print(f"\n🎯 快速结论:")
            if result['success_rate'] >= 0.8:
                print(f"  ✅ 模型表现优秀! 可以部署使用")
            elif result['success_rate'] >= 0.3:
                print(f"  ⚠️  模型表现一般，建议继续训练")
            else:
                print(f"  ❌ 模型表现较差，需要重新设计")
        else:
            print(f"❌ 测试失败")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

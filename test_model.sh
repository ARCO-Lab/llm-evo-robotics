#!/bin/bash

# 🎯 模型测试脚本
# 作者: Assistant
# 用途: 测试训练好的reacher2d模型

echo "🚀 开始测试训练好的模型..."

# 1. 进入虚拟环境
echo "🔧 激活虚拟环境..."
cd /home/xli149/Documents/repos/RoboGrammar
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ 虚拟环境激活失败!"
    exit 1
fi

echo "✅ 虚拟环境激活成功"

# 2. 切换到工作目录
cd /home/xli149/Documents/repos/test_robo/examples/surrogate_model

# 3. 设置模型路径
MODEL_PATH="../../trained_models/reacher2d/enhanced_test/08-28-2025-20-07-09/best_models/final_model_step_119999.pth"

# 4. 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 模型文件不存在: $MODEL_PATH"
    echo "📁 检查可用的模型文件..."
    ls -la ../../trained_models/reacher2d/enhanced_test/08-28-2025-20-07-09/best_models/
    exit 1
fi

echo "✅ 找到模型文件: $MODEL_PATH"

# 5. 创建Python测试脚本
cat > temp_model_test.py << 'EOF'
#!/usr/bin/env python3
import torch
import numpy as np
import sys
import os
import time

# 设置数据类型避免类型错误
torch.set_default_dtype(torch.float32)

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/attn_model'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/sac'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/utils'))

from reacher2d_env import Reacher2DEnv
from attn_model import AttnModel
from sac_model import AttentionSACWithBuffer
from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder

def test_model_performance(model_path, num_episodes=5, render=True):
    """测试模型性能"""
    print(f"🎯 测试模型: {model_path}")
    
    # 环境配置
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human' if render else None,
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
        'debug_level': 'SILENT'
    }
    
    try:
        # 创建环境
        env = Reacher2DEnv(**env_params)
        num_joints = env.action_space.shape[0]
        
        print(f"✅ 环境创建成功")
        print(f"   关节数: {num_joints}")
        print(f"   Action space: {env.action_space}")
        
        # 创建GNN编码器
        reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
        gnn_embed = reacher2d_encoder.get_gnn_embeds(
            num_links=num_joints, 
            link_lengths=env_params['link_lengths']
        )
        
        print(f"✅ GNN编码器创建成功")
        
        # 创建SAC模型
        attn_model = AttnModel(128, 130, 130, 4)
        sac = AttentionSACWithBuffer(attn_model, num_joints, 
                                   buffer_capacity=10000, batch_size=64,
                                   lr=1e-5, env_type='reacher2d')
        
        print(f"✅ SAC模型创建成功")
        
        # 加载模型
        if os.path.exists(model_path):
            model_data = torch.load(model_path, map_location='cpu')
            
            # 加载Actor权重
            if 'actor_state_dict' in model_data:
                sac.actor.load_state_dict(model_data['actor_state_dict'], strict=False)
                print("✅ Actor权重加载成功")
            
            # 显示模型信息
            print(f"📋 模型信息:")
            print(f"   训练步数: {model_data.get('step', 'N/A')}")
            print(f"   最终成功率: {model_data.get('final_success_rate', 'N/A')}")
            print(f"   最终最小距离: {model_data.get('final_min_distance', 'N/A')}")
            
        else:
            print(f"❌ 模型文件不存在: {model_path}")
            return None
        
        # 测试多个episode
        success_count = 0
        total_rewards = []
        min_distances = []
        episode_lengths = []
        goal_threshold = 35.0
        
        print(f"\n🎮 开始测试 {num_episodes} episodes")
        print(f"   目标阈值: {goal_threshold} pixels")
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = 500
            min_distance_this_episode = float('inf')
            episode_success = False
            
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            while step_count < max_steps:
                # 获取动作 (使用训练好的策略)
                with torch.no_grad():
                    action = sac.get_action(
                        torch.FloatTensor(obs).unsqueeze(0), 
                        gnn_embed, 
                        num_joints=num_joints, 
                        deterministic=True  # 确定性动作
                    )
                
                # 执行动作
                obs, reward, done, info = env.step(action.cpu().numpy().flatten())
                episode_reward += reward
                step_count += 1
                
                # 计算到目标的距离
                end_pos = env._get_end_effector_position()
                goal_pos = env.goal_pos
                distance = np.linalg.norm(np.array(goal_pos) - np.array(end_pos))
                min_distance_this_episode = min(min_distance_this_episode, distance)
                
                # 检查是否成功
                if distance <= goal_threshold:
                    print(f"🎉 成功到达目标! 距离: {distance:.1f} pixels")
                    episode_success = True
                    success_count += 1
                    break
                
                # 显示进度
                if step_count % 100 == 0:
                    print(f"    Step {step_count}: 距离={distance:.1f}, 奖励={reward:.2f}")
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            min_distances.append(min_distance_this_episode)
            episode_lengths.append(step_count)
            
            print(f"  📊 Episode {episode + 1} 结果:")
            print(f"    总奖励: {episode_reward:.2f}")
            print(f"    最小距离: {min_distance_this_episode:.1f} pixels")
            print(f"    步骤数: {step_count}")
            print(f"    成功: {'✅ 是' if episode_success else '❌ 否'}")
        
        # 测试总结
        success_rate = success_count / num_episodes
        avg_reward = np.mean(total_rewards)
        avg_min_distance = np.mean(min_distances)
        avg_episode_length = np.mean(episode_lengths)
        
        print(f"\n{'='*60}")
        print(f"🏆 测试结果总结:")
        print(f"  测试Episodes: {num_episodes}")
        print(f"  成功次数: {success_count}")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  平均奖励: {avg_reward:.2f}")
        print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
        print(f"  平均Episode长度: {avg_episode_length:.1f} steps")
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
        
        env.close()
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward, 
            'avg_min_distance': avg_min_distance,
            'avg_episode_length': avg_episode_length,
            'success_count': success_count,
            'total_episodes': num_episodes
        }
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    # 获取参数
    model_path = sys.argv[1] if len(sys.argv) > 1 else "../../trained_models/reacher2d/enhanced_test/08-28-2025-20-07-09/best_models/final_model_step_119999.pth"
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    render = True if len(sys.argv) <= 3 or sys.argv[3].lower() != 'false' else False
    
    print(f"🚀 开始测试...")
    print(f"   模型路径: {model_path}")
    print(f"   Episodes: {num_episodes}")
    print(f"   渲染: {render}")
    
    result = test_model_performance(model_path, num_episodes, render)
    
    if result:
        print(f"\n🎯 快速结论:")
        if result['success_rate'] >= 0.8:
            print(f"  ✅ 模型表现优秀! 继续当前训练策略")
        elif result['success_rate'] >= 0.3:
            print(f"  ⚠️  模型表现一般，建议继续训练或调整参数")
        else:
            print(f"  ❌ 模型表现较差，需要重新审视奖励函数或网络结构")
    else:
        print(f"❌ 测试失败")
EOF

# 6. 运行Python测试脚本
echo "🎮 开始测试模型性能..."
python temp_model_test.py "$MODEL_PATH" 5 true

# 7. 清理临时文件
rm -f temp_model_test.py

echo "🏁 测试完成!"

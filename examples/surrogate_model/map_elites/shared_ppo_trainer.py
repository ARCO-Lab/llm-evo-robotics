"""
多进程共享PPO训练器
支持多个进程协同训练一个PPO模型
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from queue import Queue, Empty
from threading import Thread
import time
from collections import deque
import pickle
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SharedExperienceBuffer:
    """共享经验缓冲区"""
    def __init__(self, buffer_size=50000, min_batch_size=1000):
        self.buffer_size = buffer_size
        self.min_batch_size = min_batch_size
        
        # 使用多进程安全的队列
        self.experience_queue = mp.Queue(maxsize=buffer_size)
        self.batch_ready_event = mp.Event()
        self.shutdown_event = mp.Event()
        
        # 统计信息
        self.total_experiences = mp.Value('i', 0)
        self.total_batches = mp.Value('i', 0)
        
    def add_experience(self, experience):
        """添加经验到缓冲区"""
        try:
            # 序列化经验数据
            serialized_exp = pickle.dumps(experience)
            self.experience_queue.put(serialized_exp, timeout=1.0)
            
            with self.total_experiences.get_lock():
                self.total_experiences.value += 1
                
            # 检查是否有足够的经验进行训练
            if self.total_experiences.value >= self.min_batch_size:
                self.batch_ready_event.set()
                
        except Exception as e:
            print(f"⚠️ 添加经验失败: {e}")
    
    def get_batch(self, batch_size=None):
        """获取一批经验用于训练"""
        if batch_size is None:
            batch_size = min(self.min_batch_size, self.total_experiences.value)
            
        experiences = []
        for _ in range(min(batch_size, self.experience_queue.qsize())):
            try:
                serialized_exp = self.experience_queue.get_nowait()
                experience = pickle.loads(serialized_exp)
                experiences.append(experience)
            except:
                break
                
        if experiences:
            with self.total_experiences.get_lock():
                self.total_experiences.value -= len(experiences)
            
            with self.total_batches.get_lock():
                self.total_batches.value += 1
                
        return experiences
    
    def is_ready(self):
        """检查是否有足够的经验进行训练"""
        return self.batch_ready_event.is_set()
    
    def reset_ready(self):
        """重置准备状态"""
        self.batch_ready_event.clear()
    
    def shutdown(self):
        """关闭缓冲区"""
        self.shutdown_event.set()


class SharedPPOTrainer:
    """共享PPO训练器"""
    def __init__(self, model_config, training_config):
        self.model_config = model_config
        self.training_config = training_config
        
        # 创建共享经验缓冲区
        self.experience_buffer = SharedExperienceBuffer(
            buffer_size=training_config.get('buffer_size', 50000),
            min_batch_size=training_config.get('min_batch_size', 1000)
        )
        
        # 模型参数共享（使用文件或共享内存）
        self.model_path = training_config.get('model_path', './shared_ppo_model.pth')
        self.update_interval = training_config.get('update_interval', 100)
        
        # 训练进程
        self.trainer_process = None
        self.is_training = mp.Value('b', False)
        
    def start_training(self):
        """启动训练进程"""
        print("🚀 启动共享PPO训练进程...")
        
        self.is_training.value = True
        self.trainer_process = mp.Process(
            target=self._training_loop,
            args=(self.experience_buffer, self.model_path, self.training_config)
        )
        self.trainer_process.start()
        print(f"✅ 训练进程已启动 (PID: {self.trainer_process.pid})")
    
    def stop_training(self):
        """停止训练进程"""
        print("🛑 停止共享PPO训练...")
        
        self.is_training.value = False
        self.experience_buffer.shutdown()
        
        if self.trainer_process and self.trainer_process.is_alive():
            self.trainer_process.join(timeout=5.0)
            if self.trainer_process.is_alive():
                self.trainer_process.terminate()
                self.trainer_process.join()
        
        print("✅ 训练进程已停止")
    
    @staticmethod
    def _training_loop(experience_buffer, model_path, config):
        """训练循环（在独立进程中运行）"""
        print("🎯 进入PPO训练循环...")
        
        # 在训练进程中初始化模型
        try:
            # 🔧 导入必要的模块
            import os
            import torch.nn as nn
            import torch.optim as optim
            import time
            from datetime import datetime
            
            # 🔧 简化的PPO模型初始化
            print("🤖 初始化简化PPO模型...")
            
            # 检查是否有已保存的模型
            load_existing_model = os.path.exists(model_path)
            if load_existing_model:
                print(f"🔍 发现已保存的模型: {model_path}")
            else:
                print("🆕 创建新的PPO模型")
            
            # 创建简单的Actor-Critic网络
            
            class SimpleActor(nn.Module):
                def __init__(self, obs_dim, action_dim, hidden_dim=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim),
                        nn.Tanh()
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            class SimpleCritic(nn.Module):
                def __init__(self, obs_dim, hidden_dim=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            # 🔧 动态获取观察维度（从第一个经验中推断）
            print("⏳ 等待第一个经验以确定观察维度...")
            first_batch = None
            while not experience_buffer.shutdown_event.is_set():
                experiences = experience_buffer.get_batch(batch_size=1)
                if experiences:
                    first_batch = experiences
                    break
                time.sleep(0.1)
            
            if first_batch:
                actual_obs_dim = len(first_batch[0]['observation'])
                actual_action_dim = len(first_batch[0]['action'])
                print(f"📊 检测到观察维度: {actual_obs_dim}, 动作维度: {actual_action_dim}")
            else:
                actual_obs_dim = config.get('observation_dim', 14)
                actual_action_dim = config.get('action_dim', 3)
                print(f"⚠️ 使用默认维度: obs={actual_obs_dim}, action={actual_action_dim}")
            
            # 初始化网络
            obs_dim = actual_obs_dim
            action_dim = actual_action_dim
            hidden_dim = config.get('hidden_dim', 256)
            
            actor = SimpleActor(obs_dim, action_dim, hidden_dim)
            critic = SimpleCritic(obs_dim, hidden_dim)
            
            actor_optimizer = optim.Adam(actor.parameters(), lr=config.get('lr', 2e-4))
            critic_optimizer = optim.Adam(critic.parameters(), lr=config.get('lr', 2e-4))
            
            update_count = 0
            
            # 🔧 加载已保存的模型（如果存在）
            if load_existing_model:
                try:
                    print(f"🔄 正在加载已保存的模型...")
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    actor.load_state_dict(checkpoint['actor'])
                    critic.load_state_dict(checkpoint['critic'])
                    actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                    update_count = checkpoint.get('update_count', 0)
                    
                    print(f"✅ 成功加载模型 - 已完成 {update_count} 次更新")
                    print(f"📊 模型参数:")
                    print(f"   观察维度: {obs_dim}")
                    print(f"   动作维度: {action_dim}")
                    print(f"   隐藏层维度: {hidden_dim}")
                    
                except Exception as e:
                    print(f"⚠️ 加载模型失败，使用随机初始化: {e}")
                    update_count = 0
            
            print("✅ 简化PPO模型初始化完成")
            
            # 🔧 如果有第一批经验，先处理它
            if first_batch:
                print(f"🔄 处理初始经验批次 ({len(first_batch)} 个)...")
                try:
                    # 处理第一批经验
                    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
                    
                    for exp in first_batch:
                        obs_batch.append(exp['observation'])
                        action_batch.append(exp['action'])
                        reward_batch.append(exp['reward'])
                        next_obs_batch.append(exp['next_observation'])
                        done_batch.append(exp['done'])
                    
                    # 转换为张量
                    obs_batch = torch.FloatTensor(np.array(obs_batch))
                    action_batch = torch.FloatTensor(np.array(action_batch))
                    reward_batch = torch.FloatTensor(reward_batch)
                    next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
                    done_batch = torch.BoolTensor(done_batch)
                    
                    # 简化训练
                    actor.train()
                    critic.train()
                    values = critic(obs_batch).squeeze()
                    actions_pred = actor(obs_batch)
                    value_loss = torch.nn.functional.mse_loss(values, reward_batch)
                    action_loss = torch.nn.functional.mse_loss(actions_pred, action_batch)
                    total_loss = value_loss + action_loss
                    
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    total_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()
                    
                    update_count += 1
                    print(f"📊 初始批次更新完成 - 损失: {total_loss.item():.4f}")
                    
                except Exception as e:
                    print(f"⚠️ 初始批次处理失败: {e}")
            
            while not experience_buffer.shutdown_event.is_set():
                # 等待足够的经验
                if not experience_buffer.is_ready():
                    time.sleep(0.1)
                    continue
                
                # 获取经验批次
                experiences = experience_buffer.get_batch()
                if not experiences:
                    continue
                
                print(f"🔄 处理 {len(experiences)} 个经验...")
                
                # 将经验转换为PPO训练格式
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
                
                for exp in experiences:
                    obs_batch.append(exp['observation'])
                    action_batch.append(exp['action'])
                    reward_batch.append(exp['reward'])
                    next_obs_batch.append(exp['next_observation'])
                    done_batch.append(exp['done'])
                
                # 转换为张量（改进版本）
                obs_batch = torch.FloatTensor(np.array(obs_batch))
                action_batch = torch.FloatTensor(np.array(action_batch))
                reward_batch = torch.FloatTensor(reward_batch)
                next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
                done_batch = torch.BoolTensor(done_batch)
                
                # 执行简化的PPO更新
                try:
                    # 🔧 简化的PPO训练逻辑
                    actor.train()
                    critic.train()
                    
                    # 计算价值和动作
                    values = critic(obs_batch).squeeze()
                    actions_pred = actor(obs_batch)
                    
                    # 简化的损失计算
                    value_loss = torch.nn.functional.mse_loss(values, reward_batch)
                    action_loss = torch.nn.functional.mse_loss(actions_pred, action_batch)
                    
                    total_loss = value_loss + action_loss
                    
                    # 反向传播
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    total_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()
                    
                    print(f"📊 PPO更新完成 - 批次: {len(experiences)}, 损失: {total_loss.item():.4f}")
                    
                    update_count += 1
                    
                    # 🔧 更频繁地保存模型 - 每次更新都保存
                    model_state = {
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                        'actor_optimizer': actor_optimizer.state_dict(),
                        'critic_optimizer': critic_optimizer.state_dict(),
                        'update_count': update_count
                    }
                    
                    # 确保保存目录存在
                    import os
                    model_dir = os.path.dirname(model_path)
                    if model_dir and not os.path.exists(model_dir):
                        os.makedirs(model_dir, exist_ok=True)
                        print(f"📁 创建模型保存目录: {model_dir}")
                    
                    torch.save(model_state, model_path)
                    print(f"💾 模型已保存 (更新次数: {update_count}) -> {model_path}")
                    
                    # 额外保存一个带时间戳的备份
                    if update_count % 5 == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = model_path.replace('.pth', f'_backup_{timestamp}.pth')
                        torch.save(model_state, backup_path)
                        print(f"💾 备份模型已保存: {backup_path}")
                        
                except Exception as e:
                    print(f"❌ PPO更新失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                experience_buffer.reset_ready()
                
        except Exception as e:
            print(f"❌ 训练循环错误: {e}")
            import traceback
            traceback.print_exc()
    
    def add_experience(self, experience):
        """添加经验到共享缓冲区"""
        self.experience_buffer.add_experience(experience)
    
    def load_latest_model(self):
        """加载最新的模型参数"""
        try:
            import os
            if os.path.exists(self.model_path):
                return torch.load(self.model_path)
            else:
                print("⚠️ 模型文件不存在，使用随机初始化")
                return None
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
            return None


class MultiProcessPPOWorker:
    """多进程PPO工作器"""
    def __init__(self, worker_id, shared_trainer, robot_config):
        self.worker_id = worker_id
        self.shared_trainer = shared_trainer
        self.robot_config = robot_config
        
        # 本地PPO模型（用于推理）
        self.local_ppo = None
        self.last_model_update = 0
        self.model_update_interval = 100  # 每100步更新一次模型
        
    def initialize_local_model(self):
        """初始化本地模型"""
        print(f"🤖 工作器 {self.worker_id} 初始化本地PPO模型...")
        
        # 加载最新的共享模型参数
        shared_params = self.shared_trainer.load_latest_model()
        
        # 初始化本地模型
        # 这里需要根据你的PPO实现调整
        # self.local_ppo = AttentionPPOWithBuffer(...)
        
        if shared_params:
            # self.local_ppo.load_state_dict(shared_params)
            print(f"✅ 工作器 {self.worker_id} 加载了共享模型参数")
        else:
            print(f"✅ 工作器 {self.worker_id} 使用随机初始化")
    
    def collect_experience(self, num_steps=1000):
        """收集经验并发送到共享缓冲区"""
        print(f"🎯 工作器 {self.worker_id} 开始收集经验...")
        
        # 这里需要根据你的环境实现
        for step in range(num_steps):
            # 模拟经验收集
            experience = {
                'observation': np.random.randn(10),  # 示例观察
                'action': np.random.randn(3),        # 示例动作
                'reward': np.random.randn(),         # 示例奖励
                'next_observation': np.random.randn(10),  # 示例下一状态
                'done': step % 100 == 99,           # 示例完成标志
                'worker_id': self.worker_id,
                'robot_config': self.robot_config
            }
            
            # 发送经验到共享缓冲区
            self.shared_trainer.add_experience(experience)
            
            # 定期更新本地模型
            if step % self.model_update_interval == 0:
                self.update_local_model()
        
        print(f"✅ 工作器 {self.worker_id} 完成经验收集")
    
    def update_local_model(self):
        """更新本地模型参数"""
        shared_params = self.shared_trainer.load_latest_model()
        if shared_params and self.local_ppo:
            # self.local_ppo.load_state_dict(shared_params)
            self.last_model_update += 1
            if self.last_model_update % 10 == 0:
                print(f"🔄 工作器 {self.worker_id} 更新了模型参数")


# 使用示例
def demo_shared_ppo_training():
    """演示共享PPO训练"""
    print("🚀 演示多进程共享PPO训练")
    
    # 配置
    model_config = {
        'observation_dim': 10,
        'action_dim': 3,
        'hidden_dim': 256
    }
    
    training_config = {
        'lr': 2e-4,
        'buffer_size': 10000,
        'min_batch_size': 500,
        'model_path': './shared_ppo_demo.pth',
        'update_interval': 100
    }
    
    # 创建共享训练器
    shared_trainer = SharedPPOTrainer(model_config, training_config)
    
    try:
        # 启动训练进程
        shared_trainer.start_training()
        
        # 创建多个工作器
        workers = []
        worker_processes = []
        
        for i in range(3):  # 3个工作器
            robot_config = {
                'num_joints': 3 + i,
                'link_lengths': [90.0] * (3 + i)
            }
            
            worker = MultiProcessPPOWorker(i, shared_trainer, robot_config)
            workers.append(worker)
            
            # 启动工作器进程
            process = mp.Process(
                target=worker.collect_experience,
                args=(500,)  # 每个工作器收集500步经验
            )
            worker_processes.append(process)
            process.start()
        
        # 等待所有工作器完成
        for process in worker_processes:
            process.join()
        
        print("🎉 所有工作器完成经验收集")
        time.sleep(2)  # 让训练进程处理剩余经验
        
    finally:
        # 停止训练
        shared_trainer.stop_training()
        print("✅ 共享PPO训练演示完成")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    demo_shared_ppo_training()

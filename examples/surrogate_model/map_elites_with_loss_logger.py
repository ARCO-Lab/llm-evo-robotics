#!/usr/bin/env python3
"""
MAP-Elites训练器 - 集成损失记录器版本
功能：
- 在运行MAP-Elites训练的同时启动独立的损失记录进程
- 记录attention、GNN、PPO网络的每步损失
- 实时生成损失曲线图表
- 支持多种训练模式
"""

import os
import sys
import time
import argparse
import atexit
import signal
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'map_elites'))

# 导入损失记录器
from loss_logger_interface import LossLoggerInterface, start_loss_logging, stop_loss_logging

# 导入MAP-Elites训练器
from map_elites.map_elites_trainer import (
    MAPElitesEvolutionTrainer, 
    start_real_training, 
    start_advanced_training,
    start_multiprocess_rendering_training,
    start_shared_ppo_training
)

class MAPElitesWithLossLogger:
    """MAP-Elites训练器 - 集成损失记录器"""
    
    def __init__(self, experiment_name=None, enable_loss_logging=True,
                 loss_log_dir="network_loss_logs", loss_update_interval=15.0):
        """
        初始化MAP-Elites训练器 + 损失记录器
        
        Args:
            experiment_name: 实验名称
            enable_loss_logging: 是否启用损失记录
            loss_log_dir: 损失日志目录
            loss_update_interval: 损失图表更新间隔（秒）
        """
        self.experiment_name = experiment_name or f"map_elites_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_loss_logging = enable_loss_logging
        self.loss_log_dir = loss_log_dir
        self.loss_update_interval = loss_update_interval
        
        # 损失记录器接口
        self.loss_logger_interface = None
        
        # 注册清理函数
        atexit.register(self.cleanup)
        
        print(f"🚀 MAP-Elites + 损失记录器初始化完成")
        print(f"   实验名称: {self.experiment_name}")
        print(f"   损失记录: {'启用' if enable_loss_logging else '禁用'}")
        
    def start_loss_logging(self):
        """启动损失记录器"""
        if not self.enable_loss_logging:
            print("⚠️  损失记录已禁用")
            return
            
        if self.loss_logger_interface is not None:
            print("⚠️  损失记录器已启动")
            return
            
        print("🎯 启动损失记录器...")
        
        # 🆕 设置环境变量，让训练进程知道实验名称
        os.environ['LOSS_EXPERIMENT_NAME'] = self.experiment_name
        
        # 🔧 使用简化的损失监控器（更可靠）
        try:
            from simple_loss_monitor import start_simple_loss_monitor
            self.simple_monitor = start_simple_loss_monitor(self.experiment_name)
            print(f"✅ 简化损失监控器已启动")
            print(f"   实验名称: {self.experiment_name}")
            print(f"   日志目录: simple_loss_logs/{self.experiment_name}_loss_log")
        except Exception as e:
            print(f"❌ 简化损失监控器启动失败: {e}")
            self.simple_monitor = None
        
        # 备用：尝试启动原始损失记录器
        try:
            self.loss_logger_interface = start_loss_logging(
                experiment_name=f"{self.experiment_name}_loss_log",
                log_dir=self.loss_log_dir,
                networks=['attention', 'ppo', 'gnn', 'sac', 'total'],
                update_interval=self.loss_update_interval
            )
            
            if self.loss_logger_interface:
                print(f"✅ 高级损失记录器也已启动")
                print(f"   日志目录: {self.loss_logger_interface.get_log_dir()}")
        except Exception as e:
            print(f"⚠️ 高级损失记录器启动失败: {e}")
            self.loss_logger_interface = None
            
    def stop_loss_logging(self):
        """停止损失记录器"""
        # 停止简化监控器
        if hasattr(self, 'simple_monitor') and self.simple_monitor:
            print("🛑 停止简化损失监控器...")
            from simple_loss_monitor import stop_simple_loss_monitor
            stop_simple_loss_monitor()
            self.simple_monitor = None
            print("✅ 简化损失监控器已停止")
            
        # 停止高级损失记录器
        if self.loss_logger_interface:
            print("🛑 停止高级损失记录器...")
            stop_loss_logging()
            self.loss_logger_interface = None
            print("✅ 高级损失记录器已停止")
            
    def cleanup(self):
        """清理资源"""
        self.stop_loss_logging()
        
    def run_basic_training(self):
        """运行基础MAP-Elites训练"""
        print("\n🚀 启动基础MAP-Elites训练")
        print("=" * 60)
        
        # 启动损失记录器
        self.start_loss_logging()
        
        try:
            # 运行MAP-Elites训练
            start_real_training()
            
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_loss_logging()
            
    def run_advanced_training(self):
        """运行高级MAP-Elites训练"""
        print("\n🚀 启动高级MAP-Elites训练")
        print("=" * 60)
        
        # 启动损失记录器
        self.start_loss_logging()
        
        try:
            # 运行MAP-Elites训练
            start_advanced_training()
            
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_loss_logging()
            
    def run_multiprocess_training(self):
        """运行多进程MAP-Elites训练"""
        print("\n🚀 启动多进程MAP-Elites训练")
        print("=" * 60)
        
        # 启动损失记录器
        self.start_loss_logging()
        
        try:
            # 运行MAP-Elites训练
            start_multiprocess_rendering_training()
            
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_loss_logging()
            
    def run_shared_ppo_training(self):
        """运行共享PPO MAP-Elites训练"""
        print("\n🚀 启动共享PPO MAP-Elites训练")
        print("=" * 60)
        
        # 启动损失记录器
        self.start_loss_logging()
        
        try:
            # 运行MAP-Elites训练
            start_shared_ppo_training()
            
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_loss_logging()
            
    def run_custom_training(self, base_args, **trainer_kwargs):
        """运行自定义MAP-Elites训练"""
        print("\n🚀 启动自定义MAP-Elites训练")
        print("=" * 60)
        
        # 启动损失记录器
        self.start_loss_logging()
        
        try:
            # 创建训练器
            trainer = MAPElitesEvolutionTrainer(
                base_args=base_args,
                **trainer_kwargs
            )
            
            # 开始进化
            trainer.run_evolution()
            
            print("\n🎉 训练完成!")
            
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_loss_logging()


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='MAP-Elites训练器 + 损失记录器')
    
    # 训练模式选择
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'advanced', 'multiprocess', 'shared-ppo', 'custom'],
                       help='训练模式')
    
    # 损失记录器参数
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--disable-loss-logging', action='store_true',
                       help='禁用损失记录')
    parser.add_argument('--loss-log-dir', type=str, default='network_loss_logs',
                       help='损失日志目录')
    parser.add_argument('--loss-update-interval', type=float, default=15.0,
                       help='损失图表更新间隔（秒）')
    
    # 自定义训练参数（当mode='custom'时使用）
    parser.add_argument('--env-type', type=str, default='reacher2d',
                       help='环境类型')
    parser.add_argument('--save-dir', type=str, default='./map_elites_results',
                       help='保存目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--num-initial-random', type=int, default=10,
                       help='初始随机个体数')
    parser.add_argument('--training-steps-per-individual', type=int, default=2000,
                       help='每个个体的训练步数')
    parser.add_argument('--num-generations', type=int, default=20,
                       help='进化代数')
    parser.add_argument('--individuals-per-generation', type=int, default=5,
                       help='每代新个体数')
    parser.add_argument('--enable-rendering', action='store_true',
                       help='启用环境渲染')
    parser.add_argument('--silent-mode', action='store_true',
                       help='静默模式')
    parser.add_argument('--use-genetic-fitness', action='store_true',
                       help='使用遗传算法fitness')
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("🎯 MAP-Elites训练器 + 损失记录器")
    print("=" * 60)
    print(f"训练模式: {args.mode}")
    print(f"实验名称: {args.experiment_name}")
    print(f"损失记录: {'禁用' if args.disable_loss_logging else '启用'}")
    
    # 创建训练器
    trainer = MAPElitesWithLossLogger(
        experiment_name=args.experiment_name,
        enable_loss_logging=not args.disable_loss_logging,
        loss_log_dir=args.loss_log_dir,
        loss_update_interval=args.loss_update_interval
    )
    
    # 设置信号处理
    def signal_handler(signum, frame):
        print(f"\n🛑 接收到信号 {signum}，正在清理...")
        trainer.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 根据模式运行训练
    try:
        if args.mode == 'basic':
            trainer.run_basic_training()
        elif args.mode == 'advanced':
            trainer.run_advanced_training()
        elif args.mode == 'multiprocess':
            trainer.run_multiprocess_training()
        elif args.mode == 'shared-ppo':
            trainer.run_shared_ppo_training()
        elif args.mode == 'custom':
            # 创建自定义训练参数
            base_args = argparse.Namespace()
            base_args.env_type = args.env_type
            base_args.num_processes = 1
            base_args.seed = args.seed
            base_args.save_dir = args.save_dir
            base_args.use_real_training = True
            base_args.lr = args.lr
            base_args.alpha = 0.2
            base_args.tau = 0.005
            base_args.gamma = 0.99
            base_args.update_frequency = 1
            
            # 自定义训练器参数
            trainer_kwargs = {
                'num_initial_random': args.num_initial_random,
                'training_steps_per_individual': args.training_steps_per_individual,
                'enable_rendering': args.enable_rendering,
                'silent_mode': args.silent_mode,
                'use_genetic_fitness': args.use_genetic_fitness,
                'enable_multiprocess': False,
                'max_workers': 1,
                'enable_visualization': True,
                'visualization_interval': 5
            }
            
            trainer.run_custom_training(base_args, **trainer_kwargs)
        else:
            print(f"❌ 未知训练模式: {args.mode}")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

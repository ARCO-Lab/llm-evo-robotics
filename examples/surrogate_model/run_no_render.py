#!/usr/bin/env python3
"""
无渲染模式运行脚本
强制禁用所有渲染功能
"""

import os
import sys

def run_no_render_training(experiment_name, training_steps=500, num_generations=1, individuals_per_generation=2):
    """运行无渲染训练"""
    
    # 设置环境变量强制禁用渲染
    os.environ['DISABLE_RENDER'] = '1'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # 强制使用虚拟显示驱动
    
    print(f"🚫 强制禁用渲染模式")
    print(f"   实验名称: {experiment_name}")
    print(f"   训练步数: {training_steps}")
    print(f"   代数: {num_generations}")
    print(f"   每代个体数: {individuals_per_generation}")
    
    # 直接调用MAP-Elites训练器
    sys.path.append('map_elites')
    from map_elites_trainer import MAPElitesEvolutionTrainer
    import argparse
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.env_name = 'reacher2d'
    base_args.seed = 42
    base_args.lr = 3e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    base_args.batch_size = 64
    base_args.buffer_capacity = 100000
    base_args.warmup_steps = 1000
    base_args.target_entropy_factor = 1.0
    base_args.update_frequency = 1
    base_args.save_dir = f'./no_render_results/{experiment_name}'
    
    # 创建训练器 - 强制禁用渲染
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=individuals_per_generation,
        training_steps_per_individual=training_steps,
        enable_rendering=False,        # 强制禁用渲染
        silent_mode=False,             # 保持输出以便调试
        use_genetic_fitness=True,
        enable_multiprocess=False,     # 禁用多进程避免复杂性
        max_workers=1,
        enable_visualization=False,    # 禁用可视化
        visualization_interval=999999  # 设置极大值避免可视化
    )
    
    print(f"✅ 训练器已创建 (强制无渲染模式)")
    print(f"   adapter.enable_rendering = {trainer.adapter.enable_rendering}")
    
    # 运行训练
    trainer.run_evolution(num_generations, individuals_per_generation)
    
    print(f"🎉 无渲染训练完成！")
    return trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='无渲染模式训练')
    parser.add_argument('--experiment-name', type=str, required=True, help='实验名称')
    parser.add_argument('--training-steps', type=int, default=500, help='训练步数')
    parser.add_argument('--num-generations', type=int, default=1, help='代数')
    parser.add_argument('--individuals-per-generation', type=int, default=2, help='每代个体数')
    
    args = parser.parse_args()
    
    run_no_render_training(
        args.experiment_name,
        args.training_steps,
        args.num_generations,
        args.individuals_per_generation
    )

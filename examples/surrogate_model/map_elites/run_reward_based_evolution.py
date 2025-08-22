#!/usr/bin/env python3
"""
运行基于reward比例选择的MAP-Elites进化
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from map_elites_trainer import MAPElitesEvolutionTrainer
import argparse

def main():
    """运行MAP-Elites实验"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='基于reward比例选择的MAP-Elites进化')
    parser.add_argument('--generations', type=int, default=10, help='进化代数')
    parser.add_argument('--individuals-per-gen', type=int, default=5, help='每代个体数')
    parser.add_argument('--initial-random', type=int, default=8, help='初始随机个体数')
    parser.add_argument('--training-steps', type=int, default=2000, help='每个个体的训练步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--enable-rendering', action='store_true', help='启用可视化渲染')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    print(f"🧬 启动基于reward比例选择的MAP-Elites实验")
    print(f"📊 参数: {args.generations}代, 每代{args.individuals_per_gen}个体")
    print(f"🎯 选择策略: 基于reward比例")
    print(f"🎨 可视化: {'启用' if args.enable_rendering else '禁用'}")
    print(f"🔇 静默模式: {'禁用' if args.verbose else '启用'}")
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.seed = args.seed
    base_args.update_frequency = 1
    
    # 创建训练器
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=args.initial_random,
        training_steps_per_individual=args.training_steps,
        enable_rendering=args.enable_rendering,
        silent_mode=not args.verbose  # verbose=True时，silent_mode=False
    )
    
    # 运行进化
    trainer.run_evolution(
        num_generations=args.generations,
        individuals_per_generation=args.individuals_per_gen
    )
    
    print("🎉 MAP-Elites实验完成！")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
English MAP-Elites Visualization Generator (No Chinese Characters)
Generates comprehensive visualizations with English labels only
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from collections import defaultdict

def read_csv_data(csv_file: str) -> dict:
    """
    Read CSV file and return data dictionary
    """
    data = defaultdict(list)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    # Try to convert to numbers
                    try:
                        if '.' in value:
                            data[key].append(float(value))
                        else:
                            data[key].append(int(value))
                    except (ValueError, TypeError):
                        data[key].append(value)
        
        print(f"Successfully read CSV file: {len(data[list(data.keys())[0]])} records")
        print(f"Columns: {list(data.keys())}")
        return data
        
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return {}

def create_map_elites_heatmap(data: dict, output_path: str) -> bool:
    """
    Create MAP-Elites heatmap visualization
    """
    try:
        # Check required columns
        required_cols = ['num_links', 'total_length', 'fitness']
        if not all(col in data for col in required_cols):
            print(f"Missing required columns: {required_cols}")
            return False
        
        num_links = np.array(data['num_links'])
        total_length = np.array(data['total_length'])
        fitness = np.array(data['fitness'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'MAP-Elites Training Results Heatmap Analysis\\nIndividuals: {len(num_links)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Number of Links vs Total Length scatter plot
        scatter1 = ax1.scatter(num_links, total_length, c=fitness, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='white')
        ax1.set_xlabel('Number of Links')
        ax1.set_ylabel('Total Length (px)')
        ax1.set_title('Robot Structure vs Performance')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Fitness Score')
        
        # 2. Learning rate analysis (if available)
        if 'lr' in data and 'alpha' in data:
            lr = np.array(data['lr'])
            alpha = np.array(data['alpha'])
            scatter2 = ax2.scatter(lr, alpha, c=fitness, 
                                 cmap='plasma', s=100, alpha=0.7, edgecolors='white')
            ax2.set_xlabel('Learning Rate')
            ax2.set_ylabel('SAC Alpha Parameter')
            ax2.set_title('Hyperparameter Space')
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Fitness Score')
        else:
            ax2.text(0.5, 0.5, 'Learning Rate Data\\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        
        # 3. Fitness distribution
        ax3.hist(fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Fitness Score')
        ax3.set_ylabel('Number of Individuals')
        ax3.set_title('Fitness Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_fitness = np.mean(fitness)
        max_fitness = np.max(fitness)
        ax3.axvline(mean_fitness, color='red', linestyle='--', 
                   label=f'Mean: {mean_fitness:.3f}')
        ax3.axvline(max_fitness, color='green', linestyle='--', 
                   label=f'Best: {max_fitness:.3f}')
        ax3.legend()
        
        # 4. Robot structure diversity
        unique_links, counts = np.unique(num_links, return_counts=True)
        ax4.bar(unique_links, counts, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Number of Links')
        ax4.set_ylabel('Number of Individuals')
        ax4.set_title('Robot Structure Diversity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MAP-Elites heatmap saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to generate MAP-Elites heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_metrics_plot(data: dict, output_path: str) -> bool:
    """
    Create training metrics visualization
    """
    try:
        if 'fitness' not in data:
            return False
        
        fitness = np.array(data['fitness'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Performance Metrics Analysis\\nTotal Individuals: {len(fitness)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Fitness vs Success Rate
        if 'success_rate' in data:
            success_rate = np.array(data['success_rate'])
            ax1.scatter(fitness, success_rate, s=100, alpha=0.7, color='blue', edgecolors='white')
            ax1.set_xlabel('Fitness Score')
            ax1.set_ylabel('Success Rate')
            ax1.set_title('Performance Correlation')
            ax1.grid(True, alpha=0.3)
            
            # Success rate distribution
            ax2.hist(success_rate, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Success Rate')
            ax2.set_ylabel('Number of Individuals')
            ax2.set_title('Success Rate Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Success Rate Data\\nNot Available', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'Success Rate Data\\nNot Available', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Reward analysis
        if 'avg_reward' in data:
            avg_reward = np.array(data['avg_reward'])
            ax3.scatter(fitness, avg_reward, s=100, alpha=0.7, color='red', edgecolors='white')
            ax3.set_xlabel('Fitness Score')
            ax3.set_ylabel('Average Reward')
            ax3.set_title('Fitness vs Reward Correlation')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Reward Data\\nNot Available', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Generational evolution (if available)
        if 'generation' in data:
            generation = np.array(data['generation'])
            
            # Group by generation and calculate statistics
            generations = {}
            for i, gen in enumerate(generation):
                if gen not in generations:
                    generations[gen] = []
                generations[gen].append(fitness[i])
            
            gen_nums = sorted(generations.keys())
            gen_max = [max(generations[gen]) for gen in gen_nums]
            gen_mean = [np.mean(generations[gen]) for gen in gen_nums]
            
            ax4.plot(gen_nums, gen_max, 'b-o', label='Best Fitness', linewidth=2, markersize=6)
            ax4.plot(gen_nums, gen_mean, 'r--s', label='Average Fitness', linewidth=2, markersize=6)
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Fitness Score')
            ax4.set_title('Evolutionary Progress')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Generation Data\\nNot Available', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics visualization saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to generate training metrics visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_neural_network_loss_simulation(data: dict, output_path: str) -> bool:
    """
    Create simulated neural network loss visualization
    """
    try:
        if 'fitness' not in data:
            return False
        
        fitness = np.array(data['fitness'])
        n_individuals = len(fitness)
        
        # Simulate different network component loss changes
        np.random.seed(42)  # Ensure reproducible results
        
        # Simulate training process for each individual
        training_steps = 100
        steps = np.arange(training_steps)
        
        # Simulate different network losses
        actor_losses = []
        critic_losses = []
        attention_losses = []
        gnn_losses = []
        
        for i in range(n_individuals):
            # Generate different loss curves based on fitness
            base_fitness = fitness[i]
            
            # Actor loss: starts high, gradually decreases
            actor_loss = 10.0 * (1 - base_fitness) * np.exp(-steps / 50) + np.random.normal(0, 0.1, training_steps)
            actor_losses.append(actor_loss)
            
            # Critic loss: similar but slightly different pattern
            critic_loss = 8.0 * (1 - base_fitness) * np.exp(-steps / 40) + np.random.normal(0, 0.15, training_steps)
            critic_losses.append(critic_loss)
            
            # Attention loss: faster convergence
            attention_loss = 5.0 * (1 - base_fitness) * np.exp(-steps / 30) + np.random.normal(0, 0.08, training_steps)
            attention_losses.append(attention_loss)
            
            # GNN loss: medium convergence speed
            gnn_loss = 6.0 * (1 - base_fitness) * np.exp(-steps / 45) + np.random.normal(0, 0.12, training_steps)
            gnn_losses.append(gnn_loss)
        
        # Calculate average loss curves
        avg_actor_loss = np.mean(actor_losses, axis=0)
        avg_critic_loss = np.mean(critic_losses, axis=0)
        avg_attention_loss = np.mean(attention_losses, axis=0)
        avg_gnn_loss = np.mean(gnn_losses, axis=0)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Neural Network Loss Analysis (Simulated from {n_individuals} individuals)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Main network loss curves
        ax1.plot(steps, avg_actor_loss, 'r-', linewidth=2, label='Actor Loss', alpha=0.8)
        ax1.plot(steps, avg_critic_loss, 'b-', linewidth=2, label='Critic Loss', alpha=0.8)
        ax1.plot(steps, avg_attention_loss, 'g-', linewidth=2, label='Attention Loss', alpha=0.8)
        ax1.plot(steps, avg_gnn_loss, 'm-', linewidth=2, label='GNN Loss', alpha=0.8)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Network Component Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Final loss comparison
        final_losses = [avg_actor_loss[-1], avg_critic_loss[-1], 
                       avg_attention_loss[-1], avg_gnn_loss[-1]]
        network_names = ['Actor', 'Critic', 'Attention', 'GNN']
        colors = ['red', 'blue', 'green', 'magenta']
        
        bars = ax2.bar(network_names, final_losses, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Final Loss Value')
        ax2.set_title('Network Convergence Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        # 3. Convergence speed analysis
        # Calculate steps to reach 50% of initial loss for each network
        convergence_steps = []
        initial_losses = [avg_actor_loss[0], avg_critic_loss[0], 
                         avg_attention_loss[0], avg_gnn_loss[0]]
        all_losses = [avg_actor_loss, avg_critic_loss, avg_attention_loss, avg_gnn_loss]
        
        for i, loss_curve in enumerate(all_losses):
            target = initial_losses[i] * 0.5
            convergence_step = np.argmax(loss_curve <= target)
            convergence_steps.append(convergence_step if convergence_step > 0 else training_steps)
        
        bars = ax3.bar(network_names, convergence_steps, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Steps to 50% Initial Loss')
        ax3.set_title('Convergence Speed Comparison')
        ax3.grid(True, alpha=0.3)
        
        for bar, steps in zip(bars, convergence_steps):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{steps}', ha='center', va='bottom')
        
        # 4. Training stability analysis
        # Calculate standard deviation of last 20 steps
        stability_scores = []
        for loss_curve in all_losses:
            last_20_steps = loss_curve[-20:]
            stability = np.std(last_20_steps)
            stability_scores.append(stability)
        
        bars = ax4.bar(network_names, stability_scores, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Loss Stability (Std Dev)')
        ax4.set_title('Training Stability Analysis')
        ax4.grid(True, alpha=0.3)
        
        for bar, stability in zip(bars, stability_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{stability:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Neural network loss visualization saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to generate neural network loss visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comprehensive_report(data: dict, output_path: str) -> bool:
    """
    Create comprehensive statistics report
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MAP-Elites Comprehensive Training Results Report\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total Individuals: {len(data[list(data.keys())[0]])}\\n\\n")
            
            # Basic statistics
            f.write("Basic Statistics:\\n")
            f.write("-" * 30 + "\\n")
            
            if 'fitness' in data:
                fitness = np.array(data['fitness'])
                f.write(f"Fitness Statistics:\\n")
                f.write(f"  Mean: {np.mean(fitness):.4f}\\n")
                f.write(f"  Best: {np.max(fitness):.4f}\\n")
                f.write(f"  Worst: {np.min(fitness):.4f}\\n")
                f.write(f"  Std Dev: {np.std(fitness):.4f}\\n\\n")
            
            if 'success_rate' in data:
                success_rate = np.array(data['success_rate'])
                f.write(f"Success Rate Statistics:\\n")
                f.write(f"  Average Success Rate: {np.mean(success_rate):.4f}\\n")
                f.write(f"  Best Success Rate: {np.max(success_rate):.4f}\\n")
                f.write(f"  Individuals with >50% Success: {np.sum(success_rate > 0.5)}\\n\\n")
            
            if 'num_links' in data:
                num_links = np.array(data['num_links'])
                f.write(f"Robot Structure Statistics:\\n")
                f.write(f"  Link Count Range: {np.min(num_links)}-{np.max(num_links)}\\n")
                unique_links, counts = np.unique(num_links, return_counts=True)
                for links, count in zip(unique_links, counts):
                    f.write(f"  {int(links)} Links: {count} individuals ({count/len(num_links)*100:.1f}%)\\n")
                f.write("\\n")
            
            if 'total_length' in data:
                total_length = np.array(data['total_length'])
                f.write(f"Robot Length Statistics:\\n")
                f.write(f"  Average Length: {np.mean(total_length):.1f}px\\n")
                f.write(f"  Length Range: {np.min(total_length):.1f}-{np.max(total_length):.1f}px\\n\\n")
            
            # Generational analysis
            if 'generation' in data:
                generation = np.array(data['generation'])
                fitness = np.array(data['fitness'])
                
                f.write("Generational Evolution Analysis:\\n")
                f.write("-" * 30 + "\\n")
                
                generations = {}
                for i, gen in enumerate(generation):
                    if gen not in generations:
                        generations[gen] = []
                    generations[gen].append(fitness[i])
                
                for gen in sorted(generations.keys()):
                    gen_fitness = generations[gen]
                    f.write(f"Generation {gen}: {len(gen_fitness)} individuals, ")
                    f.write(f"Best={max(gen_fitness):.4f}, ")
                    f.write(f"Average={np.mean(gen_fitness):.4f}\\n")
                f.write("\\n")
            
            # Best individual information
            if 'fitness' in data:
                fitness = np.array(data['fitness'])
                best_idx = np.argmax(fitness)
                f.write("Best Individual Details:\\n")
                f.write("-" * 30 + "\\n")
                f.write(f"Fitness: {fitness[best_idx]:.4f}\\n")
                
                if 'num_links' in data:
                    f.write(f"Number of Links: {data['num_links'][best_idx]}\\n")
                if 'total_length' in data:
                    f.write(f"Total Length: {data['total_length'][best_idx]:.1f}px\\n")
                if 'success_rate' in data:
                    f.write(f"Success Rate: {data['success_rate'][best_idx]:.4f}\\n")
                if 'lr' in data:
                    f.write(f"Learning Rate: {data['lr'][best_idx]:.2e}\\n")
                if 'alpha' in data:
                    f.write(f"SAC Alpha: {data['alpha'][best_idx]:.4f}\\n")
        
        print(f"Comprehensive report saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to generate comprehensive report: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate English MAP-Elites Visualizations")
    parser.add_argument("--csv-file", type=str, 
                       default="./experiment_results/session_20250917_160838/results.csv",
                       help="CSV results file")
    parser.add_argument("--output-dir", type=str, 
                       default="./map_elites_shared_ppo_results/visualizations_english",
                       help="Output directory")
    args = parser.parse_args()
    
    print("Generating English MAP-Elites Visualizations")
    print("=" * 60)
    
    if not os.path.exists(args.csv_file):
        print(f"CSV file not found: {args.csv_file}")
        return
    
    # Read data
    data = read_csv_data(args.csv_file)
    if not data:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    
    # 1. Generate MAP-Elites heatmap
    print("\\n1. Generating MAP-Elites Heatmap...")
    heatmap_path = os.path.join(args.output_dir, "map_elites_heatmap_english.png")
    if create_map_elites_heatmap(data, heatmap_path):
        success_count += 1
    
    # 2. Generate training metrics visualization
    print("\\n2. Generating Training Metrics Visualization...")
    metrics_path = os.path.join(args.output_dir, "training_metrics_english.png")
    if create_training_metrics_plot(data, metrics_path):
        success_count += 1
    
    # 3. Generate neural network loss visualization (simulated)
    print("\\n3. Generating Neural Network Loss Visualization...")
    loss_path = os.path.join(args.output_dir, "neural_network_losses_english.png")
    if create_neural_network_loss_simulation(data, loss_path):
        success_count += 1
    
    # 4. Generate comprehensive report
    print("\\n4. Generating Comprehensive Report...")
    report_path = os.path.join(args.output_dir, "comprehensive_report_english.txt")
    if create_comprehensive_report(data, report_path):
        success_count += 1
    
    print("\\n" + "=" * 60)
    print(f"Visualization generation complete! Success: {success_count}/4")
    print(f"All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()



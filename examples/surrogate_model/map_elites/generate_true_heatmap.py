#!/usr/bin/env python3
"""
True MAP-Elites Heatmap Generator
Creates actual grid-based heatmaps showing performance across behavioral dimensions
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from collections import defaultdict
from scipy.interpolate import griddata

def read_csv_data(csv_file: str) -> dict:
    """Read CSV file and return data dictionary"""
    data = defaultdict(list)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    try:
                        if '.' in value:
                            data[key].append(float(value))
                        else:
                            data[key].append(int(value))
                    except (ValueError, TypeError):
                        data[key].append(value)
        
        print(f"Successfully read CSV file: {len(data[list(data.keys())[0]])} records")
        return data
        
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return {}

def create_true_map_elites_heatmap(data: dict, output_path: str) -> bool:
    """
    Create true MAP-Elites heatmap with grid-based visualization
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
        fig.suptitle('MAP-Elites True Heatmap Analysis', fontsize=16, fontweight='bold')
        
        # 1. True heatmap: Links vs Length
        # Create bins for discretization
        link_bins = np.arange(num_links.min() - 0.5, num_links.max() + 1.5, 1)
        length_bins = np.linspace(total_length.min() * 0.95, total_length.max() * 1.05, 10)
        
        # Create 2D histogram for fitness values
        fitness_grid = np.full((len(link_bins)-1, len(length_bins)-1), np.nan)
        
        for i in range(len(num_links)):
            link_idx = np.digitize(num_links[i], link_bins) - 1
            length_idx = np.digitize(total_length[i], length_bins) - 1
            
            if 0 <= link_idx < len(link_bins)-1 and 0 <= length_idx < len(length_bins)-1:
                if np.isnan(fitness_grid[link_idx, length_idx]):
                    fitness_grid[link_idx, length_idx] = fitness[i]
                else:
                    # Take the maximum fitness if multiple individuals in same cell
                    fitness_grid[link_idx, length_idx] = max(fitness_grid[link_idx, length_idx], fitness[i])
        
        # Plot heatmap
        im1 = ax1.imshow(fitness_grid.T, cmap='viridis', origin='lower', 
                        extent=[link_bins[0], link_bins[-1], length_bins[0], length_bins[-1]],
                        aspect='auto', interpolation='nearest')
        ax1.set_xlabel('Number of Links')
        ax1.set_ylabel('Total Length (px)')
        ax1.set_title('MAP-Elites Archive Heatmap\\n(Structure vs Performance)')
        plt.colorbar(im1, ax=ax1, label='Best Fitness')
        
        # Add grid lines
        ax1.set_xticks(link_bins)
        ax1.set_yticks(length_bins[::2])  # Show every other tick for clarity
        ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # 2. Hyperparameter heatmap (if available)
        if 'lr' in data and 'alpha' in data:
            lr = np.array(data['lr'])
            alpha = np.array(data['alpha'])
            
            # Create bins for hyperparameters
            lr_bins = np.logspace(np.log10(lr.min()), np.log10(lr.max()), 8)
            alpha_bins = np.linspace(alpha.min(), alpha.max(), 8)
            
            # Create 2D histogram
            hyperparam_grid = np.full((len(lr_bins)-1, len(alpha_bins)-1), np.nan)
            
            for i in range(len(lr)):
                lr_idx = np.digitize(lr[i], lr_bins) - 1
                alpha_idx = np.digitize(alpha[i], alpha_bins) - 1
                
                if 0 <= lr_idx < len(lr_bins)-1 and 0 <= alpha_idx < len(alpha_bins)-1:
                    if np.isnan(hyperparam_grid[lr_idx, alpha_idx]):
                        hyperparam_grid[lr_idx, alpha_idx] = fitness[i]
                    else:
                        hyperparam_grid[lr_idx, alpha_idx] = max(hyperparam_grid[lr_idx, alpha_idx], fitness[i])
            
            im2 = ax2.imshow(hyperparam_grid.T, cmap='plasma', origin='lower',
                            extent=[np.log10(lr_bins[0]), np.log10(lr_bins[-1]), 
                                   alpha_bins[0], alpha_bins[-1]],
                            aspect='auto', interpolation='nearest')
            ax2.set_xlabel('Log10(Learning Rate)')
            ax2.set_ylabel('SAC Alpha Parameter')
            ax2.set_title('Hyperparameter Space Heatmap')
            plt.colorbar(im2, ax=ax2, label='Best Fitness')
            ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
        else:
            ax2.text(0.5, 0.5, 'Hyperparameter Data\\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        
        # 3. Generation-based heatmap (if available)
        if 'generation' in data:
            generation = np.array(data['generation'])
            
            # Create generation vs structure heatmap
            gen_bins = np.arange(generation.min() - 0.5, generation.max() + 1.5, 1)
            
            gen_structure_grid = np.full((len(gen_bins)-1, len(link_bins)-1), np.nan)
            
            for i in range(len(generation)):
                gen_idx = np.digitize(generation[i], gen_bins) - 1
                link_idx = np.digitize(num_links[i], link_bins) - 1
                
                if 0 <= gen_idx < len(gen_bins)-1 and 0 <= link_idx < len(link_bins)-1:
                    if np.isnan(gen_structure_grid[gen_idx, link_idx]):
                        gen_structure_grid[gen_idx, link_idx] = fitness[i]
                    else:
                        gen_structure_grid[gen_idx, link_idx] = max(gen_structure_grid[gen_idx, link_idx], fitness[i])
            
            im3 = ax3.imshow(gen_structure_grid.T, cmap='coolwarm', origin='lower',
                            extent=[gen_bins[0], gen_bins[-1], link_bins[0], link_bins[-1]],
                            aspect='auto', interpolation='nearest')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Number of Links')
            ax3.set_title('Evolution Progress Heatmap')
            plt.colorbar(im3, ax=ax3, label='Best Fitness')
            ax3.grid(True, alpha=0.3, color='white', linewidth=0.5)
        else:
            ax3.text(0.5, 0.5, 'Generation Data\\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        # 4. Performance vs Success Rate heatmap
        if 'success_rate' in data:
            success_rate = np.array(data['success_rate'])
            
            # Create bins
            fitness_bins = np.linspace(fitness.min(), fitness.max(), 8)
            success_bins = np.linspace(success_rate.min(), success_rate.max(), 8)
            
            # Create density heatmap (count of individuals)
            density_grid = np.zeros((len(fitness_bins)-1, len(success_bins)-1))
            
            for i in range(len(fitness)):
                fit_idx = np.digitize(fitness[i], fitness_bins) - 1
                suc_idx = np.digitize(success_rate[i], success_bins) - 1
                
                if 0 <= fit_idx < len(fitness_bins)-1 and 0 <= suc_idx < len(success_bins)-1:
                    density_grid[fit_idx, suc_idx] += 1
            
            im4 = ax4.imshow(density_grid.T, cmap='YlOrRd', origin='lower',
                            extent=[fitness_bins[0], fitness_bins[-1], 
                                   success_bins[0], success_bins[-1]],
                            aspect='auto', interpolation='nearest')
            ax4.set_xlabel('Fitness Score')
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Performance Distribution Heatmap')
            plt.colorbar(im4, ax=ax4, label='Number of Individuals')
            ax4.grid(True, alpha=0.3, color='white', linewidth=0.5)
        else:
            ax4.text(0.5, 0.5, 'Success Rate Data\\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"True MAP-Elites heatmap saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to generate true MAP-Elites heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_interpolated_heatmap(data: dict, output_path: str) -> bool:
    """
    Create interpolated heatmap for smoother visualization
    """
    try:
        if not all(col in data for col in ['num_links', 'total_length', 'fitness']):
            return False
        
        num_links = np.array(data['num_links'])
        total_length = np.array(data['total_length'])
        fitness = np.array(data['fitness'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MAP-Elites Interpolated Heatmaps', fontsize=16, fontweight='bold')
        
        # 1. Interpolated structure heatmap
        # Create regular grid
        xi = np.linspace(num_links.min(), num_links.max(), 20)
        yi = np.linspace(total_length.min(), total_length.max(), 20)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate fitness values
        Zi = griddata((num_links, total_length), fitness, (Xi, Yi), method='cubic', fill_value=0)
        
        im1 = ax1.contourf(Xi, Yi, Zi, levels=20, cmap='viridis', alpha=0.8)
        ax1.scatter(num_links, total_length, c=fitness, s=100, cmap='viridis', 
                   edgecolors='white', linewidth=1, zorder=5)
        ax1.set_xlabel('Number of Links')
        ax1.set_ylabel('Total Length (px)')
        ax1.set_title('Interpolated Structure Performance')
        plt.colorbar(im1, ax=ax1, label='Fitness Score')
        
        # 2. Generation progress
        if 'generation' in data:
            generation = np.array(data['generation'])
            
            # Calculate average fitness per generation
            gen_unique = np.unique(generation)
            gen_fitness = []
            gen_counts = []
            
            for gen in gen_unique:
                mask = generation == gen
                gen_fitness.append(np.mean(fitness[mask]))
                gen_counts.append(np.sum(mask))
            
            # Create heatmap showing generation progress
            gen_grid = np.zeros((len(gen_unique), 10))  # 10 arbitrary "quality" bins
            
            for i, (gen, fit, count) in enumerate(zip(gen_unique, gen_fitness, gen_counts)):
                quality_level = int(fit * 9)  # Map to 0-9 range
                gen_grid[i, :quality_level+1] = count
            
            im2 = ax2.imshow(gen_grid.T, cmap='Blues', origin='lower', aspect='auto',
                            extent=[gen_unique[0]-0.5, gen_unique[-1]+0.5, 0, 1])
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Normalized Quality Level')
            ax2.set_title('Generation Quality Distribution')
            plt.colorbar(im2, ax=ax2, label='Individual Count')
        else:
            ax2.text(0.5, 0.5, 'Generation Data\\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        
        # 3. Hyperparameter correlation heatmap
        if 'lr' in data and 'alpha' in data:
            lr = np.array(data['lr'])
            alpha = np.array(data['alpha'])
            
            # Create correlation matrix
            params = np.column_stack([lr, alpha, fitness])
            param_names = ['Learning Rate', 'Alpha', 'Fitness']
            
            correlation_matrix = np.corrcoef(params.T)
            
            im3 = ax3.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(param_names)))
            ax3.set_yticks(range(len(param_names)))
            ax3.set_xticklabels(param_names, rotation=45)
            ax3.set_yticklabels(param_names)
            ax3.set_title('Parameter Correlation Matrix')
            
            # Add correlation values
            for i in range(len(param_names)):
                for j in range(len(param_names)):
                    text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im3, ax=ax3, label='Correlation Coefficient')
        else:
            ax3.text(0.5, 0.5, 'Hyperparameter Data\\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        # 4. Archive coverage visualization
        # Show which cells in the MAP-Elites grid are filled
        link_range = np.arange(num_links.min(), num_links.max() + 1)
        length_bins = np.linspace(total_length.min(), total_length.max(), 10)
        
        coverage_grid = np.zeros((len(link_range), len(length_bins)-1))
        
        for i in range(len(num_links)):
            link_idx = num_links[i] - num_links.min()
            length_idx = np.digitize(total_length[i], length_bins) - 1
            
            if 0 <= link_idx < len(link_range) and 0 <= length_idx < len(length_bins)-1:
                coverage_grid[link_idx, length_idx] = 1
        
        im4 = ax4.imshow(coverage_grid.T, cmap='RdYlGn', origin='lower',
                        extent=[link_range[0]-0.5, link_range[-1]+0.5, 
                               length_bins[0], length_bins[-1]],
                        aspect='auto')
        ax4.set_xlabel('Number of Links')
        ax4.set_ylabel('Total Length (px)')
        ax4.set_title('Archive Coverage Map\\n(Green = Filled, Red = Empty)')
        ax4.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Interpolated heatmap saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Failed to generate interpolated heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate True MAP-Elites Heatmaps")
    parser.add_argument("--csv-file", type=str, 
                       default="./experiment_results/session_20250917_160838/results.csv",
                       help="CSV results file")
    parser.add_argument("--output-dir", type=str, 
                       default="./map_elites_shared_ppo_results/true_heatmaps",
                       help="Output directory")
    args = parser.parse_args()
    
    print("Generating True MAP-Elites Heatmaps")
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
    
    # 1. Generate true MAP-Elites heatmap
    print("\\n1. Generating True MAP-Elites Grid Heatmap...")
    heatmap_path = os.path.join(args.output_dir, "true_map_elites_heatmap.png")
    if create_true_map_elites_heatmap(data, heatmap_path):
        success_count += 1
    
    # 2. Generate interpolated heatmap
    print("\\n2. Generating Interpolated Heatmap...")
    interpolated_path = os.path.join(args.output_dir, "interpolated_heatmap.png")
    if create_interpolated_heatmap(data, interpolated_path):
        success_count += 1
    
    print("\\n" + "=" * 60)
    print(f"Heatmap generation complete! Success: {success_count}/2")
    print(f"All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()


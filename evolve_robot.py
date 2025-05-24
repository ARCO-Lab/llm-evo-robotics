import numpy as np
import pybullet as p
import pybullet_data
import tempfile
import os
import time
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from robot_evolution_fixed import decode_gene, generate_urdf, simulate_robot_multi
from fix_robot_model import test_robot_with_gene

class RobotDesignProblem(Problem):
    """机器人设计多目标优化问题"""
    
    def __init__(self, n_var=50, use_gui=False):
        super().__init__(
            n_var=n_var,       # 基因变量数量
            n_obj=4,           # 目标函数数量: 距离、路径直线性、稳定性、能量消耗
            n_constr=0,        # 约束条件数量
            xl=np.zeros(n_var), # 基因下限
            xu=np.ones(n_var)   # 基因上限
        )
        self.use_gui = use_gui
        
    def _evaluate(self, X, out, *args, **kwargs):
        """评估机器人设计的适应度"""
        n_individuals = X.shape[0]
        F = np.zeros((n_individuals, self.n_obj))
        
        for i in range(n_individuals):
            gene = X[i, :]
            print(f"评估个体 {i+1}/{n_individuals}")
            
            # 解码基因为机器人配置
            robot_config = decode_gene(gene)
            
            # 模拟机器人并获取性能指标
            metrics = simulate_robot_multi(
                robot_config, 
                gui=self.use_gui and i == 0,  # 只为第一个个体显示GUI
                sim_time=10.0
            )
            
            # 记录性能指标
            F[i, 0] = -metrics[0]  # 距离 (最大化，所以取负)
            F[i, 1] = -metrics[1]  # 路径直线性 (最大化，所以取负)
            F[i, 2] = metrics[2]   # 稳定性 (最小化)
            F[i, 3] = metrics[3]   # 能量消耗 (最小化)
        
        out["F"] = F

def run_evolution(pop_size=20, n_gen=10, use_gui=False):
    """运行进化优化过程"""
    print("开始机器人进化优化...")
    
    # 定义问题
    problem = RobotDesignProblem(n_var=50, use_gui=use_gui)
    
    # 定义算法
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行优化
    results = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=True,
        save_history=True
    )
    
    # 记录结束时间
    end_time = time.time()
    print(f"优化完成! 耗时: {end_time - start_time:.2f} 秒")
    
    # 获取结果
    X = results.X  # 决策变量
    F = results.F  # 目标函数值
    
    # 保存最终种群和Pareto前沿
    save_results(X, F)
    
    # 可视化Pareto前沿
    visualize_pareto_front(F)
    
    # 生成并保存最佳设计
    generate_best_designs(X, F)

def save_results(X, F):
    """保存进化结果"""
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 保存决策变量
    np.savetxt("results/decision_variables.csv", X, delimiter=",")
    
    # 保存目标函数值
    np.savetxt("results/objective_values.csv", F, delimiter=",")
    
    print("结果已保存到 'results' 目录")

def visualize_pareto_front(F):
    """可视化Pareto前沿"""
    # 创建两个2D图和一个3D图
    fig = plt.figure(figsize=(15, 10))
    
    # 2D图: 距离 vs 能量消耗
    ax1 = fig.add_subplot(221)
    ax1.scatter(-F[:, 0], F[:, 3], s=30)
    ax1.set_xlabel("距离 (最大化)")
    ax1.set_ylabel("能量消耗 (最小化)")
    ax1.set_title("距离 vs 能量消耗")
    
    # 2D图: 距离 vs 路径直线性
    ax2 = fig.add_subplot(222)
    ax2.scatter(-F[:, 0], -F[:, 1], s=30)
    ax2.set_xlabel("距离 (最大化)")
    ax2.set_ylabel("路径直线性 (最大化)")
    ax2.set_title("距离 vs 路径直线性")
    
    # 2D图: 路径直线性 vs 稳定性
    ax3 = fig.add_subplot(223)
    ax3.scatter(-F[:, 1], F[:, 2], s=30)
    ax3.set_xlabel("路径直线性 (最大化)")
    ax3.set_ylabel("稳定性 (最小化)")
    ax3.set_title("路径直线性 vs 稳定性")
    
    # 3D图: 距离 vs 路径直线性 vs 稳定性
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(-F[:, 0], -F[:, 1], F[:, 2], s=30)
    ax4.set_xlabel("距离")
    ax4.set_ylabel("路径直线性")
    ax4.set_zlabel("稳定性")
    ax4.set_title("3D Pareto前沿")
    
    plt.tight_layout()
    plt.savefig("results/pareto_front.png")
    print("Pareto前沿图已保存到 'results/pareto_front.png'")

def generate_best_designs(X, F):
    """生成并保存最佳设计"""
    # 找出各目标下的最佳个体
    best_distance_idx = np.argmin(F[:, 0])    # 最大距离 (F中是负的)
    best_linearity_idx = np.argmin(F[:, 1])   # 最大直线性 (F中是负的)
    best_stability_idx = np.argmin(F[:, 2])   # 最小翻滚/俯仰
    best_energy_idx = np.argmin(F[:, 3])      # 最小能量消耗
    
    # 计算综合评分 - 归一化每个指标并加权求和
    normalized_F = np.zeros_like(F)
    for j in range(F.shape[1]):
        min_val = np.min(F[:, j])
        max_val = np.max(F[:, j])
        if max_val > min_val:
            normalized_F[:, j] = (F[:, j] - min_val) / (max_val - min_val)
        else:
            normalized_F[:, j] = 0
    
    # 权重可以根据需要调整
    weights = np.array([0.5, 0.2, 0.2, 0.1])  # 距离权重最高
    scores = np.sum(normalized_F * weights, axis=1)
    best_overall_idx = np.argmin(scores)
    
    # 记录最佳设计
    best_designs = {
        'best_overall': decode_gene(X[best_overall_idx]),
        'best_distance': decode_gene(X[best_distance_idx]),
        'best_linearity': decode_gene(X[best_linearity_idx]),
        'best_stability': decode_gene(X[best_stability_idx]),
        'best_energy': decode_gene(X[best_energy_idx])
    }
    
    # 生成并保存URDF文件
    for design_name, design in best_designs.items():
        urdf = generate_urdf(design)
        with open(f"{design_name}.urdf", "w") as f:
            f.write(urdf)
        print(f"已保存 {design_name}.urdf")
    
    # 保存基因
    np.savetxt("results/best_overall_gene.csv", X[best_overall_idx], delimiter=",")
    np.savetxt("results/best_distance_gene.csv", X[best_distance_idx], delimiter=",")
    np.savetxt("results/best_linearity_gene.csv", X[best_linearity_idx], delimiter=",")
    np.savetxt("results/best_stability_gene.csv", X[best_stability_idx], delimiter=",")
    np.savetxt("results/best_energy_gene.csv", X[best_energy_idx], delimiter=",")
    
    print("所有最佳设计已保存")
    
    return best_designs

def test_best_design(design_type="best_overall"):
    """测试最佳设计"""
    if not os.path.exists(f"{design_type}.urdf"):
        print(f"找不到 {design_type}.urdf 文件，请先运行进化优化")
        return
    
    # 加载基因
    if os.path.exists(f"results/{design_type}_gene.csv"):
        gene = np.loadtxt(f"results/{design_type}_gene.csv", delimiter=",")
        test_robot_with_gene(gene)
    else:
        print(f"找不到 results/{design_type}_gene.csv 文件")

def main():
    print("机器人进化优化")
    print("1. 运行小规模进化 (20个体, 5代)")
    print("2. 运行中等规模进化 (30个体, 10代)")
    print("3. 运行大规模进化 (50个体, 20代)")
    print("4. 测试最佳综合设计")
    print("5. 测试最佳距离设计")
    print("6. 测试最佳直线性设计")
    print("7. 测试最佳稳定性设计")
    print("8. 测试最佳能源效率设计")
    
    choice = input("\n请选择(1-8): ")
    
    if choice == '1':
        run_evolution(pop_size=20, n_gen=5, use_gui=True)
    elif choice == '2':
        run_evolution(pop_size=30, n_gen=10, use_gui=True)
    elif choice == '3':
        run_evolution(pop_size=50, n_gen=20, use_gui=True)
    elif choice == '4':
        test_best_design("best_overall")
    elif choice == '5':
        test_best_design("best_distance")
    elif choice == '6':
        test_best_design("best_linearity")
    elif choice == '7':
        test_best_design("best_stability")
    elif choice == '8':
        test_best_design("best_energy")
    else:
        print("无效选择，运行小规模进化")
        run_evolution(pop_size=20, n_gen=5, use_gui=True)

if __name__ == "__main__":
    main() 
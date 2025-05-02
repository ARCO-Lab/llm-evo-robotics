import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from robot.robot_generator import generate_robot_xml
from optimization.fitness_evaluator import evaluate_population

NUM_PLANES = 10
SAVE_INTERVAL = 100  # 每 100 代保存一次
TOTAL_GENERATIONS = 1000  # 🔥 总共 1000 代

class RobotOptimizationProblem(Problem):
    def __init__(self, num_variables, components, actuators):
        super().__init__(n_var=num_variables, n_obj=2, n_constr=0, xl=0, xu=1, type_var=np.bool_)
        self.components = components
        self.actuators = actuators
        self.best_robot = None  # 记录最优个体
        self.best_reward = -np.inf

    def _evaluate(self, individuals, out, *args, **kwargs):
        """计算适应度函数"""
        gen_number = kwargs["n_gen"]
        rewards, fitness_results = evaluate_population(individuals, self.components, self.actuators, gen_number)

        # **每 100 代保存当前最优个体**
        if gen_number % SAVE_INTERVAL == 0:
            best_idx = np.argmax(rewards)
            self.best_robot = individuals[best_idx]
            self.best_reward = rewards[best_idx]
            best_xml = generate_robot_xml([self.best_robot], self.components, self.actuators)
            with open(f"../configs/best_robot_gen_{gen_number}.xml", "w") as f:
                f.write(best_xml)
            print(f"🔥 代 {gen_number}: 最优机器人保存！")

        out["F"] = np.array(fitness_results)

def run_nsga2(num_variables, components, actuators):
    """运行 NSGA-II 进行机器人优化"""
    algorithm = NSGA2(pop_size=NUM_PLANES)

    problem = RobotOptimizationProblem(num_variables, components, actuators)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", TOTAL_GENERATIONS),  # 🔥 运行 1000 代
        seed=1,
        verbose=True
    )

    return res.X[np.argmin(res.F[:, 0])]


if __name__ == "__main__":
    print("✅ 测试 NSGA-II 进化...")
    from robot.xml_parser import load_components, load_actuators
    components = load_components("basic_components.xml")
    actuators = load_actuators("basic_actuators.xml")

    num_variables = len(components) + len(actuators)
    best_design = run_nsga2(num_variables, components, actuators)

    print("测试完成，最优基因编码:", best_design)

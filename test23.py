# evolve_robot_2_0.py - 完整版本（含 main 函数，已修复）

import numpy as np
import random
import pybullet as p
import pybullet_data
import time
import json

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS

# ...[simulate_robot, decode_gene, create_link_shape remain unchanged from previous cell]...


def render_gene(gene):
    print("👁️ 正在展示机器人...")
    simulate_robot(gene, gui=True)


class RobotMoveToTargetProblem(Problem):
    def __init__(self, max_links=6):
        super().__init__(n_var=1 + max_links * 7,
                         n_obj=1,
                         n_constr=0,
                         xl=0.0,
                         xu=1.0)
        self.max_links = max_links

    def _evaluate(self, X, out, *args, **kwargs):
        f = []
        for idx, x in enumerate(X):
            gene = decode_gene(x, self.max_links)
            try:
                fitness = simulate_robot(gene, gui=False)
            except Exception:
                fitness = 0.0
            print(f"[Eval {idx}] Links={gene['num_links']}  Motors={gene['has_motor']}  Fitness={fitness:.4f}")
            f.append(-fitness)
        out["F"] = np.column_stack([f])


if __name__ == "__main__":
    problem = RobotMoveToTargetProblem(max_links=6)

    algorithm = NSGA2(
        pop_size=10,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True,
        sampling=LHS()
    )

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 20),
                   seed=1,
                   verbose=True,
                   save_history=True)

    print("\n🎬 可视化每代最优个体：")
    for i, gen in enumerate(res.history):
        opt = gen.opt[0]
        gene = decode_gene(opt.X)
        print(f"Generation {i+1}, fitness: {-opt.F[0]:.4f}")
        render_gene(gene)

    if res.X.ndim == 1:
        best_gene = decode_gene(res.X)
    else:
        best_idx = np.argmin(res.F)
        best_gene = decode_gene(res.X[best_idx])

    with open("best_gene.json", "w") as f:
        json.dump(convert_to_builtin(best_gene), f, indent=2)

    print("\n✅ 最优结构基因已保存为 best_gene.json")
    render_gene(best_gene)


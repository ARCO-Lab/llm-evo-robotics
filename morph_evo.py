import numpy as np
from pymoo.core.problem import Problem
import logging


class RobotDesignProblem(Problem):

    def __init__(self, n_var=100, use_gui=False, verbose=False, pause_after_eval=False, 
                 add_diversity=True, n_constr = 4, min_stability = 0.55, max_energy = 700,
                 terrain_type = "rough", sim_time = 10.0):

        self.logger = self._setup_logger(verbose)

        super().__init__(
            n_var= n_var,
            n_obj = 5 if add_diversity else 4,
            n_constr = n_constr,
            xl = np.zeros(n_var),
            xu = np.ones(n_var)
        )

        self.use_gui = use_gui
        self.pause_after_eval = pause_after_eval
        self.add_diversity = add_diversity
        self.evaluated_designs = [] #记录已评估的设计,用于计算多样性

        self.min_stability = min_stability
        self.max_energy = max_energy
        self.terrain_type = terrain_type
        self.sim_time = sim_time

        self.evaluated_designs = []
        self.best_individuals = None
        self.best_fitness = None

        self.current_structure_types = self._structure_counter()
        self.total_structure_types = self._structure_counter()

        self.generation_stats = []

        self.current_generation = 0

        self.total_generations = 0

    def _setup_logger(self, verbose):
        logger = logging.getLogger(f"{__name__}.RobotDesignProblem")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
    
    def _structure_counter(self):
        return {
            'wheeled': 0,
            'legged': 0,
            'hybrid': 0,
            'other':0
        }
    def reset_generation_stats(self):
        self.current_structure_types = self._structure_counter()
    
    def record_generation_stats(self):
        self.generation_stats.append({
            'generation': self.current_generation,
            'structure_types': self.current_structure_types.copy(),
            'best_fitness':self.best_fitness.copy() if self.best_fitness is not None else None
        })

    def set_total_generations(self, generations):

        self.total_generations = generations
    
    def start_new_generation(self):
        self.current_generation += 1
        self.reset_generation_stats()


    def prepare_robot_config(self, gene):
        
        robot_config = decode_gene(gene)

        if 'parent_indices' not in robot_config:
            self.add_default_parent_indices(robot_config)

        if 'joint_positions' not in robot_config:
            self._add_default_joint_positions(robot_config)

        return robot_config


    def _evaluate(self, X, out, *args, **kwargs):
        n_individuals = X.shape[0]

        F = np.zeros((n_individuals, self.n_obj))

        G = np.zeros((n_individuals, self.n_constr))        
        
        self._log_progress(n_individuals)

        for i in range(n_individuals):
            gene = X[i, :]
            if self.verbose:
                self.logger.info(f"\n评估个体 {i+1}/ {n_individuals}")
            robot_config = self.prepare_robot_config(gene)


        
def run_genetic_optimization(pop_size=10, n_gen=5, use_gui=True, use_constraints=True, verbose=True, pause_after_eval=True, diverse_mode=True, save_designs=True):
    
    print("\n开始遗传算法优化机器人设计...")
    print(f"种群大小: {pop_size}, 进化代数: {n_gen}")
    print(f"使用结构约束: {'是' if use_constraints else '否'}")
    print(f"增加结构多样性: {'是' if diverse_mode else '否'}")
    print(f"显示模拟可视化: {'是' if use_gui else '否'}")
    print(f"打印详细结构信息: {'是' if verbose else '否'}")
    print(f"每次评估后暂停: {'是' if pause_after_eval else '否'}")
    print(f"保存机器人设计: {'是' if save_designs else '否'}")
    print(f"启用零件连接修复: 是")
    print(f"启用有问题设计过滤: 是")

    try:
        problem = RobotDesignProblem(n_var=100, use_gui = use_gui, verbose = verbose,
                                     pause_after_eval=pause_after_eval, add_diversity= diverse_mode)
        
        if use_constraints:

            initial_pop = np.zeros((pop_size, 100))

            for i in range(pop_size):
                max_attempts = 10
                design_ok = False

                for attempt in range(max_attempts):

                    if diverse_mode:

                        if np.random.random() > 0.3:
                            gene = create_diverse_gene()
                        
                        else:
                            gene = create_constrained_gene()

                    else:

                        gene = create_constrained_gene()

                    robot_config = decode_gene(gene)
                    robot_config = fix_prismatic_joints(robot_config)



    except Exception as e:
        print(f"遗传算法优化过长中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None











if __name__ == "__main__":
    pop_size = int(input("请输入种群大小 (建议3-10):"))
    n_gen = int(input("请输入进化代数 (建议2-5):"))
    print_verbose = input("是否打印详细结构信息? (y/n): ").lower() == 'y'
    pause_after_eval = input("是否在每次评估后暂停? (y/n): ").lower() == 'y'
    save_designs = input("是否保存所有进化的设计? (y/n):").lower() == 'y'
    best_gene = run_genetic_optimization(pop_size, n_gen, use_gui=True,
                                         use_constraints=True, verbose=print_verbose,
                                         pause_after_eval=pause_after_eval,
                                         diverse_mode=True, save_designs=save_designs)


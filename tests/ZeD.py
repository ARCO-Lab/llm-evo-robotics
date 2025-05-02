import xml.etree.ElementTree as ET
import random
import numpy as np
import copy
import multiprocessing
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination
import mujoco
import stable_baselines3 as sb3

# 解析 XML 组件
def load_components(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    components = {body.get("name"): body for body in root.findall(".//body")}
    return components

def load_actuators(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    actuators = {actuator.get("name"): actuator for actuator in root.findall(".//actuator/*")}
    return actuators

# 读取基本组件 & Actuator
components = load_components("basic_components/basic_components.xml")
actuators = load_actuators("basic_components/basic_actuators.xml")

# 定义 plane 数量
NUM_PLANES = 10  # 这里假设同时评估 10 个个体
PLANE_SPACING = 5.0  # 每个 plane 之间的间隔

# 生成一个 MuJoCo XML，包含多个 plane
def generate_multi_plane_xml(individuals):
    root = ET.Element("mujoco", model="multi_robot_env")
    worldbody = ET.SubElement(root, "worldbody")
    actuators_section = ET.SubElement(root, "actuator")

    # 创建多个独立的 plane
    for i in range(NUM_PLANES):
        plane = ET.SubElement(worldbody, "geom", type="plane", pos=f"{i * PLANE_SPACING} 0 0", size="5 5 0.1")

    # 创建多个个体，每个放置在不同的 plane 上
    for i, binary_encoding in enumerate(individuals):
        if i >= NUM_PLANES:
            break  # 确保不会超过 plane 数量

        # 生成机器人 body
        robot_body = ET.SubElement(worldbody, "body", name=f"robot_{i}", pos=f"{i * PLANE_SPACING} 0 0.5")
        for idx, (name, body) in enumerate(components.items()):
            if binary_encoding[idx] == 1:
                robot_body.append(copy.deepcopy(body))

        # 生成 actuator
        actuator_offset = len(components)
        for idx, (name, actuator) in enumerate(actuators.items()):
            if binary_encoding[actuator_offset + idx] == 1:
                actuators_section.append(copy.deepcopy(actuator))

    # 转换为 XML 字符串
    return ET.tostring(root, encoding="unicode")

# NSGA-II 进化问题定义
class RobotEvolution(Problem):
    def __init__(self):
        super().__init__(n_var=len(components) + len(actuators),
                         n_obj=2,  # 目标函数数量
                         n_constr=0,
                         xl=0, xu=1)
        self.eval_counter = 0

    def _evaluate(self, X, out, *args, **kwargs):
        fitness_values = []
        
        # 以 batch 方式进行评估（每次最多 NUM_PLANES 个个体）
        for i in range(0, X.shape[0], NUM_PLANES):
            batch = X[i:i+NUM_PLANES]
            robot_xml = generate_multi_plane_xml(batch)
            batch_fitness = self.parallel_simulate_and_evaluate(robot_xml, len(batch))
            fitness_values.extend(batch_fitness)

        out["F"] = np.array(fitness_values)

    def parallel_simulate_and_evaluate(self, robot_xml, num_individuals):
        """
        并行运行 MuJoCo 仿真，评估多个个体
        """
        with open("multi_robot.xml", "w") as f:
            f.write(robot_xml)
        
        # 运行 MuJoCo
        model = mujoco.MjModel.from_xml_path("multi_robot.xml")
        data = mujoco.MjData(model)
        sim = mujoco.MjSim(model)

        # 创建多个 PPO 代理
        envs = [sb3.common.env_util.make_vec_env("multi_robot.xml") for _ in range(num_individuals)]
        models = [sb3.PPO("MlpPolicy", env, verbose=0) for env in envs]

        # 训练
        for model in models:
            model.learn(total_timesteps=1000)

        # 评估每个个体的 fitness
        fitness_results = []
        for model, env in zip(models, envs):
            obs = env.reset()
            total_reward = 0
            for _ in range(500):  # 只测试 500 步
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            env.close()

            num_components = np.sum(X[:len(components)])  # 计算使用的组件数
            fitness_results.append([-total_reward, num_components])  # NSGA-II 目标：最小化奖励和复杂度
        
        return fitness_results

# 运行 NSGA-II 进化
algorithm = NSGA2(
    pop_size=20,
    n_offsprings=20,
    sampling=BinaryRandomSampling(),
    crossover=PointCrossover(prob=0.9),
    mutation=BitflipMutation(prob=0.1),
    eliminate_duplicates=True
)

res = minimize(RobotEvolution(),
               algorithm,
               termination=get_termination("n_gen", 10),
               verbose=True)

# 输出最佳设计
best_design = res.X[0]  # 取 Pareto 前沿上的最佳个体
print("最优机器人基因编码:", best_design)

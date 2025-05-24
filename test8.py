# 完整主程序封装，含运动检查和5秒仿真时间

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


def convert_to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    else:
        return obj


def decode_gene(x, max_links=4):
    num_links = int(x[0] * max_links) + 1
    connections, joint_types, has_motor, shapes = [], [], [], []
    is_wheel, is_claw, is_sensor = [], [], []

    idx = 1
    for i in range(num_links):
        parent = int(x[idx] * (i + 1)) if i > 0 else 0
        jt = p.JOINT_REVOLUTE if x[idx + 1] < 0.5 else p.JOINT_FIXED
        motor = jt == p.JOINT_REVOLUTE and x[idx + 2] < 0.7
        shape = int(x[idx + 3] * 3)

        connections.append((parent, i + 1))
        joint_types.append(jt)
        has_motor.append(motor)
        shapes.append(shape)
        is_wheel.append(x[idx + 4] < 0.3 and shape == 1)
        is_claw.append(x[idx + 5] < 0.1)
        is_sensor.append(x[idx + 6] < 0.1)
        idx += 7

    if not any(has_motor):
        for i, jt in enumerate(joint_types):
            if jt == p.JOINT_REVOLUTE:
                has_motor[i] = True
                break

    return {
        'num_links': num_links,
        'connections': connections,
        'joint_types': joint_types,
        'has_motor': has_motor,
        'shapes': shapes,
        'is_wheel': is_wheel,
        'is_claw': is_claw,
        'is_sensor': is_sensor
    }


def create_shape(shape_id, is_sensor=False):
    color = [0, 1, 0, 0.5] if is_sensor else [1, 0, 0, 1]
    if shape_id == 0:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.05])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.05], rgbaColor=color)
    elif shape_id == 1:
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.1)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.1, rgbaColor=color)
    else:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.07)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.07, rgbaColor=color)
    return col, vis


def simulate_robot(gene, gui=False, show_goal=True, sim_time=5.0):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    goal_pos = [0.5, 0, 0.1]
    if show_goal:
        p.loadURDF("cube_small.urdf", basePosition=goal_pos, globalScaling=1.5)

    base_col, base_vis = create_shape(0)
    robot_id = p.createMultiBody(1.0, base_col, base_vis, [0, 0, 0.3])
    link_ids = [robot_id]

    for i in range(gene['num_links']):
        parent = gene['connections'][i][0]
        parent_body = link_ids[parent]
        jt = gene['joint_types'][i]
        motor = gene['has_motor'][i]
        shape = gene['shapes'][i]
        claw = gene['is_claw'][i]
        sensor = gene['is_sensor'][i]

        col_id, vis_id = create_shape(shape, is_sensor=sensor)
        child_body = p.createMultiBody(1.0, col_id, vis_id, [0, 0, 1 + i * 0.05])
        link_ids.append(child_body)

        p.createConstraint(parent_body, -1, child_body, -1, jt, [0, 0, 1],
                           [0, 0, 0.1], [0, 0, -0.1])

        if motor and jt == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(child_body, 0, p.VELOCITY_CONTROL,
                                    targetVelocity=5.0, force=100.0)

        if claw:
            for offset in [-0.05, 0.05]:
                c_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.01])
                c_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.01],
                                            rgbaColor=[0.6, 0.6, 0.6, 1])
                claw_part = p.createMultiBody(0.1, c_col, c_vis,
                                              [offset, 0, 1.1 + i * 0.05])
                p.createConstraint(child_body, -1, claw_part, -1,
                                   p.JOINT_FIXED, [0, 0, 1],
                                   [0, 0, 0.1], [0, 0, 0])

    start_pos, _ = p.getBasePositionAndOrientation(robot_id)

    for _ in range(int(sim_time / (1. / 240.))):
        p.stepSimulation()
        if gui:
            time.sleep(1. / 240.)

    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    delta = np.linalg.norm(np.array(end_pos) - np.array(start_pos))

    if delta < 0.01:
        print("⚠️ 本结构在模拟期间几乎未移动！")

    dist_to_goal = np.linalg.norm(np.array(end_pos) - np.array(goal_pos))
    p.disconnect()
    return 1.0 / (1e-3 + dist_to_goal)


def render_gene(gene):
    print("👁️ 正在展示机器人...")
    simulate_robot(gene, gui=True)


class RobotMoveToTargetProblem(Problem):
    def __init__(self, max_links=4):
        super().__init__(n_var=1 + max_links * 7,
                         n_obj=1,
                         n_constr=0,
                         xl=0.0,
                         xu=1.0)
        self.max_links = max_links

    def _evaluate(self, X, out, *args, **kwargs):
        f = []
        for x in X:
            gene = decode_gene(x, self.max_links)
            try:
                fitness = simulate_robot(gene, gui=False)
            except Exception:
                fitness = 0.0
            f.append(-fitness)
        out["F"] = np.column_stack([f])


if __name__ == "__main__":
    problem = RobotMoveToTargetProblem(max_links=4)

    algorithm = NSGA2(
        pop_size=10,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 5),
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


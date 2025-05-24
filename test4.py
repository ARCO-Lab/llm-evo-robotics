import numpy as np
import random
import pybullet as p
import pybullet_data
import time

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# -----------------------------
# Gene decode logic with shape, claw, sensor, wheel
# -----------------------------
def decode_gene(x, max_links=4):
    num_links = int(x[0] * max_links) + 1
    connections = []
    joint_types = []
    has_motor = []
    shapes = []
    is_wheel = []
    is_claw = []
    is_sensor = []

    idx = 1
    for i in range(num_links):
        parent = int(x[idx] * (i + 1)) if i > 0 else 0
        jt = p.JOINT_REVOLUTE if x[idx + 1] < 0.5 else p.JOINT_FIXED
        has_mtr = jt == p.JOINT_REVOLUTE and x[idx + 2] < 0.7
        shape = int(x[idx + 3] * 3)  # 0=box, 1=cylinder, 2=sphere

        connections.append((parent, i + 1))
        joint_types.append(jt)
        has_motor.append(has_mtr)
        shapes.append(shape)
        is_wheel.append(x[idx + 4] < 0.2 and shape == 1)  # cylinder + wheel flag
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

# -----------------------------
# Common shape creation
# -----------------------------
def create_shape(shape_id, is_sensor=False):
    color = [0, 1, 0, 0.5] if is_sensor else [1, 0, 0, 1]
    if shape_id == 0:  # box
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.05])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.05, 0.05], rgbaColor=color)
    elif shape_id == 1:  # cylinder
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.1)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.1, rgbaColor=color)
    else:  # sphere
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.07)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.07, rgbaColor=color)
    return col, vis

# -----------------------------
# Robot simulation (used by pymoo)
# -----------------------------
def simulate_robot(gene, gui=False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    base_col, base_vis = create_shape(0)
    robot_id = p.createMultiBody(1.0, base_col, base_vis, [0, 0, 0.3])
    link_ids = [robot_id]

    for i in range(gene['num_links']):
        parent = gene['connections'][i][0]
        parent_body = link_ids[parent]
        jt = gene['joint_types'][i]
        use_motor = gene['has_motor'][i]
        shape = gene['shapes'][i]
        claw = gene['is_claw'][i]
        sensor = gene['is_sensor'][i]

        col_id, vis_id = create_shape(shape, is_sensor=sensor)
        child_body = p.createMultiBody(1.0, col_id, vis_id, [0, 0, 1 + i * 0.05])
        link_ids.append(child_body)

        p.createConstraint(parent_body, -1, child_body, -1, jt, [0, 0, 1], [0, 0, 0.1], [0, 0, -0.1])

        if use_motor and jt == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(child_body, 0, p.VELOCITY_CONTROL, targetVelocity=1.0, force=5.0)

        if claw:
            # add fixed small claw parts
            for offset in [-0.05, 0.05]:
                c_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.01])
                c_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.01, 0.01], rgbaColor=[0.6, 0.6, 0.6, 1])
                claw_part = p.createMultiBody(0.1, c_col, c_vis, [offset, 0, 1.1 + i * 0.05])
                p.createConstraint(child_body, -1, claw_part, -1, p.JOINT_FIXED, [0, 0, 1], [0, 0, 0.1], [0, 0, 0])

    for _ in range(int(2.0 / (1. / 240.))):
        p.stepSimulation()
        if gui:
            time.sleep(1. / 240.)

    base_pos, _ = p.getBasePositionAndOrientation(robot_id)
    p.disconnect()
    return base_pos[0]

# -----------------------------
# pymoo Problem
# -----------------------------
class RobotMorphologyProblem(Problem):
    def __init__(self, max_links=4):
        super().__init__(n_var=1 + max_links * 7,
                         n_obj=2,
                         n_constr=0,
                         xl=0.0,
                         xu=1.0)
        self.max_links = max_links

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        for x in X:
            gene = decode_gene(x, self.max_links)
            try:
                dist = simulate_robot(gene)
            except Exception:
                dist = -1.0
            f1.append(-dist)
            f2.append(gene['num_links'])
        out["F"] = np.column_stack([f1, f2])

# -----------------------------
# Visualize best robot
# -----------------------------
def render_best_gene(gene):
    print("👀 正在可视化最优结构（按 Ctrl+C 退出）")
    simulate_robot(gene, gui=True)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    problem = RobotMorphologyProblem(max_links=4)

    algorithm = NSGA2(
        pop_size=20,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 10),
                   seed=1,
                   verbose=True)

    best_idx = np.argmin(res.F[:, 0])
    best_gene = decode_gene(res.X[best_idx])
    print("\n✅ 最优结构基因:")
    print(best_gene)

    render_best_gene(best_gene)



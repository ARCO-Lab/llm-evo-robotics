from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.float_random_sampling import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import numpy as np
import random
import pybullet as p
import pybullet_data
import time


# Define how to decode a vector X into a robot gene
def decode_gene(x, max_links=4):
    num_links = int(x[0] * max_links) + 1
    connections = []
    joint_types = []
    has_motor = []

    idx = 1
    for i in range(num_links):
        parent = int(x[idx] * (i + 1)) if i > 0 else 0
        connections.append((parent, i + 1))
        jt = p.JOINT_REVOLUTE if x[idx + 1] < 0.5 else p.JOINT_FIXED
        joint_types.append(jt)
        has_motor.append(jt == p.JOINT_REVOLUTE and x[idx + 2] < 0.7)
        idx += 3

    if not any(has_motor):
        for i, jt in enumerate(joint_types):
            if jt == p.JOINT_REVOLUTE:
                has_motor[i] = True
                break

    return {
        'num_links': num_links,
        'connections': connections,
        'joint_types': joint_types,
        'has_motor': has_motor
    }


# Create robot and simulate
def simulate_robot(gene):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    link_size = [0.1, 0.05, 0.05]
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=link_size)
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=link_size)
    base_pos = [0, 0, 0.3]
    robot_id = p.createMultiBody(baseMass=1.0,
                                 baseCollisionShapeIndex=base_col,
                                 baseVisualShapeIndex=base_vis,
                                 basePosition=base_pos)

    link_ids = [robot_id]
    for i in range(gene['num_links']):
        parent = gene['connections'][i][0]
        parent_body = link_ids[parent]
        jt = gene['joint_types'][i]
        use_motor = gene['has_motor'][i]

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=link_size)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=link_size)
        child_body = p.createMultiBody(baseMass=1.0,
                                       baseCollisionShapeIndex=col_id,
                                       baseVisualShapeIndex=vis_id,
                                       basePosition=[0, 0, 1 + i * 0.05])
        link_ids.append(child_body)

        p.createConstraint(parent_body,
                           -1,
                           child_body,
                           -1,
                           jointType=jt,
                           jointAxis=[0, 0, 1],
                           parentFramePosition=[0, 0, 0.1],
                           childFramePosition=[0, 0, -0.1])

        if use_motor and jt == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(child_body,
                                    0,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=1.0,
                                    force=5.0)

    for _ in range(int(2.0 / (1. / 240.))):
        p.stepSimulation()

    pos, _ = p.getBasePositionAndOrientation(robot_id)
    p.disconnect()
    return pos[0]  # X distance


# Define Pymoo Problem
class RobotMorphologyProblem(Problem):
    def __init__(self, max_links=4):
        super().__init__(n_var=1 + max_links * 3,
                         n_obj=2,
                         n_constr=0,
                         xl=0.0,
                         xu=1.0)
        self.max_links = max_links

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []  # -distance (maximize)
        f2 = []  # num_links (minimize)

        for x in X:
            gene = decode_gene(x, self.max_links)
            try:
                dist = simulate_robot(gene)
            except Exception:
                dist = -1.0
            f1.append(-dist)
            f2.append(gene['num_links'])

        out["F"] = np.column_stack([f1, f2])


# Configure and run NSGA-II
problem = RobotMorphologyProblem(max_links=4)

algorithm = NSGA2(pop_size=20,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(prob=0.1, eta=20),
                  eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               termination=('n_gen', 10),
               seed=1,
               verbose=True)

# Display best result
best_idx = np.argmin(res.F[:, 0])
best_gene = decode_gene(res.X[best_idx])

print("\nBest gene:", best_gene)


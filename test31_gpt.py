import numpy as np
import random
import pybullet as p
import pybullet_data
import time
import json
import os
import tempfile

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling

# --------- Custom Sampling to encourage complex structures ---------
class CustomInitSampling(LatinHypercubeSampling):
    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples)
        for i in range(int(n_samples * 0.2)):
            X[i, 0] = np.random.uniform(0.7, 1.0)  # Encourage longer structures
        return X

# --------- Utility Functions ---------
def convert_to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    else:
        return obj

# --------- Gene Decoder ---------
def decode_gene(x, max_links=8):
    num_links = int(np.clip(2 + (x[0] ** 0.5) * (max_links - 2), 2, max_links))
    joint_types, has_motor, shapes = [], [], []
    is_wheel, wheel_types, joint_axes = [], [], []
    idx = 1
    for i in range(num_links):
        jt_val = x[idx] * 4
        jt = p.JOINT_REVOLUTE if jt_val < 1 else p.JOINT_FIXED if jt_val < 2 else p.JOINT_PRISMATIC
        motor = jt != p.JOINT_FIXED and x[idx + 1] < 0.8
        shape = int(x[idx + 2] * 5)
        wheel = shape == 1 and x[idx + 3] < 0.4
        wheel_type = int(x[idx + 4] * 3) if wheel else 0
        axis_val = x[idx + 5] * 3
        axis = [1, 0, 0] if axis_val < 1 else [0, 1, 0] if axis_val < 2 else [0, 0, 1]
        joint_types.append(jt)
        has_motor.append(motor)
        shapes.append(shape)
        is_wheel.append(wheel)
        wheel_types.append(wheel_type)
        joint_axes.append(axis)
        idx += 8
    if not any(has_motor):
        for i, jt in enumerate(joint_types):
            if jt != p.JOINT_FIXED:
                has_motor[i] = True
                break
    return {
        'num_links': num_links,
        'joint_types': joint_types,
        'has_motor': has_motor,
        'shapes': shapes,
        'is_wheel': is_wheel,
        'wheel_types': wheel_types,
        'joint_axes': joint_axes
    }

# --------- URDF Generator ---------
def generate_urdf(gene):
    urdf = '<?xml version="1.0"?>\n<robot name="evolved_robot">\n'
    urdf += '  <link name="base_link">\n'
    urdf += '    <visual><geometry><box size="0.2 0.1 0.1"/></geometry></visual>\n'
    urdf += '    <collision><geometry><box size="0.2 0.1 0.1"/></geometry></collision>\n'
    urdf += '    <inertial>\n'
    urdf += '      <mass value="1.0"/>\n'
    urdf += '      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>\n'
    urdf += '    </inertial>\n'
    urdf += '  </link>\n'
    for i in range(1, gene['num_links']):
        urdf += f'  <link name="link{i}">\n'
        urdf += '    <visual><geometry><box size="0.2 0.1 0.1"/></geometry></visual>\n'
        urdf += '    <collision><geometry><box size="0.2 0.1 0.1"/></geometry></collision>\n'
        urdf += '    <inertial>\n'
        urdf += '      <mass value="1.0"/>\n'
        urdf += '      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>\n'
        urdf += '    </inertial>\n'
        urdf += '  </link>\n'
        parent = f"link{i-1}" if i > 1 else "base_link"
        axis = gene['joint_axes'][i]
        axis_str = f"{axis[0]} {axis[1]} {axis[2]}"
        joint_type = "revolute" if gene['joint_types'][i] == p.JOINT_REVOLUTE else "fixed"
        urdf += f'  <joint name="joint{i}" type="{joint_type}">\n'
        urdf += f'    <parent link="{parent}"/>\n'
        urdf += f'    <child link="link{i}"/>\n'
        urdf += f'    <origin xyz="0.2 0 0"/>\n'
        urdf += f'    <axis xyz="{axis_str}"/>\n'
        urdf += '    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>\n'
        urdf += '  </joint>\n'
    urdf += '</robot>'
    return urdf

# --------- Simulator ---------
def simulate_robot(gene, gui=False, sim_time=10.0):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    urdf_str = generate_urdf(gene)
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as f:
        f.write(urdf_str.encode())
        urdf_path = f.name
    try:
        robot_id = p.loadURDF(urdf_path, [0, 0, 0.3], useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)
        for i in range(p.getNumJoints(robot_id)):
            if i < len(gene['has_motor']) and gene['has_motor'][i]:
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=random.uniform(-1, 1), force=100)
        p.stepSimulation()
        start_pos, _ = p.getBasePositionAndOrientation(robot_id)
        for _ in range(int(sim_time / (1. / 240.))):
            p.stepSimulation()
            if gui:
                time.sleep(1. / 240.)
        end_pos, _ = p.getBasePositionAndOrientation(robot_id)
        move = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    except:
        move = 0.0
    finally:
        if os.path.exists(urdf_path):
            os.unlink(urdf_path)
        p.disconnect()
    structure_score = 0.05 * gene['num_links'] + 0.02 * sum(gene['has_motor']) + 0.01 * sum(gene['is_wheel'])
    score = move
    return score, structure_score if any(gene['has_motor']) else (0.01, 0.01)

# --------- PyMoo Problem ---------
class RobotProblem(Problem):
    def __init__(self, max_links=8):
        super().__init__(n_var=1 + max_links * 8, n_obj=2, n_constr=0, xl=0.0, xu=1.0)
        self.max_links = max_links

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            gene = decode_gene(x, self.max_links)
            f1, f2 = simulate_robot(gene, gui=False)
            F.append([-f1, -f2])
        out["F"] = np.array(F)

# --------- Main ---------
if __name__ == "__main__":
    problem = RobotProblem(max_links=8)
    algorithm = NSGA2(
        pop_size=50,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.3, eta=20),
        eliminate_duplicates=True,
        sampling=CustomInitSampling()
    )
    res = minimize(problem, algorithm, termination=('n_gen', 100), seed=1, verbose=True)
    best_idx = np.argmin([f[0] for f in res.F])
    best_x = res.X if res.X.ndim == 1 else res.X[best_idx]
    best_gene = decode_gene(best_x)
    print("\nRendering best individual...")
    simulate_robot(best_gene, gui=True)
    with open("best_gene.json", "w") as f:
        json.dump(convert_to_builtin(best_gene), f, indent=2)

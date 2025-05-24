# evolve_robot_2_0.py - 修复 createMultiBody 长度一致性错误

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


def convert_to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    else:
        return obj


def decode_gene(x, max_links=6):
    num_links = max(2, int(x[0] * max_links) + 1)
    joint_types, has_motor, shapes = [], [], []
    is_wheel = []
    idx = 1
    for i in range(num_links):
        jt = p.JOINT_REVOLUTE if x[idx + 1] < 0.5 else p.JOINT_FIXED
        motor = jt == p.JOINT_REVOLUTE and x[idx + 2] < 0.7
        shape = int(x[idx + 3] * 3)
        joint_types.append(jt)
        has_motor.append(motor)
        shapes.append(shape)
        is_wheel.append(x[idx + 4] < 0.3 and shape == 1)
        idx += 7

    if not any(has_motor):
        for i, jt in enumerate(joint_types):
            if jt == p.JOINT_REVOLUTE:
                has_motor[i] = True
                break

    return {
        'num_links': num_links,
        'joint_types': joint_types,
        'has_motor': has_motor,
        'shapes': shapes,
        'is_wheel': is_wheel,
    }


def create_link_shape(shape_id, is_wheel=False):
    color = [0.8, 0.2, 0.2, 1] if not is_wheel else [0.1, 0.1, 0.1, 1]
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


def simulate_robot(gene, gui=False, show_goal=True, sim_time=8.0):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    goal_pos = [0.5, 0, 0.1]
    if show_goal:
        goal_id = p.loadURDF("cube_small.urdf", basePosition=goal_pos, globalScaling=1.5)
        p.changeVisualShape(goal_id, -1, rgbaColor=[0, 1, 0, 1])

    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=1.0,
                                     cameraYaw=90,
                                     cameraPitch=-89,
                                     cameraTargetPosition=[0.3, 0, 0.0])

    num_links_total = gene['num_links']
    num_links = num_links_total - 1
    base_col, base_vis = create_link_shape(gene['shapes'][0])

    linkMasses = [1.0 for _ in range(num_links)]
    linkCollisionShapeIndices = []
    linkVisualShapeIndices = []
    linkPositions = []
    linkOrientations = [[0, 0, 0, 1]] * num_links
    linkParentIndices = list(range(num_links))
    linkJointTypes = []
    linkJointAxis = []

    for i in range(num_links):
        shape = gene['shapes'][i + 1]
        wheel = gene['is_wheel'][i + 1]
        col, vis = create_link_shape(shape, is_wheel=wheel)
        linkCollisionShapeIndices.append(col)
        linkVisualShapeIndices.append(vis)
        linkPositions.append([0.2, 0, 0])
        linkJointTypes.append(gene['joint_types'][i + 1])
        linkJointAxis.append([0, 0, 1])

    robot_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=[0, 0, 0.3],
        linkMasses=linkMasses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkParentIndices=linkParentIndices,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis
    )

    for i in range(num_links):
        if gene['has_motor'][i + 1] and gene['joint_types'][i + 1] == p.JOINT_REVOLUTE:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=5.0, force=100.0)

    start_pos, _ = p.getBasePositionAndOrientation(robot_id)
    for _ in range(int(sim_time / (1. / 240.))):
        p.stepSimulation()
        if gui:
            time.sleep(1. / 240.)

    end_pos, _ = p.getBasePositionAndOrientation(robot_id)
    base_move = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    dist_to_goal = np.linalg.norm(np.array(end_pos) - np.array(goal_pos))
    p.disconnect()
    if not any(gene['has_motor']):
        return 0.01
    return 1.0 / (1e-3 + dist_to_goal) + base_move


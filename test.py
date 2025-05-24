import pybullet as p
import pybullet_data as pd
import tempfile
import os
import time
import matplotlib.pyplot as plt

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, -9.8)


p.loadURDF("plane.urdf")

robot_id = p.loadURDF("gene_robot.urdf", basePosition=[0, 0, 1])
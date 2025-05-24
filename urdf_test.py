import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

planId = p.loadURDF("plane.urdf")

robotId = p.loadURDF("results/best_robot_20250520_233609.urdf", [0, 0, 3])

while True:
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()
import pybullet as p
import pybullet_data
import time
import numpy as np

def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Create shapes
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
    
    link_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.1)
    link_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.1, rgbaColor=[0, 1, 0, 1])
    
    # Approach 1: Try with numpy arrays
    print("\nApproach 1: Using numpy arrays")
    try:
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.3],
            linkMasses=np.array([1.0]),
            linkCollisionShapeIndices=np.array([link_col], dtype=np.int32),
            linkVisualShapeIndices=np.array([link_vis], dtype=np.int32),
            linkPositions=np.array([[0.2, 0, 0]]),
            linkOrientations=np.array([[0, 0, 0, 1]]),
            linkParentIndices=np.array([0], dtype=np.int32),
            linkJointTypes=np.array([p.JOINT_REVOLUTE], dtype=np.int32),
            linkJointAxis=np.array([[0, 0, 1]])
        )
        print(f"Success! Robot ID: {robot_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Approach 2: Try with explicit Python lists
    print("\nApproach 2: Using explicit Python lists")
    try:
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.3],
            linkMasses=[1.0],
            linkCollisionShapeIndices=[link_col],
            linkVisualShapeIndices=[link_vis],
            linkPositions=[[0.2, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1]]
        )
        print(f"Success! Robot ID: {robot_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Approach 3: Try with tuples
    print("\nApproach 3: Using tuples")
    try:
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.3],
            linkMasses=(1.0,),
            linkCollisionShapeIndices=(link_col,),
            linkVisualShapeIndices=(link_vis,),
            linkPositions=((0.2, 0, 0),),
            linkOrientations=((0, 0, 0, 1),),
            linkParentIndices=(0,),
            linkJointTypes=(p.JOINT_REVOLUTE,),
            linkJointAxis=((0, 0, 1),)
        )
        print(f"Success! Robot ID: {robot_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Approach 4: Try with a pre-built URDF
    print("\nApproach 4: Using a pre-built URDF")
    try:
        r2d2_id = p.loadURDF("r2d2.urdf", [1, 0, 0.5])
        print(f"Successfully loaded R2D2 with ID: {r2d2_id}")
        
        # Get info about the loaded robot
        num_joints = p.getNumJoints(r2d2_id)
        print(f"R2D2 has {num_joints} joints")
        
        # Print joint info
        for i in range(num_joints):
            joint_info = p.getJointInfo(r2d2_id, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Run simulation for a while
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1/240)
        
    p.disconnect()

if __name__ == "__main__":
    main() 
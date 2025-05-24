import pybullet as p
import pybullet_data
import time

def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Create a simple robot with one link
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
    
    link_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.1)
    link_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.1, rgbaColor=[0, 1, 0, 1])
    
    # Parameters for the link
    linkMasses = [1.0]
    linkCollisionShapeIndices = [link_col]
    linkVisualShapeIndices = [link_vis]
    linkPositions = [[0.2, 0, 0]]
    linkOrientations = [[0, 0, 0, 1]]
    linkParentIndices = [0]  # Link is attached to the base (index 0)
    linkJointTypes = [p.JOINT_REVOLUTE]
    linkJointAxis = [[0, 0, 1]]  # Joint rotates around z-axis
    
    # Print array lengths for debugging
    print("Array lengths:")
    print(f"  linkMasses: {len(linkMasses)}")
    print(f"  linkCollisionShapeIndices: {len(linkCollisionShapeIndices)}")
    print(f"  linkVisualShapeIndices: {len(linkVisualShapeIndices)}")
    print(f"  linkPositions: {len(linkPositions)}")
    print(f"  linkOrientations: {len(linkOrientations)}")
    print(f"  linkParentIndices: {len(linkParentIndices)}")
    print(f"  linkJointTypes: {len(linkJointTypes)}")
    print(f"  linkJointAxis: {len(linkJointAxis)}")
    
    try:
        # Create the robot
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
        
        print(f"Successfully created robot with ID: {robot_id}")
        
        # Set motor for the joint
        p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=1.0, force=50.0)
        
        # Run simulation for a while
        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1/240)
            
    except Exception as e:
        print(f"Error creating robot: {e}")
        
    finally:
        p.disconnect()

if __name__ == "__main__":
    main() 
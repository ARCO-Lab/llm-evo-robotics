import pybullet as p
import pybullet_data
import time

def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Create a simple robot with no links first
    base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
    
    try:
        # Create the robot with just a base
        robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.3]
        )
        
        print(f"Successfully created base robot with ID: {robot_id}")
        
        # Now try loading a pre-built URDF robot to see if that works
        print("Trying to load a pre-built robot...")
        r2d2_id = p.loadURDF("r2d2.urdf", [1, 0, 0.5])
        print(f"Successfully loaded R2D2 with ID: {r2d2_id}")
        
        # Run simulation for a while
        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1/240)
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        p.disconnect()

if __name__ == "__main__":
    main() 
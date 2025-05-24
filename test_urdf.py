import pybullet as p
import pybullet_data
import time
import os

def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Load our simple robot URDF
    urdf_path = os.path.join(os.getcwd(), "simple_robot.urdf")
    print(f"Loading URDF from: {urdf_path}")
    
    try:
        robot_id = p.loadURDF(urdf_path, [0, 0, 0.3])
        print(f"Successfully loaded simple robot with ID: {robot_id}")
        
        # Get info about the loaded robot
        num_joints = p.getNumJoints(robot_id)
        print(f"Simple robot has {num_joints} joints")
        
        # Print joint info
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}")
            
        # Apply motor control to the joint
        p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=1.0, force=50.0)
        
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
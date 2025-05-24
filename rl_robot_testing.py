import os
import sys
import time
import numpy as np

import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data as pd

from stable_baselines3 import PPO


from rl_controller import RobotDesignEnv, RLRobotTester


def test_environment_initialization(urdf_path):

    print("\n testing1:  Environment initialize")
    try:

        env = RobotDesignEnv(
            urdf_path = urdf_path,
            max_steps = 100,
            render_mode = "human",
            terrain_type="flat",
            reward_type="distance"
        )

        print(f"Environment create successful")
        print(f"- action space: {env.action_space}")
        print(f"- observation space: {env.observation_space}")
        print(f"- activate joints: {env.active_joints}")

        observation  = env.reset()
        print(f"- initialize observation shape: {observation.shape}")

        env.close()
        print(f"Environment initialization pass")
        return True

    except Exception as e:
        print(f"Initialize failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_step(urdf_path):

    print(f"\n testing2: Environment step")

    try:

        env = RobotDesignEnv(
            urdf_path = urdf_path,
            max_steps = 1000,
            render_mode = "human",
            terrain_type = "flat",
            reward_type = "distance"
        )

        observation = env.reset()

        print(f"exec 100 random steps")
        for i in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(f"step {i+1} done")
            print(f"- action: {action}")
            print(f"- observation: {observation}")
            print(f"- reward: {reward}")
            print(f"- info: {info}")

            if done:
                print(f"Environment finished after {i+1} steps")
                break
            time.sleep(0.1)

        env.close()

        print(f"Environment step pass")
        return True
    

    except Exception as e:
        print(f"Environment step failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
        




if __name__ == "__main__":

    if len(sys.argv) > 1:
        urdf_path = sys.argv[1]
    
    else:
        print("please provide the urdf file path")
        sys.exit(1)

    

    if not os.path.exists(urdf_path):
        print(f"Error: URDF file does not exist - {urdf_path}")
        sys.exit(1)

    print(f"Testing URDF file: {urdf_path}")

    # test_environment_step(urdf_path)
    test_environment_initialization(urdf_path)
from stable_baselines3 import PPO
import sys
import os
import numpy as np

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.mujoco_env import create_mujoco_env

def simulate_and_evaluate(robot_xml, num_individuals=10, render_mode=False):
    """运行 MuJoCo 并评估适应度"""

    # ✅ 创建 MuJoCo 共享环境（所有机器人在同一物理世界中）
    env = create_mujoco_env(robot_xml, num_individuals, render_mode)

    # ✅ 初始化 PPO
    model = PPO("MlpPolicy", env, verbose=1)

    #TODO: 查看 model.learn 函数， 计算每个individual奖励平均值当作fitness 传出
    # ✅ 训练 PPO 让多个机器人学习
    model.learn(total_timesteps=50000)

    # ✅ 评估 PPO 训练后的策略
    obs, _ = env.reset()
    total_rewards = np.zeros(num_individuals)  # 记录所有机器人的总奖励

    for _ in range(1000):
        # ✅ PPO 需要 `obs` 是 `numpy array`，确保转换
        if isinstance(obs, list):
            obs = np.array(obs)

        # ✅ PPO 生成的 `actions`，自动适配 `num_individuals`
        actions, _states = model.predict(obs, deterministic=True)

        # ✅ 让 `PPO` 控制所有机器人
        obs, rewards, done, truncated, _ = env.step(actions)

        # # ✅ 累积所有机器人的奖励
        # total_rewards += np.array(rewards)

        # # ✅ 只要有任意机器人 `done`，就停止
        # if np.any(done) or np.any(truncated):
        #     break

    env.close()
    return [-sum(total_rewards), len(robot_xml)]

if __name__ == "__main__":
    print("✅ 测试 MuJoCo PPO 训练...")

    # ✅ 适配多个机器人共享环境
    dummy_xml = """<mujoco model="multi_robot">
    <compiler angle="radian" />
    <option timestep="0.002" />

    <asset>
        <texture name="checker" type="2d" builtin="checker" width="512" height="512" />
        <material name="gray" texture="checker" rgba="0.7 0.7 0.7 1" />
    </asset>

    <worldbody>
        <geom name="ground" type="plane" size="10 10 0.1" rgba="0.2 0.6 0.2 1"/>
        <light name="main_light" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>

        <!-- 第一个机器人 -->
        <body name="base1" pos="0 0 0.58">
            <freejoint />
            <geom type="box" size="0.2 0.2 0.1" rgba="0.8 0.8 0.8 1"/>

            <body name="LF_HIP" pos="0.277 0.116 0">
                <joint name="LF_HAA" type="hinge" axis="1 0 0" pos="0.277 0.116 0" range="-0.5 0.5"/>
                <geom type="cylinder" size="0.05 0.1" rgba="1 0 0 1"/>
            </body>

            <body name="RF_HIP" pos="0.277 -0.116 0">
                <joint name="RF_HAA" type="hinge" axis="1 0 0" pos="0.277 -0.116 0" range="-0.5 0.5"/>
                <geom type="cylinder" size="0.05 0.1" rgba="0 0 1 1"/>
            </body>
        </body>

        <!-- 第二个机器人 -->
        <body name="base2" pos="1.0 0 0.58"> <!-- 位移到 (1.0, 0, 0.58) 以避免碰撞 -->
            <freejoint />
            <geom type="box" size="0.2 0.2 0.1" rgba="0.7 0.7 0.7 1"/>

            <body name="LB_HIP" pos="0.277 0.116 0">
                <joint name="LB_HAA" type="hinge" axis="1 0 0" pos="0.277 0.116 0" range="-0.5 0.5"/>
                <geom type="cylinder" size="0.05 0.1" rgba="0 1 0 1"/>
            </body>

            <body name="RB_HIP" pos="0.277 -0.116 0">
                <joint name="RB_HAA" type="hinge" axis="1 0 0" pos="0.277 -0.116 0" range="-0.5 0.5"/>
                <geom type="cylinder" size="0.05 0.1" rgba="1 1 0 1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <!-- 第一个机器人的执行器 -->
        <motor name="LF_HAA_motor" joint="LF_HAA" ctrlrange="-1 1" gear="10"/>
        <motor name="RF_HAA_motor" joint="RF_HAA" ctrlrange="-1 1" gear="10"/>

        <!-- 第二个机器人的执行器 -->
        <motor name="LB_HAA_motor" joint="LB_HAA" ctrlrange="-1 1" gear="10"/>
        <motor name="RB_HAA_motor" joint="RB_HAA" ctrlrange="-1 1" gear="10"/>
    </actuator>
</mujoco>

    """
    
    results = simulate_and_evaluate(dummy_xml, num_individuals=2, render_mode=True)

    print("✅ 测试完成，适应度结果:", results)

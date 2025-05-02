import mujoco
import stable_baselines3 as sb3
import numpy as np
import tempfile
from gymnasium import Env
from gymnasium.spaces import Box

class MultiRobotMuJoCoEnv(Env):
    """一个 MuJoCo 环境，支持多个个体同时运行"""
    def __init__(self, xml_path, num_individuals=10, render_mode=False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.sim = mujoco.MjSim(self.model)
        self.viewer = None if not render_mode else mujoco.MjViewer(self.sim)
        self.num_individuals = num_individuals
        self.render_mode = render_mode

        # 观察空间 & 动作空间
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_individuals, 10), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(num_individuals, 5), dtype=np.float32)

    def step(self, actions):
        rewards = []
        observations = []
        for i in range(self.num_individuals):
            mujoco.mj_step(self.model, self.data)
            obs = np.random.rand(10)
            reward = -np.linalg.norm(actions[i])
            rewards.append(reward)
            observations.append(obs)
        if self.render_mode and self.viewer:
            self.viewer.render()  # 🔥 每一步可视化
        done = False
        return np.array(observations), np.array(rewards), done, {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return np.random.rand(self.num_individuals, 10)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.MjViewer(self.sim)
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None  # 释放 Viewer 资源

def simulate_and_evaluate(robot_xml, num_individuals, render_mode=False):
    """并行运行 MuJoCo 仿真，评估多个个体"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_xml:
        temp_xml.write(robot_xml.encode("utf-8"))
        temp_xml_path = temp_xml.name

    # 创建共享 MuJoCo 环境
    env = sb3.common.env_util.make_vec_env(
        lambda: MultiRobotMuJoCoEnv(temp_xml_path, num_individuals, render_mode), n_envs=1
    )
    model = sb3.PPO("MlpPolicy", env, verbose=0)
    
    # 训练 PPO
    model.learn(total_timesteps=5000)

    # 评估 fitness
    obs = env.reset()
    total_rewards = np.zeros(num_individuals)
    for _ in range(1000):  # 让所有个体运行 1000 步
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _ = env.step(actions)
        total_rewards += rewards
        if done.any():
            break

    fitness_results = [[-reward, len(robot_xml)] for reward in total_rewards]

    env.close()  # 训练完成后，释放环境
    return fitness_results

if __name__ == "__main__":
    print("✅ 运行 MuJoCo 测试...")
    dummy_xml = "<mujoco><worldbody></worldbody></mujoco>"
    results = simulate_and_evaluate(dummy_xml, num_individuals=3, render_mode=True)
    print("测试完成，返回适应度:", results)
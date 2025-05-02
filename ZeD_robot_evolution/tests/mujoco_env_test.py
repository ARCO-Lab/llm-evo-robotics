import time
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import tempfile

class MuJoCoEnv(Env):
    """一个 MuJoCo 训练环境，支持 Viewer 可视化"""

    def __init__(self, xml_path, render_mode=True):
        """初始化 MuJoCo 环境"""
        self.xml_path = xml_path
        self.render_mode = render_mode

        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 观察空间 & 动作空间
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)

        # 初始化 Viewer（使用 launch_passive）
        self.viewer = None
        if render_mode:
            self.init_viewer()

    def init_viewer(self):
        """初始化 MuJoCo Viewer"""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def step(self, action):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
        obs = np.random.rand(10)
        reward = -np.linalg.norm(action)
        done = False
        truncated = False
        info = {}

        if self.render_mode:
            self.render()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        obs = np.random.rand(10)
        info = {}
        return obs, info

    def render(self):
        """渲染 MuJoCo 场景"""
        if self.viewer:
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
            self.viewer.sync()  # ✅ 让 Viewer 同步仿真状态

    def close(self):
        """关闭 MuJoCo 环境"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def create_mujoco_env(robot_xml, render_mode=True):
    """创建 MuJoCo 环境"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_xml:
        temp_xml.write(robot_xml.encode("utf-8"))
        temp_xml_path = temp_xml.name

    return MuJoCoEnv(temp_xml_path, render_mode)

if __name__ == "__main__":
    print("✅ 启动 MuJoCo 环境...")

    # 定义一个 XML 机器人
    dummy_xml = """<mujoco>
        <worldbody>
            <body name="ball" pos="0 0 0.1">
                <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
            </body>
            <camera name="fixed" pos="0 -2 1" xyaxes="1 0 0 0 1 0"/>
        </worldbody>
    </mujoco>"""

    env = create_mujoco_env(dummy_xml, render_mode=True)

    obs, _ = env.reset()
    print("初始观察值:", obs)

    # 运行 30 秒
    start_time = time.time()
    while time.time() - start_time < 30:
        action = np.random.uniform(-1, 1, (5,))
        obs, reward, done, truncated, _ = env.step(action)
        print(f"观察值: {obs}, 奖励: {reward}")

    env.close()
    print("✅ 运行完成！")

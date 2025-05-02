import mujoco
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import tempfile
import sys
import os
from time import sleep
from add_wall_goal import add_walls_and_goals
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xml.etree.ElementTree as ET
from utils.create_temp_xml import create_temp_xml

from utils.robot_actuator_counter import parse_mujoco_xml_string  # ✅ 引入解析器

import random

class MultiRobotMuJoCoEnv(Env):
    """一个 MuJoCo 训练环境，支持多个机器人同时运行"""
    def __init__(self, xml_path, num_individuals=10, render_mode=False, environment_map=None):
        """
        初始化环境：
        - xml_path: MuJoCo XML 文件路径
        - num_individuals: 并行运行的机器人数量
        - render_mode: 是否可视化
        """
        self.xml_path = xml_path
        self.num_individuals = num_individuals
        
        self.render_mode = render_mode
        self.max_actuator_num = 10
        self.environment_map = environment_map
        with open(xml_path, "r") as f:
            self.xml_string = f.read()
       
        self.robot_stats = {
            i: {
                "name": self.environment_map["robots"][i]["name"],
                "prev_distance": float("inf"),  # 初始距离设为无穷大
                "total_reward": 0.0,
                "effective_steps": 0,
                "reached_goal": False,  # 记录是否到达目标
            }
            for i in range(self.num_individuals)
        }
        self.dones = [False] * self.num_individuals  # 记录每个机器人的终止状态“
        self.threshold = 0.2  # 目标点判定阈值

        # self._update_env_with_new_goals(self.xml_string)
        self.robot_actuator_counts = parse_mujoco_xml_string(self.xml_string)
        self.robot_actuator_indices = self.map_actuators_to_ctrl_indices()
        print(f"📢 机器人 actuator 索引范围: {self.robot_actuator_indices}")
        # ✅ 添加随机目标点
        self.xml_string = self.add_random_goal_to_grounds(self.xml_string)
        self.xml_path = create_temp_xml(self.xml_string)
        # ✅ 加载 MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv + self.max_actuator_num
        # ✅ 观察空间 & 动作空间
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_individuals, self.obs_dim), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(num_individuals, self.max_actuator_num), dtype=np.float32)


        # ✅ 初始化 Viewer
        self.viewer = None
        if render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # def _update_env_with_new_goals(self, xml_string):
    #     """每次 `reset()` 重新生成随机目标点，并更新 XML"""
    #     modified_xml_string = add_walls_and_goals(xml_string)

    #     # ✅ 保存新的 XML 到临时文件
    #     temp_xml = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    #     temp_xml.write(modified_xml_string.encode("utf-8"))
    #     temp_xml.close()
    #     self.xml_path = temp_xml.name

    #     # # ✅ 解析机器人结构，更新控制索引
    #     # self.robot_actuator_counts = parse_mujoco_xml_string(modified_xml_string)
    #     # self.robot_actuator_indices = self.map_actuators_to_ctrl_indices()

    def add_random_goal_to_grounds(self, env_xml):
        """
        在每个 ground 上生成一个随机目标点 (红球)，确保：
        1. 目标点在 ground 内（不超出墙体）。
        2. 目标点距离机器人有一定距离（半径 = 1/2 * ground 的最短边）。

        :param env_xml: MuJoCo 环境 XML（字符串）
        :return: 更新后的 XML 字符串
        """
        # ✅ 解析环境 XML
        env_root = ET.fromstring(env_xml)
        worldbody = env_root.find("worldbody")

        # ✅ 找到所有 ground
        ground_geoms = [geom for geom in worldbody.findall("geom") if "ground" in geom.get("name", "")]
        
        for i, ground_geom in enumerate(ground_geoms):
            ground_pos = list(map(float, ground_geom.get("pos", "0 0 0").split()))
            ground_size = list(map(float, ground_geom.get("size", "2 2 0.1").split()))
            ground_x, ground_y, ground_z = ground_pos[0], ground_pos[1], ground_pos[2]
            size_x, size_y = ground_size[0], ground_size[1]
            # ✅ 计算安全半径（1/2 最短边）
            safe_radius = 0.5 * min(size_x, size_y)

            # ✅ 从 `environment_map` 获取分配到该 ground 的机器人
            assigned_robot = None
            assigned_robot_id = None
            for robot_id, robot_info in self.environment_map["robots"].items():
                if robot_info["ground_id"] == i:  # 机器人属于这个 ground
                    assigned_robot = robot_info
                    assigned_robot_id = robot_id
                    break

            if assigned_robot is None:
                raise ValueError(f"❌ 未找到分配到 {ground_geom.get('name')} 的机器人！")

            robot_x, robot_y, _ = assigned_robot["pos"]

            # ✅ 随机生成目标点，确保它在 `safe_radius` 外，且在 `ground` 内
            while True:
                goal_x = random.uniform(ground_x - size_x + 0.2, ground_x + size_x - 0.2)
                goal_y = random.uniform(ground_y - size_y + 0.2, ground_y + size_y - 0.2)

                # 确保目标点不在 `safe_radius` 内
                if ((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2) ** 0.5 > safe_radius:
                    break
            goal_z = ground_z + 0.1
            # ✅ 添加目标点（红色球体）
            goal_body = ET.Element("body", name=f"goal_{i}", pos=f"{goal_x} {goal_y} {goal_z}")
            goal_geom = ET.Element("geom", type="sphere", size="0.1", rgba="1 0 0 1")  # 红色球
            goal_body.append(goal_geom)
            worldbody.append(goal_body)
            # ✅ 更新 `environment_map`，改用 `assigned_robot_id`
            self.environment_map["robots"][assigned_robot_id]["goal"] = (goal_x, goal_y, goal_z)
            print(f"goal_x: {goal_x}, goal_y: {goal_y}, goal_z: {goal_z}")

        return ET.tostring(env_root, encoding="unicode")


 
    # def check_collision(self, robot_index):
    #     """
    #     检查指定的机器人是否发生了撞墙碰撞
    #     - 通过 MuJoCo `contact` 数据检测是否与墙体接触
    #     - 仅当机器人与 `wall_*` 发生碰撞时，返回 True（忽略 ground_*）
    #     """
    #     print(f'robot info : {self.environment_map["robots"]}')
    #     robot_name = self.environment_map["robots"][robot_index]["name"]
    #     print(f"contact: {self.data.contact}")
    #     for contact in self.data.contact:
    #         print(f"contact ---: {contact}")
    #         geom1 = self.model.geom_id2name(contact.geom1)
    #         geom2 = self.model.geom_id2name(contact.geom2)
    #         print(f"geom1: {geom1}, geom2: {geom2}")

    #         if geom1 and geom2:
    #             # ✅ 确保当前碰撞是和这个 `robot` 相关
    #             if robot_name in geom1 or robot_name in geom2:
    #                 # ✅ 排除地面碰撞，只考虑撞墙
    #                 if ("wall" in geom1 or "wall" in geom2) and not ("ground" in geom1 or "ground" in geom2):
    #                     return True  # 机器人撞到了墙

    #     return False  # 没有撞墙

    def compute_reward(self):
        """
        计算每个机器人的 Reward，基于：
        1. 机器人当前坐标和目标点坐标的距离变化
        2. 机器人是否到达目标点（返回 done）
        3. 运动平滑性（惩罚大动作）
        
        :return: rewards (numpy array), done (list of bool)
        """
        rewards = []
     
     

        for i in range(self.num_individuals):
            env_info = self.environment_map["robots"][i]
            robot_info = self.robot_stats[i]

            if self.dones[i]:
                reward = 0.0
            else:
                goal_pos = env_info["goal"]  # (x, y, z)
              
                start_idx = i * 9   # 每个自由机器人 `qpos` 里占用 7 维
                end_idx = start_idx + 3  # 只取 `(x, y, z)`
                robot_pos = self.data.qpos[start_idx:end_idx]
              
                d_t = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos[:3]))
                d_t_1 = robot_info.get("prev_distance")

                #TODO: 验证奖励项的权重

                # ✅ 计算奖励项
                r1 = np.clip((d_t_1 - d_t) / (d_t_1 + 1e-6), -1, 1)  # 距离缩短奖励
                r2 = 10.0 if d_t < self.threshold else 0.0  # 终点奖励
                start_idx, end_idx = self.robot_actuator_indices[env_info["name"]]
                r3 = np.sum(self.data.ctrl[start_idx:end_idx] ** 2)
                if i == 0:
                    print(f"r1: {r1}, r2: {r2}, r3: {r3}")
                # ✅ 记录奖励
                reward = r1 + r2 + r3
                
                robot_info["total_reward"] += reward
                robot_info["effective_steps"] += 1


                if d_t < self.threshold:
                    self.dones[i] = True
                    self.robot_stats[i]["reached_goal"] = True
                
            # ✅ 更新 `prev_distance`
                self.robot_stats[i]["prev_distance"] = d_t

            rewards.append(reward)

            # ✅ 机器人到达目标点，标记 `done`

        return np.array(rewards)



    def map_actuators_to_ctrl_indices(self):
        """
        计算每个机器人 actuator 在 `self.data.ctrl` 里的索引范围
        """
        indices = {}
        start_idx = 0  # `self.data.ctrl` 的索引起点

        for robot_name, num_actuators in self.robot_actuator_counts.items():
            indices[robot_name] = (start_idx, start_idx + num_actuators)  # 存储 (start, end) 范围
            start_idx += num_actuators  # 更新起点索引

        return indices
    
    def get_observation(self):
        """
        计算 MuJoCo 观察值（obs），确保 `obs_dim` 固定
        """
        # qpos_dim = self.model.nq // self.num_individuals  # 单个机器人 `qpos` 维度
        # qvel_dim = self.model.nv // self.num_individuals  # 单个机器人 `qvel` 维度
        # max_qpos = qpos_dim * self.num_individuals  # 总 `qpos`
        # max_qvel = qvel_dim * self.num_individuals  # 总 `qvel`

        max_qpos = self.model.nq
        max_qvel = self.model.nv
        max_ctrl = self.max_actuator_num  # 最大 `actuator` 维度

        # ✅ 初始化固定大小的 `obs`
        qpos = np.zeros(max_qpos)
        qvel = np.zeros(max_qvel)
        ctrl = np.zeros(max_ctrl)

        # ✅ 复制 MuJoCo 当前状态
        qpos[:len(self.data.qpos)] = self.data.qpos[:]
        qvel[:len(self.data.qvel)] = self.data.qvel[:]
        ctrl[:len(self.data.ctrl)] = self.data.ctrl[:]

        # ✅ 组合最终 `obs`
        obs = np.concatenate([qpos, qvel, ctrl])
        return obs


    def step(self, actions):
        """执行一步仿真"""
        rewards = []
        observations = []

        for i, (robot_name, (start_idx, end_idx)) in enumerate(self.robot_actuator_indices.items()):
            num_actuators = end_idx - start_idx  # 该机器人实际 actuator 数量
            valid_actions = actions[i][:num_actuators]
            self.data.ctrl[start_idx:end_idx] = valid_actions
            # print(f"qpos dim: {self.model.nq}")
            # print(f"🤖 {robot_name} 控制信号: {valid_actions}")
        mujoco.mj_step(self.model, self.data)

         # ✅ 计算 Reward
        rewards = self.compute_reward()
        # print(f"rewards: {rewards}")

        for i in range(self.num_individuals):
            # print(f"I: {i}")
            # self.check_collision(i)
            # print(self.model.nu)
            # self.data.ctrl[:] = actions.flatten() 
            # if self.data.qpos.shape[0] >= (i + 1) * 3:  # ✅ 确保 qpos 足够长
            #     self.data.qpos[i * 3: (i + 1) * 3] += actions[i] * 0.01
            
            obs = self.get_observation()  # 生成随机观察值
            # reward = -np.linalg.norm(actions[i])  # 适应度
            # rewards.append(reward)
            observations.append(obs)

            
        # truncated = len(rewards) >= 1000  # 最大步长 1000 终止
        truncated = False

        if self.render_mode and self.viewer:
            self.render()

        return np.array(observations), rewards, self.dones, truncated, {}

    def reset(self, seed=None, options=None):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
    # ✅ 让机器人之间不碰撞，但与地面碰撞
        # for geom_id in range(self.model.ngeom):
        #     geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        #     print(f"geom_name: {geom_name}")
        #     sleep(2)
        #     if geom_name and "base" in geom_name:  # 只处理机器人 geom
        #         self.model.geom_contype[geom_id] = 0  # 机器人不触发碰撞
        #         self.model.geom_conaffinity[geom_id] = 0  # 机器人不参与碰撞
        #     elif geom_name and "ground" in geom_name:  # 允许地面发生碰撞
        #         self.model.geom_contype[geom_id] = 1
        #         self.model.geom_conaffinity[geom_id] = 1
        obs = self.get_observation()
        return obs, {}

    def render(self):
        """渲染 MuJoCo 场景"""
        if self.viewer:
            with self.viewer.lock():
                self.viewer.sync()  # ✅ 让 Viewer 同步仿真状态

    def close(self):
        """关闭 MuJoCo 环境"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # def create_temp_xml(self, robot_xml):

    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_xml:
    #         temp_xml.write(robot_xml.encode("utf-8"))
    #         temp_xml_path = temp_xml.name

    #     return temp_xml_path

def create_mujoco_env(robot_xml, num_individuals=10, render_mode=False, environment_map=None):
    """
    创建 MuJoCo 环境并返回 gym-style 的 `Env` 对象
    """
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_xml:
    #     temp_xml.write(robot_xml.encode("utf-8"))
    #     temp_xml_path = temp_xml.name

    temp_xml_path = create_temp_xml(robot_xml)
    return MultiRobotMuJoCoEnv(temp_xml_path, num_individuals, render_mode, environment_map)


def insert_robots_into_environment(env_xml, robot_xmls):
        """
        将机器人 XML 插入到环境 XML 的不同 ground 区域
        :param env_xml:  环境的 MuJoCo XML 字符串
        :param robot_xmls: 机器人 XML 列表，每个机器人是一个字符串
        :return: 更新后的完整 XML 字符串
        """
        # ✅ 解析环境 XML
        env_root = ET.fromstring(env_xml)
        worldbody = env_root.find("worldbody")

        # ✅ 如果环境中没有 actuator，创建一个
        actuator_root = env_root.find("actuator")
        if actuator_root is None:
            actuator_root = ET.Element("actuator")
            env_root.append(actuator_root)

        # ✅ 找到所有 ground 区域
        ground_geoms = [geom for geom in worldbody.findall("geom") if "ground" in geom.get("name", "")]
        num_grounds = len(ground_geoms)

        if num_grounds == 0:
            raise ValueError("❌ 未找到 ground 区域，无法分配机器人！")

        environment_map = {
               "grounds": {},
                "robots": {},
        }

        for i, ground_geom in enumerate(ground_geoms):
            ground_name = ground_geom.get("name", f"ground_{i}")
            ground_pos = list(map(float, ground_geom.get("pos", "0 0 0").split()))
            ground_size = list(map(float, ground_geom.get("size", "2 5 0.1").split()))

            environment_map["grounds"][i] = {
                "name": ground_name,  # ✅ 存入 ground 名称
                "pos": tuple(ground_pos),
                "size": tuple(ground_size),
            }

        # ✅ 遍历机器人 XML 并插入到不同的 ground 区域
        for i, robot_xml in enumerate(robot_xmls):
            ground_index = i % num_grounds  # 轮流分配机器人
            ground_geom = ground_geoms[ground_index]

            # ✅ 获取 ground 位置
            ground_pos = ground_geom.get("pos", "0 0 0").split()
            x_pos, y_pos, z_pos = map(float, ground_pos)

            # ✅ 解析机器人 XML 并调整位置
            robot_root = ET.fromstring(robot_xml)
            robot_name = robot_root.get("name", f"robot_{i}")  # 机器人名称
            if robot_root is not None:
                robot_root.set("pos", f"{x_pos} {y_pos} {z_pos + 0.5}")  # 确保机器人在地面之上

            robot_actuator = robot_root.find("actuator")
            if robot_actuator is not None:
                for motor in robot_actuator.findall("motor"):
                    actuator_root.append(motor)
                    
            robot_root.remove(robot_actuator)
           

            environment_map["robots"][i] = {
                "name": robot_name,
                "ground_id": ground_index,
                "pos": (x_pos, y_pos, z_pos + 0.5),
                "body": robot_root,  # 存储 `ET.Element`
                "actuator": robot_actuator,  # 存储 `ET.Element`
                "goal": None,  # 目标点，等 reset 之后赋值
            }
            # ✅ 将机器人插入 worldbody
            worldbody.append(robot_root)
        
        # tree = ET.ElementTree(env_root)
        # with open("env_root.json", "wb") as f:
        #     tree.write(f, encoding="utf-8", xml_declaration=True)
       
        # sleep(2)
        # print(f"environment_map: {environment_map}")
        # ✅ 生成最终 XML 字符串
        return ET.tostring(env_root, encoding="unicode"), environment_map

if __name__ == "__main__":
    print("✅ 测试 MuJoCo 环境封装...")

    # ✅ XML 结构：多个机器人
#     dummy_xml = """<mujoco model="multi_robot">
#     <compiler angle="radian" />
#     <option timestep="0.002" />

#     <asset>
#         <texture name="checker" type="2d" builtin="checker" width="512" height="512" />
#         <material name="gray" texture="checker" rgba="0.7 0.7 0.7 1" />
#     </asset>

#     <worldbody>
#         <geom name="ground" type="plane" size="10 10 0.1" rgba="0.2 0.6 0.2 1"/>
#         <light name="main_light" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>

#         <!-- 第一个机器人 -->
#         <body name="base1" pos="0 0 0.58">
#             <freejoint />
#             <geom type="box" size="0.2 0.2 0.1" rgba="0.8 0.8 0.8 1"/>

#             <body name="LF_HIP" pos="0.277 0.116 0">
#                 <joint name="LF_HAA" type="hinge" axis="1 0 0" pos="0.277 0.116 0" range="-0.5 0.5"/>
#                 <geom type="cylinder" size="0.05 0.1" rgba="1 0 0 1"/>
#             </body>

#             <body name="RF_HIP" pos="0.277 -0.116 0">
#                 <joint name="RF_HAA" type="hinge" axis="1 0 0" pos="0.277 -0.116 0" range="-0.5 0.5"/>
#                 <geom type="cylinder" size="0.05 0.1" rgba="0 0 1 1"/>
#             </body>
#         </body>

#         <!-- 第二个机器人 -->
#         <body name="base2" pos="0 0 0.58"> <!-- 位移到 (1.0, 0, 0.58) 以避免碰撞 -->
#             <freejoint />
#             <geom type="box" size="0.2 0.2 0.1" rgba="0.7 0.7 0.7 1"/>

#             <body name="LB_HIP" pos="0.277 0.116 0">
#                 <joint name="LB_HAA" type="hinge" axis="1 0 0" pos="0.277 0.116 0" range="-0.5 0.5"/>
#                 <geom type="cylinder" size="0.05 0.1" rgba="0 1 0 1"/>
#             </body>

#             <body name="RB_HIP" pos="0.277 -0.116 0">
#                 <joint name="RB_HAA" type="hinge" axis="1 0 0" pos="0.277 -0.116 0" range="-0.5 0.5"/>
#                 <geom type="cylinder" size="0.05 0.1" rgba="1 1 0 1"/>
#             </body>
#         </body>
#     </worldbody>

#     <actuator>
#         <!-- 第一个机器人的执行器 -->
#         <motor name="LF_HAA_motor" joint="LF_HAA" ctrlrange="-1 1" gear="10"/>
#         <motor name="RF_HAA_motor" joint="RF_HAA" ctrlrange="-1 1" gear="10"/>

#         <!-- 第二个机器人的执行器 -->
#         <motor name="LB_HAA_motor" joint="LB_HAA" ctrlrange="-1 1" gear="10"/>
#         <motor name="RB_HAA_motor" joint="RB_HAA" ctrlrange="-1 1" gear="10"/>
#     </actuator>
# </mujoco>
#     """

    dummy_env_xml = """<mujoco model="multi_robot_env">
    <compiler angle="radian" />
    <option timestep="0.002" />

    <asset>
        <texture name="checker" type="2d" builtin="checker" width="512" height="512" />
        <material name="gray" texture="checker" rgba="0.7 0.7 0.7 1" />
    </asset>

    <worldbody>
        <light name="main_light" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>

        <!-- 机器人区域 1 -->
        <geom name="ground_1" type="plane" size="2 5 0.1" pos="-3 0 0" rgba="0.2 0.6 0.2 1"/>
        <geom name="wall_left_1" type="box" pos="-5 0 0.1" size="0.1 5 0.2" rgba="0.3 0.3 0.3 1"/>
        <geom name="wall_right_1" type="box" pos="-1 0 0.1" size="0.1 5 0.2" rgba="0.3 0.3 0.3 1"/>
        <geom name="wall_front_1" type="box" pos="-3 -5 0.1" size="2 0.1 0.2" rgba="0.3 0.3 0.3 1"/>
        <geom name="wall_back_1" type="box" pos="-3 5 0.1" size="2 0.1 0.2" rgba="0.3 0.3 0.3 1"/>

        <!-- 机器人区域 2 -->
        <geom name="ground_2" type="plane" size="2 5 0.1" pos="3 0 0" rgba="0.2 0.6 0.2 1"/>
        <geom name="wall_left_2" type="box" pos="1 0 0.1" size="0.1 5 0.2" rgba="0.3 0.3 0.3 1"/>
        <geom name="wall_right_2" type="box" pos="5 0 0.1" size="0.1 5 0.2" rgba="0.3 0.3 0.3 1"/>
        <geom name="wall_front_2" type="box" pos="3 -5 0.1" size="2 0.1 0.2" rgba="0.3 0.3 0.3 1"/>
        <geom name="wall_back_2" type="box" pos="3 5 0.1" size="2 0.1 0.2" rgba="0.3 0.3 0.3 1"/>

        <!-- 未来可扩展区域 -->
        <!-- 这里可以继续添加更多区域 -->

    </worldbody>
</mujoco>"""

    dummy_robot_xmls = [
    """<body name="base1" pos="0 0 0.58">
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

        <actuator>
            <motor name="LF_HAA_motor" joint="LF_HAA" ctrlrange="-1 1" gear="10"/>
            <motor name="RF_HAA_motor" joint="RF_HAA" ctrlrange="-1 1" gear="10"/>
        </actuator>
    </body>""",

    """<body name="base2" pos="0 0 0.58">
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

        <actuator>
            <motor name="LB_HAA_motor" joint="LB_HAA" ctrlrange="-1 1" gear="10"/>
            <motor name="RB_HAA_motor" joint="RB_HAA" ctrlrange="-1 1" gear="10"/>
        </actuator>
    </body>"""
]

    dummy_xml, environment_map = insert_robots_into_environment(dummy_env_xml, dummy_robot_xmls)

    env = create_mujoco_env(dummy_xml, num_individuals=2, render_mode=True, environment_map=environment_map)

    obs, _ = env.reset()
    print("初始观察值:", obs)

    for i in range(1000000):
        actions = np.random.uniform(-1, 1, (2, 10))  # 两个机器人
        next_obs, rewards, done, truncated, _ = env.step(actions)
        # print(f"Step {i}: 奖励 {rewards}")
        if all(done) or truncated:
            # print(f"test done")
            break
    env.close()
    print("✅ 测试完成！")

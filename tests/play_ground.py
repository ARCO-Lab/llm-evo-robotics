import mujoco
import mujoco.viewer
import numpy as np

# 定义仿真模型文件路径
MODEL_PATH = "scene.xml"  # 请确保有一个两腿机器人模型 XML 文件

# 加载MuJoCo模型
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 定义仿真时间步长
time_step = 0.01  # 10ms

# 定义步态周期和控制参数
cycle_time = 1.0  # 一个完整步态周期的时间（单位：秒）
half_cycle = cycle_time / 2  # 半周期时间
amplitude = np.array([0.5, 0.3, 0.2])  # 每个关节的动作幅度
frequency = 2 * np.pi / cycle_time  # 正弦波频率

# 定义功能调整参数
W = np.array([0.8, 0.8, 0.8])  # 缩放矩阵
b = np.array([0.1, 0.05, 0.0])  # 偏移量

# 定义群操作

def identity_action(left_action):
    """
    单位操作，保持左腿动作不变
    """
    return left_action

def time_shift_action(left_action, time_elapsed):
    """
    时间偏移操作，模拟右腿相对于左腿滞后半个周期
    :param left_action: 左腿当前的动作
    :param time_elapsed: 当前仿真时间
    """
    return amplitude * np.sin(frequency * (time_elapsed + half_cycle))

def functional_adjustment(left_action):
    """
    功能调整操作，对左腿动作进行缩放和偏移
    :param left_action: 左腿当前的动作
    """
    return W * left_action + b

def combined_action(left_action, time_elapsed):
    """
    复合操作：时间偏移 + 功能调整
    :param left_action: 左腿当前的动作
    :param time_elapsed: 当前仿真时间
    """
    time_shifted = time_shift_action(left_action, time_elapsed)
    return functional_adjustment(time_shifted)

# 初始化仿真器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 仿真循环
    time_elapsed = 0.0
    while viewer.is_running():
        # 当前时间更新
        time_elapsed += time_step

        # 左腿的动作（正弦函数模拟）
        left_action = amplitude * np.sin(frequency * time_elapsed)

        # 使用群操作生成右腿动作
        right_action_identity = identity_action(left_action)  # 单位元
        right_action_time_shifted = time_shift_action(left_action, time_elapsed)  # 时间偏移
        right_action_functional = functional_adjustment(left_action)  # 功能调整
        right_action_combined = combined_action(left_action, time_elapsed)  # 复合操作

        # 选择一个右腿动作（这里使用复合操作）
        right_action = right_action_combined

        # 应用控制到模型
        # 假设左腿关节是[0, 1, 2]，右腿关节是[3, 4, 5]
        data.ctrl[0:3] = left_action  # 左腿控制
        data.ctrl[3:6] = right_action  # 右腿控制

        # 仿真步进
        mujoco.mj_step(model, data)

        # 更新渲染
        viewer.sync()

print("Simulation ended.")

import numpy as np
from simulation.mujoco_simulation import simulate_and_evaluate

def evaluate_population(individuals, components, actuators, generation, render_interval=10):
    """
    计算所有个体的适应度，返回 `NSGA-II` 需要的 `F` 值。
    
    参数:
    - individuals: 二进制基因编码列表
    - components: 可用的组件
    - actuators: 可用的驱动器
    - generation: 当前进化代数
    - render_interval: 每 `N` 代可视化一次

    返回:
    - 适应度列表
    """
    from robot.robot_generator import generate_robot_xml  # 只有在运行时才导入，避免循环导入

    # 生成机器人 XML
    robot_xml = generate_robot_xml(individuals, components, actuators)

    # 是否可视化 MuJoCo（每 `render_interval` 代）
    render_mode = (generation % render_interval == 0)

    # 运行 MuJoCo 评估适应度
    fitness_results = simulate_and_evaluate(robot_xml, num_individuals=len(individuals), render_mode=render_mode)

    # 适应度: -reward (因为 NSGA-II 需要最小化)
    rewards = -np.array(fitness_results)[:, 0]

    return rewards, fitness_results




### **📌 添加测试入口**
if __name__ == "__main__":
    print("✅ 测试适应度计算...")

    from robot.xml_parser import load_components, load_actuators

    # **加载 XML 组件 & Actuators**
    components = load_components("configs/basic_components.xml")
    actuators = load_actuators("configs/basic_actuators.xml")

    # **创建随机个体**
    num_individuals = 5  # 测试 5 个个体
    num_genes = len(components) + len(actuators)
    dummy_individuals = np.random.randint(0, 2, (num_individuals, num_genes))

    # **计算适应度**
    rewards, fitness_results = evaluate_population(dummy_individuals, components, actuators, generation=1, render_interval=10)

    print("🎯 适应度计算完成，结果如下：")
    for i, (reward, details) in enumerate(zip(rewards, fitness_results)):
        print(f"  🔹 个体 {i}: 适应度 = {reward}, 详情 = {details}")

    print("✅ 适应度测试完成！")
from xml_parser import load_components, load_actuators
from nsga2_optimization import run_nsga2

# **只解析 XML 一次**
components = load_components("basic_components.xml")
actuators = load_actuators("basic_actuators.xml")

# 计算基因变量的总数（组件 + actuator）
num_variables = len(components) + len(actuators)

# 运行 NSGA-II（传递解析后的 components 和 actuators）
best_design = run_nsga2(num_variables, components, actuators)
print("最优机器人基因编码:", best_design)

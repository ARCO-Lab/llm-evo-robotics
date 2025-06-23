import numpy as np
import pyrobotdesign as rd
import tasks
import os
import sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)

def print_graph_structure(graph):
    """打印graph的完整结构"""
    print(f"=== Graph: {graph.name} ===")
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    
    # 打印节点信息
    print("\n--- Nodes ---")
    for i, node in enumerate(graph.nodes):
        attrs = node.attrs
        print(f"  {i}: {node.name}")
        print(f"    Label: {attrs.label}")
        print(f"    Shape: {attrs.shape}")
        print(f"    Length: {attrs.length:.3f}, Radius: {attrs.radius:.3f}")
        print(f"    Base: {attrs.base}")
        print(f"    Color: {attrs.color}")
    
    # 打印边信息
    print("\n--- Edges ---")
    for i, edge in enumerate(graph.edges):
        head = graph.nodes[edge.head].name
        tail = graph.nodes[edge.tail].name
        attrs = edge.attrs
        print(f"  {i}: {head} -> {tail}")
        print(f"    Joint: {attrs.joint_type}")
        print(f"    Axis: {attrs.joint_axis}")
        print(f"    Position: {attrs.joint_pos:.3f}")
        print(f"    Scale: {attrs.scale:.3f}")
        print(f"    Mirror: {attrs.mirror}")

def print_graph_structure_with_connections(graph):
    """打印graph的完整结构，清晰显示连接关系"""
    print(f"=== Graph: {graph.name} ===")
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    
    # 创建连接映射
    connections_out = {}  # 节点的输出连接
    connections_in = {}   # 节点的输入连接
    for i in range(len(graph.nodes)):
        connections_out[i] = []
        connections_in[i] = []
    
    # 填充连接映射
    for edge_idx, edge in enumerate(graph.edges):
        connections_out[edge.head].append((edge.tail, edge_idx, edge))
        connections_in[edge.tail].append((edge.head, edge_idx, edge))
    
    # 打印节点信息和连接关系
    print("\n--- Nodes with Connections ---")
    for i, node in enumerate(graph.nodes):
        attrs = node.attrs
        
        # 节点基本信息
        base_marker = "[BASE]" if attrs.base else "[NODE]"
        shape_name = {
            1: "CAPSULE",
            2: "CYLINDER", 
            0: "NONE"
        }.get(attrs.shape, "UNKNOWN")
        
        print(f"\n{base_marker} Node {i}: '{node.name}'")
        print(f"    Properties: Shape={shape_name}, Length={attrs.length:.3f}, Radius={attrs.radius:.3f}")
        print(f"    Color: {attrs.color}, Base: {attrs.base}")
        
        # 输入连接
        if connections_in[i]:
            print(f"    INPUT connections:")
            for parent_idx, edge_idx, edge in connections_in[i]:
                parent_name = graph.nodes[parent_idx].name
                joint_name = {
                    0: "NONE",
                    1: "FREE",
                    2: "HINGE",
                    3: "FIXED"
                }.get(edge.attrs.joint_type, "UNKNOWN")
                
                print(f"       <- FROM Node {parent_idx} '{parent_name}' (Edge {edge_idx})")
                print(f"          Joint: {joint_name}, Axis: {edge.attrs.joint_axis}")
                print(f"          Position: {edge.attrs.joint_pos:.3f}, Scale: {edge.attrs.scale:.3f}")
        else:
            print(f"    INPUT connections: None")
        
        # 输出连接
        if connections_out[i]:
            print(f"    OUTPUT connections:")
            for child_idx, edge_idx, edge in connections_out[i]:
                child_name = graph.nodes[child_idx].name
                joint_name = {
                    0: "NONE",
                    1: "FREE", 
                    2: "HINGE",
                    3: "FIXED"
                }.get(edge.attrs.joint_type, "UNKNOWN")
                
                print(f"       -> TO Node {child_idx} '{child_name}' (Edge {edge_idx})")
                print(f"          Joint: {joint_name}, Axis: {edge.attrs.joint_axis}")
                print(f"          Position: {edge.attrs.joint_pos:.3f}, Scale: {edge.attrs.scale:.3f}")
        else:
            print(f"    OUTPUT connections: None")
    
    # 打印连接关系总览
    print(f"\n--- Connection Overview ---")
    for i, edge in enumerate(graph.edges):
        head_node = graph.nodes[edge.head]
        tail_node = graph.nodes[edge.tail]
        attrs = edge.attrs
        
        joint_name = {
            0: "NONE",
            1: "FREE",
            2: "HINGE", 
            3: "FIXED"
        }.get(attrs.joint_type, "UNKNOWN")
        
        print(f"  Edge {i}: '{head_node.name}' --[{joint_name}]--> '{tail_node.name}'")
        print(f"    Joint Type: {joint_name}")
        print(f"    Joint Axis: {attrs.joint_axis}")
        print(f"    Position: {attrs.joint_pos:.3f}")
        print(f"    Scale: {attrs.scale:.3f}, Mirror: {attrs.mirror}")
        if attrs.joint_type == 2:  # HINGE
            print(f"    Limits: [{attrs.joint_lower_limit:.3f}, {attrs.joint_upper_limit:.3f}]")
            print(f"    Torque: {attrs.joint_torque:.3f}, KP: {attrs.joint_kp:.3f}, KD: {attrs.joint_kd:.3f}")
def print_graph_tree_view(graph):
    """以树形结构显示连接关系"""
    print(f"\n=== Tree View: {graph.name} ===")
    
    # 找到根节点（没有输入连接的节点）
    has_input = set()
    for edge in graph.edges:
        has_input.add(edge.tail)
    
    root_nodes = [i for i in range(len(graph.nodes)) if i not in has_input]
    
    if not root_nodes:
        print("ERROR: No root nodes found! (Circular graph?)")
        return
    
    # 创建子节点映射
    children = {}
    for i in range(len(graph.nodes)):
        children[i] = []
    
    for edge in graph.edges:
        children[edge.head].append((edge.tail, edge))
    
    def print_tree_node(node_idx, level=0, visited=None):
        if visited is None:
            visited = set()
        
        if node_idx in visited:
            print("  " * level + "WARNING: Circular reference detected!")
            return
        
        visited.add(node_idx)
        node = graph.nodes[node_idx]
        attrs = node.attrs
        
        # 节点标记
        base_marker = "[BASE]" if attrs.base else "[NODE]"
        shape_name = {
            1: "CAPSULE",
            2: "CYLINDER",
            0: "NONE"
        }.get(attrs.shape, "UNKNOWN")
        
        indent = "  " * level
        print(f"{indent}{base_marker} {node.name} (Node {node_idx}) - {shape_name}")
        print(f"{indent}   Properties: L={attrs.length:.3f}, R={attrs.radius:.3f}")
        
        # 打印子节点
        for child_idx, edge in children[node_idx]:
            if child_idx not in visited:
                joint_name = {
                    0: "NONE",
                    1: "FREE",
                    2: "HINGE",
                    3: "FIXED"
                }.get(edge.attrs.joint_type, "UNKNOWN")
                
                print(f"{indent}  |")
                print(f"{indent}  +-- [{joint_name}] Joint")
                print_tree_node(child_idx, level + 1, visited.copy())
    
    # 从根节点开始打印
    for root_idx in root_nodes:
        print_tree_node(root_idx)

def run_basic_simulation():
    # 1. 设置任务
    task = tasks.FlatTerrainTask(episode_len=128)
    
    # 2. 加载语法规则
    graphs = rd.load_graphs("data/designs/grammar_apr30.dot")  # 替换为你的语法文件路径
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    
    # 3. 定义规则序列（这里用示例序列，你需要根据你的语法调整）
    rule_sequence = [0, 2, 9, 5, 6, 7, 14]  # 替换为你的规则序列
    
    # 4. 构建机器人
    from design_search import make_graph, build_normalized_robot, presimulate
    
    graph = make_graph(rules, rule_sequence)
    # print_graph_structure(graph=graph)
    # print_graph_structure_with_connections(graph=graph)
    print_graph_tree_view(graph=graph)
    robot = build_normalized_robot(graph)
    
    # 5. 设置机器人外观（可选）
    for link in robot.links:
        if link.shape == rd.LinkShape.NONE:
            link.shape = rd.LinkShape.CAPSULE
            link.length = 0.1
            link.radius = 0.025
            link.color = [1.0, 0.0, 1.0]
        if link.joint_type == rd.JointType.NONE:
            link.joint_type = rd.JointType.FIXED
            link.joint_color = [1.0, 0.0, 1.0]
    
    # 6. 创建仿真环境
    sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(sim)
    
    # 7. 计算机器人初始位置并添加到仿真
    robot_init_pos, has_self_collision = presimulate(robot)
    print(f"robot_init_pos: {robot_init_pos}, has_self_collision: {has_self_collision}")
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = sim.find_robot_index(robot)
    
    # 8. 创建可视化窗口
    viewer = rd.GLFWViewer()
    
    # 9. 设置相机参数
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_robot_world_aabb(robot_idx, lower, upper)
    viewer.camera_params.position = 0.5 * (lower + upper)
    viewer.camera_params.yaw = -np.pi / 4
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 1.5 * np.linalg.norm(upper - lower)
    
    # 10. 仿真循环
    sim_time = 0
    while not viewer.should_close():
        # 更新仿真
        sim.step()
        
        # 更新相机（可选：跟踪机器人）
        lower = np.zeros(3)
        upper = np.zeros(3)
        sim.get_robot_world_aabb(robot_idx, lower, upper)
        target_pos = 0.5 * (lower + upper)
        camera_pos = viewer.camera_params.position.copy()
        camera_pos += 5.0 * task.time_step * (target_pos - camera_pos)
        viewer.camera_params.position = camera_pos
        
        # 更新和渲染
        viewer.update(task.time_step)
        viewer.render(sim)
        
        sim_time += task.time_step

if __name__ == '__main__':
    run_basic_simulation()
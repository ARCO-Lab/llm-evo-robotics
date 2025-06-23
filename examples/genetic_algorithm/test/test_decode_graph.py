import argparse
import pyrobotdesign as rd
import numpy as np
import tasks
import os
import sys
import random
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
print(f"base_dir: {base_dir}")
sys.path.append(os.path.join(base_dir, 'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))
print(f"sys.path: {sys.path}")
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals, simulate, presimulate
from viewer import view_trajectory


def decode_gene_to_graph(structure):
    graph = rd.Graph()
    id_to_index = {}
    
    # Step 1: 创建所有节点
    nodes_list = []
    for node in structure:
        gnode = rd.Node()
        gnode.name = f"n{node['id']}"
        gnode.attrs.label = node["type"]
        gnode.attrs.shape = rd.LinkShape.CAPSULE
        gnode.attrs.length = node["geometry"]["length"]
        gnode.attrs.radius = node["geometry"]["radius"]
        gnode.attrs.density = node["dynamic"]["density"]
        gnode.attrs.friction = 0.5  # 默认摩擦系数
        
        # 添加到临时列表
        nodes_list.append(gnode)
        # 记录节点ID到索引的映射
        id_to_index[int(node["id"])] = len(nodes_list) - 1
    
    # 一次性赋值给图
    graph.nodes = nodes_list
    
    # 调试信息：打印节点映射
    print(f"节点映射: {id_to_index}")
    print(f"节点总数: {len(graph.nodes)}")
    
    # Step 2: 创建所有边（连接）
    edges_list = []
    for node in structure:
        node_id = int(node["id"])
        
        # 确保当前节点存在于映射中
        if node_id not in id_to_index:
            raise ValueError(f"节点 ID {node_id} 未出现在 id_to_index 中")
        
        tail_index = id_to_index[node_id]
        print(f"处理节点 {node_id}, tail_index: {tail_index}")
        
        # 处理每个子节点
        for child_id in node["children"]:
            child_id = int(child_id)
            
            # 确保子节点存在于映射中
            if child_id not in id_to_index:
                raise ValueError(f"子节点 ID {child_id} 未出现在 id_to_index 中")
            
            head_index = id_to_index[child_id]
            print(f"  连接到子节点 {child_id}, head_index: {head_index}")
            
            # 验证索引有效性
            if tail_index < 0 or head_index < 0:
                raise ValueError(f"无效的索引: tail_index={tail_index}, head_index={head_index}")
            
            # 找到子节点的详细信息
            child = next((n for n in structure if int(n["id"]) == child_id), None)
            if child is None:
                raise ValueError(f"结构中找不到 id 为 {child_id} 的子节点")
            
            # 创建边
            edge = rd.Edge()
            edge.tail = tail_index
            edge.head = head_index
            
            # 设置关节信息 - 修复运动问题
            joint = child.get("joint", None)
            if joint:
                # 方案1: 使用固定关节（完全静止）
                edge.attrs.joint_type = rd.JointType.HINGE
                
                # 方案2: 如果需要可动关节但保持静止，使用以下设置
                # edge.attrs.joint_type = rd.JointType.HINGE
                # edge.attrs.joint_axis = joint["axis"]
                # edge.attrs.joint_pos = 0.0  # 目标位置为0
                # edge.attrs.joint_kp = 10000.0  # 非常高的比例增益
                # edge.attrs.joint_kd = 1000.0   # 高微分增益
                # edge.attrs.joint_torque = 0.0  # 关键：不施加额外扭矩
                # edge.attrs.joint_lower_limit = joint["limit"][0]
                # edge.attrs.joint_upper_limit = joint["limit"][1]
                # edge.attrs.joint_control_mode = rd.JointControlMode.POSITION
                
            edges_list.append(edge)
    
    # 一次性赋值给图
    graph.edges = edges_list
    
    return graph
def run_simulation_test(robot):
    task = tasks.FlatTerrainTask(episode_len=128)
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
    print(f"机器人初始位置: {robot_init_pos}, 是否自碰撞: {has_self_collision}")
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(1.0, 0.0, 0.0, 0.0))
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


def run_simulation_1(robot, task, opt_seed, jobs, episodes):
    input_sequence, result = simulate(robot, task, opt_seed, jobs, episodes)
    print("Result:", result, "input_sequence:", input_sequence)
    robot_init_pos, has_self_collision = presimulate(robot)
    if has_self_collision:
        print("Warning: robot self-collides in initial configuration")

    main_sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(main_sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = main_sim.find_robot_index(robot)


       
    # 在调用 view_trajectory 之前，检查关节目标值
    print(f"🔍 关节初始状态:")
    joint_positions = np.zeros(main_sim.get_robot_dof_count(robot_idx))
    main_sim.get_joint_positions(robot_idx, joint_positions)
    print(f"  初始关节位置: {joint_positions}")
    camera_params, record_step_indices = view_trajectory(
        main_sim, robot_idx, input_sequence, task)
    


def run_simulation(robot, task, opt_seed, jobs, episodes):
    # 添加任务参数调试
    print(f"🔍 Task 参数:")
    print(f"  force_std: {task.force_std}")
    print(f"  torque_std: {task.torque_std}")
    print(f"  noise_seed: {task.noise_seed}")
    print(f"  time_step: {task.time_step}")
    print(f"  interval: {task.interval}")
    
    input_sequence, result = simulate(robot, task, opt_seed, jobs, episodes)
    print("Result:", result, "input_sequence:", input_sequence)
    
    # 添加机器人关节信息调试
    print(f"🔍 机器人关节信息:")
    for i, link in enumerate(robot.links):
        print(f"  Link {i}:")
        print(f"    joint_type: {link.joint_type}")
        print(f"    joint_kp: {link.joint_kp}")
        print(f"    joint_kd: {link.joint_kd}")
        print(f"    joint_torque: {link.joint_torque}")
        if hasattr(link, 'joint_control_mode'):
            print(f"    joint_control_mode: {link.joint_control_mode}")
        if hasattr(link, 'joint_pos'):
            print(f"    joint_pos: {link.joint_pos}")
    
    robot_init_pos, has_self_collision = presimulate(robot)
    if has_self_collision:
        print("Warning: robot self-collides in initial configuration")

    main_sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(main_sim)
    main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = main_sim.find_robot_index(robot)
    
    # 检查关节初始状态
    print(f"🔍 关节初始状态:")
    joint_positions = np.zeros(main_sim.get_robot_dof_count(robot_idx))
    main_sim.get_joint_positions(robot_idx, joint_positions)
    print(f"  初始关节位置: {joint_positions}")
    print(f"  机器人DOF数量: {main_sim.get_robot_dof_count(robot_idx)}")
    
    # 移除 get_joint_targets 调用，因为该方法不存在
    
    camera_params, record_step_indices = view_trajectory(
        main_sim, robot_idx, input_sequence, task)
    


if __name__ == "__main__":
    # 首先检查模块中有哪些属性
    print("pyrobotdesign 模块属性:")
    for attr in dir(rd):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    # 检查 EdgeAttributes 的默认值
    edge_attrs = rd.EdgeAttributes()
    print(f"\n默认 EdgeAttributes:")
    print(f"  joint_axis 类型: {type(edge_attrs.joint_axis)}")
    print(f"  joint_axis 值: {edge_attrs.joint_axis}")
    print(f"  joint_rot 类型: {type(edge_attrs.joint_rot)}")
    print(f"  joint_rot 值: {edge_attrs.joint_rot}")
    
    initial_structure = [
        {
            "id": 0,
            "type": "torso",
            "pos": (0.0, 0.0, 0.0),
            "geometry": {"length": 0.3, "radius": 0.05},
            "dynamic": {"mass": 1.0, "density": 500},
            "joint": None,
            "children": [1, 2]
        },
        {
            "id": 1,
            "type": "leg",
            "pos": (0.15, 0.0, -0.2),
            "geometry": {"length": 0.2, "radius": 0.03},
            "dynamic": {"mass": 0.3, "density": 600},
            "joint": {
                "type": "revolute",
                "axis": (0, 1, 0),
                "init_angle": 0.0,
                "limit": (-0.5, 0.5),
                "speed": 1.0
            },
            "children": []
        },
        {
            "id": 2,
            "type": "leg",
            "pos": (-0.15, 0.0, -0.2),
            "geometry": {"length": 0.2, "radius": 0.03},
            "dynamic": {"mass": 0.3, "density": 600},
            "joint": {
                "type": "revolute",
                "axis": (0, 1, 0),
                "init_angle": 0.0,
                "limit": (-0.5, 0.5),
                "speed": 1.0
            },
            "children": []
        }
    ]
    
    try:
        graph = decode_gene_to_graph(initial_structure)
        print("✅ 成功转化为 rd.Graph")
        print(f"- 节点数: {len(graph.nodes)}")
        print(f"- 边数: {len(graph.edges)}")
        print(f"- 节点名称: {[n.name for n in graph.nodes]}")
        robot = build_normalized_robot(graph)
        print("✅ 成功转化为 rd.Robot")
        print(f"- 连杆数: {len(robot.links)}")
        print(f"- 连杆名称: {[n.joint_type for n in robot.links]}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    
    parser = argparse.ArgumentParser(description="Genetic Algorithm Robot design viewer.")
    parser.add_argument("task", type=str, help="Task (Python class name)")
    parser.add_argument("-e", "--episode_len", type=int, default=128, help="Length of episode")
    parser.add_argument("-j", "--jobs", type=int, required=True, help="Number of jobs/threads")

    args = parser.parse_args()
    task_class = getattr(tasks, args.task)
    task = task_class(episode_len=args.episode_len)

    
    
    opt_seed = random.getrandbits(32)
    print("Using optimization seed:", opt_seed)

    # run_simulation(robot, task, opt_seed, args.jobs, args.episode_len)



        # 2. 加载语法规则
    graphs = rd.load_graphs("data/designs/grammar_apr30.dot")  # 替换为你的语法文件路径
    rules = [rd.create_rule_from_graph(g) for g in graphs]
        # 3. 定义规则序列（这里用示例序列，你需要根据你的语法调整）
    rule_sequence = [0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8]  # 替换为你的规则序列
    
    # 4. 构建机器人
    from design_search import make_graph, build_normalized_robot, presimulate
    
    graph = make_graph(rules, rule_sequence)

    for edge in graph.edges:
        print(f"edge: {edge.tail} -> {edge.head}")
        if edge.attrs.joint_type == rd.JointType.FIXED:
            edge.attrs.joint_type = rd.JointType.HINGE
  
    robot = build_normalized_robot(graph)


    run_simulation_test(robot)

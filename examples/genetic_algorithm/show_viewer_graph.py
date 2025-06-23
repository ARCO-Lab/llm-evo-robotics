import numpy as np
import pyrobotdesign as rd
import tasks
import os
import sys

# 设置路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'design_search'))

from design_search import make_graph, build_normalized_robot

def print_graph_structure_detailed(graph):
    """打印graph的完整详细结构"""
    print("="*80)
    print(f"📊 VIEWER.PY 生成的图结构分析")
    print("="*80)
    print(f"图名称: {graph.name}")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    
    # 1. 打印所有节点信息
    print("\n" + "="*50)
    print("📋 节点详细信息")
    print("="*50)
    
    for i, node in enumerate(graph.nodes):
        attrs = node.attrs
        base_marker = "[🏠 BASE]" if attrs.base else "[🔗 NODE]"
        
        shape_name = {
            0: "NONE",
            1: "CAPSULE", 
            2: "CYLINDER"
        }.get(attrs.shape, f"UNKNOWN({attrs.shape})")
        
        print(f"\n{base_marker} 节点 {i}: '{node.name}'")
        print(f"    标签: {attrs.label}")
        print(f"    形状: {shape_name}")
        print(f"    尺寸: 长度={attrs.length:.4f}, 半径={attrs.radius:.4f}")
        print(f"    物理: 密度={attrs.density:.1f}, 摩擦={attrs.friction:.2f}")
        print(f"    颜色: {attrs.color}")
        print(f"    基节点: {attrs.base}")
    
    # 2. 分析连接关系
    print("\n" + "="*50)
    print("🔗 边/连接详细信息")
    print("="*50)
    
    # 创建连接映射
    connections_from = {}  # 从某节点出发的连接
    connections_to = {}    # 指向某节点的连接
    
    for i in range(len(graph.nodes)):
        connections_from[i] = []
        connections_to[i] = []
    
    for edge_idx, edge in enumerate(graph.edges):
        connections_from[edge.head].append((edge.tail, edge_idx, edge))
        connections_to[edge.tail].append((edge.head, edge_idx, edge))
    
    for i, edge in enumerate(graph.edges):
        head_node = graph.nodes[edge.head]
        tail_node = graph.nodes[edge.tail]
        attrs = edge.attrs
        
        joint_type_name = {
            0: "NONE",
            1: "FREE",
            2: "HINGE",
            3: "FIXED"
        }.get(attrs.joint_type, f"UNKNOWN({attrs.joint_type})")
        
        print(f"\n🔗 边 {i}: '{head_node.name}' --[{joint_type_name}]--> '{tail_node.name}'")
        print(f"    头节点: {head_node.name} (索引 {edge.head})")
        print(f"    尾节点: {tail_node.name} (索引 {edge.tail})")
        print(f"    关节类型: {joint_type_name}")
        
        if attrs.joint_type == rd.JointType.HINGE:
            print(f"    关节轴: {attrs.joint_axis}")
            print(f"    关节位置: {attrs.joint_pos:.4f}")
            print(f"    关节旋转: {attrs.joint_rot}")
            print(f"    控制参数: KP={attrs.joint_kp:.3f}, KD={attrs.joint_kd:.3f}")
            print(f"    扭矩: {attrs.joint_torque:.3f}")
            print(f"    限制: [{attrs.joint_lower_limit:.3f}, {attrs.joint_upper_limit:.3f}]")
            print(f"    控制模式: {attrs.joint_control_mode}")
        
        print(f"    缩放: {attrs.scale:.3f}")
        print(f"    镜像: {attrs.mirror}")
        print(f"    颜色: {attrs.color}")
        print(f"    ID: '{attrs.id}'")
        print(f"    标签: '{attrs.label}'")
    
    # 3. 树形结构视图
    print("\n" + "="*50)
    print("🌳 树形结构视图")
    print("="*50)
    
    # 找到根节点
    has_incoming = set()
    for edge in graph.edges:
        has_incoming.add(edge.tail)
    
    root_nodes = [i for i in range(len(graph.nodes)) if i not in has_incoming]
    
    if not root_nodes:
        print("❌ 错误: 没有找到根节点！(可能存在循环)")
        return
    
    def print_tree_node(node_idx, level=0, visited=None):
        if visited is None:
            visited = set()
        
        if node_idx in visited:
            print("  " * level + "⚠️ 警告: 检测到循环引用!")
            return
        
        visited.add(node_idx)
        node = graph.nodes[node_idx]
        attrs = node.attrs
        
        base_marker = "🏠" if attrs.base else "🔗"
        shape_name = {
            0: "NONE",
            1: "CAPSULE",
            2: "CYLINDER"
        }.get(attrs.shape, "UNKNOWN")
        
        indent = "  " * level
        print(f"{indent}{base_marker} {node.name} (节点 {node_idx}) - {shape_name}")
        print(f"{indent}   📏 L={attrs.length:.3f}, R={attrs.radius:.3f}, ρ={attrs.density:.0f}")
        
        # 打印子节点
        if node_idx in connections_from:
            for child_idx, edge_idx, edge in connections_from[node_idx]:
                if child_idx not in visited:
                    joint_name = {
                        0: "NONE",
                        1: "FREE",
                        2: "HINGE",
                        3: "FIXED"
                    }.get(edge.attrs.joint_type, "UNKNOWN")
                    
                    print(f"{indent}  │")
                    print(f"{indent}  ├── 🔧 [{joint_name}] 关节")
                    if edge.attrs.joint_type == rd.JointType.HINGE:
                        print(f"{indent}  │   轴: {edge.attrs.joint_axis}")
                        print(f"{indent}  │   KP: {edge.attrs.joint_kp:.1f}, KD: {edge.attrs.joint_kd:.1f}")
                    print_tree_node(child_idx, level + 1, visited.copy())
    
    # 从根节点开始打印
    for root_idx in root_nodes:
        print(f"\n🌱 从根节点 {root_idx} 开始:")
        print_tree_node(root_idx)
    
    # 4. 连通性分析
    print("\n" + "="*50)
    print("🔍 连通性分析")
    print("="*50)
    
    # 构建无向图邻接表
    adjacency = {i: [] for i in range(len(graph.nodes))}
    for edge in graph.edges:
        adjacency[edge.head].append(edge.tail)
        adjacency[edge.tail].append(edge.head)
    
    # DFS检查连通性
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in adjacency[node]:
            dfs(neighbor)
    
    dfs(0)
    connected_nodes = len(visited)
    
    if connected_nodes == len(graph.nodes):
        print("✅ 图是完全连通的")
    else:
        print(f"❌ 图不完全连通!")
        print(f"   连通节点数: {connected_nodes}/{len(graph.nodes)}")
        print(f"   未连通节点: {set(range(len(graph.nodes))) - visited}")
    
    # 5. 机器人构建预览
    print("\n" + "="*50)
    print("🤖 机器人构建预览")
    print("="*50)
    
    try:
        robot = build_normalized_robot(graph)
        print(f"✅ 成功构建机器人")
        print(f"   连杆数: {len(robot.links)}")
        
        print(f"\n连杆详情:")
        for i, link in enumerate(robot.links):
            parent_name = f"连杆{link.parent}" if link.parent >= 0 else "无(根连杆)"
            joint_name = {
                0: "NONE",
                1: "FREE", 
                2: "HINGE",
                3: "FIXED"
            }.get(link.joint_type, "UNKNOWN")
            
            print(f"  连杆 {i}: {link.label}")
            print(f"    父连杆: {parent_name}")
            print(f"    关节类型: {joint_name}")
            print(f"    尺寸: L={link.length:.3f}, R={link.radius:.3f}")
            if link.joint_type == rd.JointType.HINGE:
                print(f"    关节参数: KP={link.joint_kp:.1f}, KD={link.joint_kd:.1f}")
        
    except Exception as e:
        print(f"❌ 机器人构建失败: {e}")

def main():
    print("🚀 开始分析 viewer.py 中的图结构...")
    
    try:
        # 加载语法规则
        graphs = rd.load_graphs("data/designs/grammar_apr30.dot")
        rules = [rd.create_rule_from_graph(g) for g in graphs]
        
        # 使用与 viewer.py 相同的规则序列
        rule_sequence = [0, 1, 2]
        
        # 构建图
        graph = make_graph(rules, rule_sequence)
        
        # 详细分析图结构
        print_graph_structure_detailed(graph)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
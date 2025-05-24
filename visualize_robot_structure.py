import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pybullet as p

def get_shape_name(shape_id):
    """Return the name of the shape"""
    shapes = {
        0: "Box",
        1: "Cylinder",
        2: "Sphere"
    }
    return shapes.get(shape_id, "Unknown")

def visualize_robot_structure(robot_config, save_path=None, figsize=(12, 10), show_details=True):
    """
    Visualize the tree structure of the robot
    
    Args:
        robot_config (dict): Robot configuration dictionary
        save_path (str, optional): Path to save the image, if None the image will be displayed
        figsize (tuple, optional): Image size
        show_details (bool, optional): Whether to show detailed link information
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Robot Structure Visualization', fontsize=16)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and attributes
    node_types = []
    node_sizes = []
    node_details = []
    
    for i in range(robot_config['num_links']):
        # Set node attributes
        if i == 0:
            node_type = "Base"
            node_size = 1500
        elif robot_config['is_wheel'][i]:
            node_type = "Wheel"
            node_size = 1000
        else:
            # Determine node type by joint type
            joint_type = robot_config['joint_types'][i]
            if joint_type == p.JOINT_REVOLUTE:
                node_type = "Revolute Joint"
            elif joint_type == p.JOINT_PRISMATIC:
                node_type = "Prismatic Joint"
            elif joint_type == p.JOINT_FIXED:
                node_type = "Fixed Joint"
            else:
                node_type = "Other Joint"
            
            # Adjust node size based on whether it has a motor
            node_size = 800 if robot_config['has_motor'][i] else 500
        
        # Build node detail information
        if i == 0:
            details = f"Base - Shape: {get_shape_name(robot_config['shapes'][i])}"
        else:
            material = ["Metal", "Plastic", "Rubber"][robot_config['link_materials'][i]]
            has_motor = "With Motor" if robot_config['has_motor'][i] else "No Motor"
            
            if robot_config['is_wheel'][i]:
                wheel_type = "Regular Wheel" if robot_config['wheel_types'][i] == 0 else "Omni Wheel"
                size_info = f"Radius: {robot_config['link_sizes'][i][0]:.2f}, Width: {robot_config['link_sizes'][i][2]:.2f}"
                details = f"{node_type} - {wheel_type}, {material}, {has_motor}\n{size_info}"
            else:
                shape = get_shape_name(robot_config['shapes'][i])
                size = robot_config['link_sizes'][i]
                size_info = f"Size: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]"
                details = f"{node_type} - Shape: {shape}, {material}, {has_motor}\n{size_info}"
        
        # Add to lists
        node_types.append(node_type)
        node_sizes.append(node_size)
        node_details.append(details)
        
        # Add node and attributes
        G.add_node(i, type=node_type, size=node_size, details=details)
    
    # Add edges
    for i in range(1, robot_config['num_links']):
        parent_idx = robot_config['parent_indices'][i]
        if parent_idx >= 0 and parent_idx < robot_config['num_links']:
            # Add edge and connection point position attribute
            conn_point = robot_config['connection_points'][i]
            G.add_edge(parent_idx, i, connection=conn_point)
    
    # Set node colors
    node_colors = []
    for i in range(robot_config['num_links']):
        if i == 0:
            node_colors.append('lightblue')  # Base
        elif robot_config['is_wheel'][i]:
            node_colors.append('green')  # Wheel
        else:
            joint_type = robot_config['joint_types'][i]
            if joint_type == p.JOINT_REVOLUTE:
                node_colors.append('orange')  # Revolute joint
            elif joint_type == p.JOINT_PRISMATIC:
                node_colors.append('purple')  # Prismatic joint
            elif joint_type == p.JOINT_FIXED:
                node_colors.append('gray')  # Fixed joint
            else:
                node_colors.append('red')
    
    # Use hierarchical layout
    try:
        # Try to use graphviz layout, requires pygraphviz
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # If pygraphviz is not installed, use built-in layout
        pos = nx.spring_layout(G)
    
    # Draw network graph
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color=node_colors, 
            node_size=node_sizes, font_weight='bold', arrows=True,
            connectionstyle='arc3,rad=0.1', arrowsize=15)
    
    # Add node type legend
    ax1.set_title('Robot Structure Tree')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Base'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Wheel'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Revolute Joint'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Prismatic Joint'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Fixed Joint')
    ]
    ax1.legend(handles=legend_elements, loc='best')
    
    # Show edge connection information
    if show_details:
        edge_labels = {(u, v): f"Connection: {d['connection']:.2f}" 
                     for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, 
                                    font_size=8, label_pos=0.5)
    
    # Show detailed information table on the right
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(["ID", "Type", "Details"])
    for i in range(robot_config['num_links']):
        table_data.append([str(i), node_types[i], node_details[i]])
    
    # Create table
    table = ax2.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.1, 0.2, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax2.set_title('Link Details')
    
    # Add basic statistics
    stats_text = (
        f"Total Links: {robot_config['num_links']}\n"
        f"Wheel Count: {sum(robot_config['is_wheel'])}\n"
        f"Max Depth: {max(robot_config['node_depths'])}\n"
        f"Motor Count: {sum(robot_config['has_motor'])}"
    )
    ax2.text(0.1, 0.01, stats_text, transform=ax2.transAxes, 
             bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or display image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Image saved to {save_path}")
        return None
    else:
        plt.show()
        return fig

# Test function
if __name__ == "__main__":
    # Test code: generate random gene and visualize
    import numpy as np
    # Assume decode_gene function is defined in another module
    from robot_evolution_test import decode_gene
    
    # Create random gene
    random_gene = np.random.random(100)
    
    # Decode gene
    robot_config = decode_gene(random_gene)
    
    # Visualize robot structure
    visualize_robot_structure(robot_config)
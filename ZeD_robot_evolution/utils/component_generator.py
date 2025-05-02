import numpy as np
import trimesh

def generate_polygon(sides=6, radius=1.0, height=0.3):
    """
    生成一个完整的 3D 多边形，包括底面、顶面和侧面。

    参数：
        sides (int)   : 多边形的边数（必须 >= 3）
        radius (float): 半径
        height (float): 厚度（extrude 高度）

    返回：
        trimesh.Trimesh: 生成的 3D Mesh
    """
    if sides < 3:
        raise ValueError("多边形的边数必须 >= 3")

    # 计算底面顶点（均匀分布在圆周上）
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    bottom_vertices = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(sides)], axis=-1)

    # 复制一份作为顶部点，并增加高度
    top_vertices = bottom_vertices.copy()
    top_vertices[:, 2] += height  # 让顶部点抬高

    # 合并所有点（底面 + 顶面）
    all_vertices = np.vstack([bottom_vertices, top_vertices])

    # 计算面索引
    faces = []

    # 1️⃣ **生成底面**（使用 `sides` 个三角形）
    center_bottom = len(all_vertices)  # 添加底面中心点
    all_vertices = np.vstack([all_vertices, [[0, 0, 0]]])  # 底面中心点
    for i in range(sides):
        next_i = (i + 1) % sides  # 确保形成闭环
        faces.append([center_bottom, next_i, i])  # 保证顺序一致

    # 2️⃣ **生成顶面**（与底面对应）
    center_top = len(all_vertices)  # 添加顶面中心点
    all_vertices = np.vstack([all_vertices, [[0, 0, height]]])  # 顶面中心点
    top_offset = sides  # 顶面索引偏移
    for i in range(sides):
        next_i = (i + 1) % sides
        faces.append([center_top, top_offset + i, top_offset + next_i])  # 让面封闭

    # 3️⃣ **生成侧面**（每条边形成两个三角形）
    for i in range(sides):
        next_i = (i + 1) % sides  # 确保形成环状连接
        faces.append([i, next_i, top_offset + i])  # 侧面三角形 1
        faces.append([next_i, top_offset + next_i, top_offset + i])  # 侧面三角形 2

    return trimesh.Trimesh(vertices=all_vertices, faces=faces)

# 🎯 **生成完整封闭的六边形**
hexagon_mesh = generate_polygon(sides=3, radius=1.0, height=0.3)

# 🔹 **缩小整体大小**
scale = 0.1
hexagon_mesh.apply_scale(scale)

# 💾 **保存为 .obj**
hexagon_mesh.export('../configs/triangle.obj')

print("✅ `triangle.obj` 已生成，修正了缺少的上下表面！")

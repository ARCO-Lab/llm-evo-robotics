import numpy as np
import pybullet as p
import os

# 检查原始文件是否存在
robot_evolution_file = "robot_evolution_fixed.py"
if not os.path.exists(robot_evolution_file):
    print(f"错误：找不到文件 {robot_evolution_file}")
    exit(1)

# 读取原始文件内容
with open(robot_evolution_file, 'r', encoding='utf-8') as f:
    original_content = f.read()

# 查找decode_gene函数中处理轮子旋转轴的部分
wheel_axis_line = "            # 轮子通常围绕Y轴旋转"
wheel_axis_setting = "                joint_axes.append([0, 1, 0])"

# 检查这些关键行是否存在
if wheel_axis_line not in original_content or wheel_axis_setting not in original_content:
    print("警告：在原始文件中找不到轮子旋转轴设置的代码行")
    print("请手动修改robot_evolution_fixed.py文件")
    exit(1)

# 准备替换内容 - 将Y轴旋转改为Z轴旋转
modified_content = original_content.replace(
    wheel_axis_line, 
    "            # 轮子强制使用Z轴旋转"
)
modified_content = modified_content.replace(
    wheel_axis_setting,
    "                joint_axes.append([0, 0, 1])  # 使用Z轴旋转"
)

# 创建备份文件
backup_file = f"{robot_evolution_file}.bak"
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(original_content)
print(f"已创建原始文件备份: {backup_file}")

# 写入修改后的文件
with open(robot_evolution_file, 'w', encoding='utf-8') as f:
    f.write(modified_content)
print(f"已修改 {robot_evolution_file}，所有轮子现在将使用Z轴旋转")

# 额外创建一个恢复脚本
restore_script = "restore_gene_decode.py"
with open(restore_script, 'w', encoding='utf-8') as f:
    f.write(f'''
import os
import shutil

# 检查备份文件是否存在
backup_file = "{backup_file}"
if not os.path.exists(backup_file):
    print(f"错误：找不到备份文件 {{backup_file}}")
    exit(1)

# 恢复原始文件
shutil.copy(backup_file, "{robot_evolution_file}")
print(f"已从 {{backup_file}} 恢复原始文件 {robot_evolution_file}")
''')

print(f"已创建恢复脚本: {restore_script}")
print("执行 'python restore_gene_decode.py' 可以恢复原始文件")
print("\n请重新运行 fix_robot_model.py 来测试带有Z轴轮子的机器人进化") 
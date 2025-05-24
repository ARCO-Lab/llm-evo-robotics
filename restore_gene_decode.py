
import os
import shutil

# 检查备份文件是否存在
backup_file = "robot_evolution_fixed.py.bak"
if not os.path.exists(backup_file):
    print(f"错误：找不到备份文件 {backup_file}")
    exit(1)

# 恢复原始文件
shutil.copy(backup_file, "robot_evolution_fixed.py")
print(f"已从 {backup_file} 恢复原始文件 robot_evolution_fixed.py")

# 🌙 过夜实验设置指南

## 🎯 快速设置（5分钟）

### 1. 启动长时间实验

```bash
# 进入项目目录
cd /home/xli149/Documents/repos/test_robo

# 激活环境并启动实验
source /home/xli149/Documents/repos/RoboGrammar/venv/bin/activate

# 启动过夜实验（推荐使用nohup确保不被中断）
nohup python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared > overnight_experiment.log 2>&1 &

# 记录进程ID
echo $! > experiment_pid.txt
```

### 2. 检查实验状态

```bash
# 查看实验日志（实时）
tail -f overnight_experiment.log

# 查看实验进度
python examples/surrogate_model/map_elites/view_results.py

# 检查进程是否还在运行
ps -p $(cat experiment_pid.txt)
```

### 3. 第二天查看结果

```bash
# 查看所有实验会话
python examples/surrogate_model/map_elites/view_results.py --list

# 查看最新结果
python examples/surrogate_model/map_elites/view_results.py

# 导出成功结构
python examples/surrogate_model/map_elites/view_results.py --export json
```

## ⚙️ 实验参数说明

当前配置已经针对过夜实验优化：

- **成功阈值**: `0.6` (适中标准，不会太严格)
- **初始种群**: `4个个体` (并行可视化)
- **每个体训练步数**: `500步` (快速评估)
- **进化代数**: `3代` (可在代码中调整)
- **多进程**: `启用` (4个工作进程)
- **共享PPO**: `启用` (提高训练效率)
- **自动记录**: `启用` (记录所有成功结果)

## 📊 预期结果

### 过夜实验（8-10小时）预期：
- **总实验数**: 50-100个个体
- **成功个体数**: 15-30个（假设30%成功率）
- **最佳fitness**: 0.7-0.9
- **发现的成功结构**: 多种关节数配置

### 文件输出：
```
./experiment_results/session_YYYYMMDD_HHMMSS/
├── results.json                  # 所有实验结果
├── successful_results.json       # 仅成功的实验
├── results.csv                   # CSV格式（Excel可打开）
├── successful_results.csv        # 成功结果CSV
└── session_summary.txt           # 最终总结
```

## 🔍 实时监控

### 在另一个终端中设置监控：

```bash
# 每30秒更新一次实验状态
watch -n 30 "python examples/surrogate_model/map_elites/view_results.py | head -15"

# 或者创建简单的监控脚本
cat << 'EOF' > monitor_experiment.sh
#!/bin/bash
while true; do
    clear
    echo "=== 实验监控 $(date) ==="
    echo
    python examples/surrogate_model/map_elites/view_results.py | head -20
    echo
    echo "=== 最新日志 ==="
    tail -n 5 overnight_experiment.log
    sleep 60
done
EOF

chmod +x monitor_experiment.sh
./monitor_experiment.sh
```

## 🛑 安全停止实验

如果需要提前停止：

```bash
# 优雅停止（推荐）
kill $(cat experiment_pid.txt)

# 强制停止（如果优雅停止无效）
kill -9 $(cat experiment_pid.txt)

# 清理可能的子进程
pkill -f "map_elites_trainer.py"
```

## 📈 结果分析脚本

创建一个快速分析脚本：

```bash
cat << 'EOF' > analyze_results.py
#!/usr/bin/env python3
import subprocess
import sys
import os

# 激活环境并分析结果
os.chdir('/home/xli149/Documents/repos/test_robo')

print("🔍 分析实验结果...")
print("=" * 50)

# 查看会话列表
result = subprocess.run([
    'python', 'examples/surrogate_model/map_elites/view_results.py', '--list'
], capture_output=True, text=True)
print(result.stdout)

# 分析最新会话
result = subprocess.run([
    'python', 'examples/surrogate_model/map_elites/view_results.py'
], capture_output=True, text=True)
print(result.stdout)

# 导出成功结构
result = subprocess.run([
    'python', 'examples/surrogate_model/map_elites/view_results.py', '--export', 'json'
], capture_output=True, text=True)
print(result.stdout)

print("\n✅ 分析完成！")
EOF

chmod +x analyze_results.py
```

## 🎯 实验目标记录

在开始实验前，记录你的目标：

```bash
cat << EOF > experiment_goals.txt
实验日期: $(date)
实验目标: 
- 寻找fitness > 0.6的成功机器人结构
- 比较不同关节数(3-6)的性能
- 收集至少20个成功案例

预期假设:
- 4-5关节机器人可能表现最好
- 总长度在200-400px的机器人更容易成功
- 学习率在1e-4到3e-4范围内效果较好

成功标准:
- 成功率 > 20%
- 最佳fitness > 0.7
- 至少3种不同关节数的成功结构
EOF
```

## 📞 问题排查

### 常见问题及解决方案：

1. **实验卡住不动**
```bash
# 检查GPU使用情况
nvidia-smi

# 检查内存使用
free -h

# 重启实验
kill $(cat experiment_pid.txt)
nohup python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared > overnight_experiment_restart.log 2>&1 &
```

2. **磁盘空间不足**
```bash
# 检查磁盘空间
df -h

# 清理旧的实验结果（小心操作）
rm -rf ./experiment_results/session_OLD_DATE_*
```

3. **可视化窗口过多**
```bash
# 使用无可视化模式
nohup python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --no-render > overnight_experiment.log 2>&1 &
```

## 🎉 实验完成后的行动清单

- [ ] 查看最终结果统计
- [ ] 导出成功结构到JSON文件
- [ ] 备份重要实验数据
- [ ] 记录最佳配置和发现
- [ ] 计划下一步实验方向
- [ ] 清理临时文件和日志

## 📋 实验检查清单

开始实验前确认：

- [ ] 虚拟环境已激活
- [ ] 磁盘空间充足（至少1GB）
- [ ] 网络连接稳定
- [ ] 使用了nohup确保不被中断
- [ ] 记录了实验目标和假设
- [ ] 设置了监控脚本
- [ ] 知道如何安全停止实验

---

**祝你的过夜实验成功！明天早上应该会有很多有趣的发现！** 🚀🌅

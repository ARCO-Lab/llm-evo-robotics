# 🧪 实验成功记录系统使用指南

## 📊 系统概述

实验成功记录系统会自动记录MAP-Elites训练过程中所有成功的机器人结构和训练参数，帮助你跟踪哪些配置能够达到目标性能。

### 🎯 主要功能
- ✅ 自动记录所有实验结果（成功和失败）
- 📊 基于fitness阈值判断成功/失败
- 💾 同时保存JSON和CSV格式数据
- 📈 生成详细的统计分析
- 🔍 提供结果查看和分析工具
- 📁 按会话组织实验数据

## 🚀 快速开始

### 1. 启动带记录的实验

```bash
# 启动共享PPO训练（自动启用成功记录）
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared

# 启动传统训练（自动启用成功记录）
python examples/surrogate_model/map_elites/map_elites_trainer.py --train
```

### 2. 查看实验结果

```bash
# 列出所有实验会话
python examples/surrogate_model/map_elites/view_results.py --list

# 查看最新实验结果
python examples/surrogate_model/map_elites/view_results.py

# 查看特定会话结果
python examples/surrogate_model/map_elites/view_results.py 20250915_220326

# 导出成功结构
python examples/surrogate_model/map_elites/view_results.py --export json
python examples/surrogate_model/map_elites/view_results.py --export csv
```

## 📁 文件结构

```
./experiment_results/
├── session_20250915_220326/          # 实验会话目录
│   ├── results.json                  # 所有结果(JSON格式)
│   ├── results.csv                   # 所有结果(CSV格式)
│   ├── successful_results.json       # 仅成功结果(JSON)
│   ├── successful_results.csv        # 仅成功结果(CSV)
│   ├── session_summary.txt           # 会话总结
│   └── analysis_plots.png            # 分析图表(需要matplotlib)
└── session_20250915_230145/          # 另一个会话
    └── ...
```

## ⚙️ 配置选项

### 成功阈值设置

默认成功阈值为 `0.6`，你可以根据需要调整：

```python
# 在map_elites_trainer.py中修改
trainer = MAPElitesEvolutionTrainer(
    # ... 其他参数 ...
    success_threshold=0.7,        # 提高成功标准
    enable_success_logging=True   # 启用记录
)
```

### 适合不同训练时长的阈值建议

- **短时间测试** (500步): `success_threshold=0.4`
- **中等训练** (2000步): `success_threshold=0.6`
- **长时间训练** (5000+步): `success_threshold=0.7`
- **严格标准**: `success_threshold=0.8`

## 📊 记录的数据字段

### 机器人结构信息
- `num_links`: 关节数量
- `link_lengths`: 各段链长列表
- `total_length`: 总长度

### 训练参数
- `lr`: 学习率
- `alpha`: SAC Alpha参数
- `training_steps`: 训练步数
- `buffer_capacity`: 缓冲区容量
- `batch_size`: 批次大小

### 性能指标
- `fitness`: 综合适应度分数
- `success_rate`: 成功率
- `avg_reward`: 平均奖励
- `training_time`: 训练耗时
- `episodes_completed`: 完成的回合数
- `final_distance_to_target`: 最终与目标的距离

### 实验元信息
- `experiment_id`: 实验唯一ID
- `timestamp`: 时间戳
- `generation`: 进化代数
- `parent_id`: 父代ID
- `is_successful`: 是否成功

## 📈 结果分析功能

### 基本统计
```
总实验数: 150
成功实验数: 45
成功率: 30.0%
平均fitness: 0.542
最佳fitness: 0.834
```

### 按关节数分析
```
📊 按关节数统计:
   3关节: 12个 (平均fitness: 0.645, 最高: 0.789)
   4关节: 18个 (平均fitness: 0.712, 最高: 0.834)
   5关节: 15个 (平均fitness: 0.698, 最高: 0.801)
```

### 最佳结果展示
```
🥇 最佳结果:
   实验ID: gen_2_45821
   关节数: 4
   链长: [72.3, 68.1, 59.7, 51.2]
   总长度: 251.3
   Fitness: 0.834
   成功率: 89.2%
   代数: 2
```

## 🎯 长时间实验建议

### 为过夜实验做准备

1. **设置合适的参数**:
```bash
# 启动长时间训练
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared
```

2. **使用nohup确保不被中断**:
```bash
nohup python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared > experiment.log 2>&1 &
```

3. **定期检查进度**:
```bash
# 查看最新结果
python examples/surrogate_model/map_elites/view_results.py

# 查看日志
tail -f experiment.log
```

### 实验监控

你可以在训练过程中实时查看成功情况：

```bash
# 在另一个终端中监控
watch -n 30 "python examples/surrogate_model/map_elites/view_results.py | head -20"
```

## 🔍 故障排除

### 常见问题

1. **没有生成结果文件**
   - 确保训练时间足够长，至少完成一个个体的评估
   - 检查文件权限和磁盘空间

2. **成功率为0**
   - 降低success_threshold阈值
   - 增加training_steps_per_individual

3. **缺少可视化图表**
   ```bash
   pip install matplotlib seaborn pandas
   ```

4. **CSV文件无法打开**
   - 使用UTF-8编码打开
   - 或使用pandas读取：`pd.read_csv('results.csv')`

### 数据恢复

如果需要从之前的实验中恢复数据：

```python
from success_logger import SuccessLogger
import json

# 加载现有会话
with open('./experiment_results/session_20250915_220326/results.json', 'r') as f:
    old_results = json.load(f)

# 重新分析
for result in old_results:
    if result['performance']['fitness'] >= 0.6:
        print(f"成功结构: {result['experiment_id']}")
```

## 📋 实验清单

在开始长时间实验前，确认以下项目：

- [ ] 设置了合适的success_threshold
- [ ] 确认有足够的磁盘空间
- [ ] 设置了合理的训练参数
- [ ] 启用了成功记录 (enable_success_logging=True)
- [ ] 准备了实验监控脚本
- [ ] 记录了实验的目标和假设

## 🎉 实验结束后

1. **查看最终结果**:
```bash
python examples/surrogate_model/map_elites/view_results.py
```

2. **导出成功结构**:
```bash
python examples/surrogate_model/map_elites/view_results.py --export json
```

3. **备份重要数据**:
```bash
cp -r ./experiment_results/session_YYYYMMDD_HHMMSS ./backup/
```

4. **分析和总结**:
   - 记录最佳配置
   - 分析成功模式
   - 计划后续实验

---

## 📞 获取帮助

如果遇到问题，可以：

1. 查看会话总结文件：`session_summary.txt`
2. 检查实验日志输出
3. 使用 `--list` 参数查看所有可用会话
4. 检查JSON文件的完整性

**祝你的实验取得成功！** 🚀

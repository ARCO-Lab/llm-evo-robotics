# 📁 模型和实验结果文件位置总结

## 🎯 核心位置概览

### 1. 📊 **实验成功记录** (最重要 - 你的成功结果都在这里)
```
./experiment_results/
├── session_20250915_220326/     # 第一次测试会话
│   ├── results.json             # 所有实验结果 (JSON格式)
│   ├── results.csv              # 所有实验结果 (CSV格式，Excel可打开)
│   ├── successful_results.json  # 仅成功的实验 (JSON格式)
│   ├── successful_results.csv   # 仅成功的实验 (CSV格式)
│   └── session_summary.txt      # 会话总结报告
├── session_20250915_220543/     # 最新的实验会话
│   ├── results.json             # 最新实验的所有结果
│   ├── results.csv              # 最新实验结果 (CSV格式)
│   ├── successful_results.json  # 最新实验的成功结果
│   └── successful_results.csv   # 最新实验成功结果 (CSV格式)
└── session_YYYYMMDD_HHMMSS/     # 每次新训练都会创建新会话
```

**这是最重要的位置！** 所有成功的机器人结构、训练参数和性能指标都记录在这里。

### 2. 🤖 **共享PPO模型** (训练过程中的模型)
```
./map_elites_shared_ppo_results/
└── shared_ppo_model.pth         # 共享PPO模型文件 (如果训练时间足够长会生成)
```

**注意**: 这个文件只有在训练进行一段时间后才会生成，需要收集足够的经验。

### 3. 🧬 **个体训练结果** (每个机器人的详细训练数据)
```
./map_elites_experiments/
├── individual_43768/            # 最新训练的个体
│   ├── best_models/             # 该个体的最佳模型
│   │   ├── best_ppo_model_step_XXX.pth
│   │   └── latest_best_model.pth
│   ├── logs.txt                 # 训练日志
│   └── training_logs/           # 详细训练记录
├── individual_43770/            # 另一个个体
├── individual_43772/            # 另一个个体
└── individual_XXXXX/            # 每个训练的个体都有独立目录
```

### 4. 🗂️ **MAP-Elites存档** (进化历史)
```
./map_elites_archive/
├── archive_gen_10.pkl           # 第10代的存档
├── archive_gen_15.pkl           # 第15代的存档
└── archive_gen_XX.pkl           # 每5代保存一次
```

### 5. 📝 **训练日志** (如果使用nohup)
```
./overnight_experiment.log       # 主训练日志 (如果使用了nohup)
./experiment_pid.txt             # 进程ID文件
```

## 🔍 **如何查看你的成功结果**

### 快速查看命令:
```bash
# 1. 查看所有实验会话
python examples/surrogate_model/map_elites/view_results.py --list

# 2. 查看最新结果
python examples/surrogate_model/map_elites/view_results.py

# 3. 查看特定会话 (替换为实际的会话ID)
python examples/surrogate_model/map_elites/view_results.py 20250915_220543

# 4. 导出成功结构
python examples/surrogate_model/map_elites/view_results.py --export json
```

### 直接查看文件:
```bash
# 查看成功结果的CSV文件 (可用Excel打开)
cat ./experiment_results/session_20250915_220543/successful_results.csv

# 查看会话总结
cat ./experiment_results/session_20250915_220543/session_summary.txt
```

## 📊 **数据格式说明**

### JSON文件结构:
```json
{
  "experiment_id": "gen_0_43626",
  "timestamp": "2025-09-15T22:05:43",
  "robot_structure": {
    "num_links": 3,
    "link_lengths": [92.2, 64.3, 89.9],
    "total_length": 246.4
  },
  "training_params": {
    "lr": 0.00031736842,
    "alpha": 0.8366654428,
    "training_steps": 500
  },
  "performance": {
    "fitness": 0.756,
    "success_rate": 0.85,
    "avg_reward": 12.3
  },
  "is_successful": true
}
```

### CSV文件列:
- `experiment_id`: 实验唯一标识
- `num_links`: 关节数
- `link_lengths`: 链长数组
- `total_length`: 总长度
- `lr`: 学习率
- `alpha`: SAC Alpha参数
- `fitness`: 适应度分数
- `success_rate`: 成功率
- `is_successful`: 是否成功

## 🎯 **重要提醒**

### 最关键的文件:
1. **`./experiment_results/session_YYYYMMDD_HHMMSS/successful_results.json`** - 包含所有成功的机器人结构
2. **`./experiment_results/session_YYYYMMDD_HHMMSS/session_summary.txt`** - 实验总结报告

### 备份建议:
```bash
# 备份重要的实验结果
cp -r ./experiment_results ./backup_experiment_results_$(date +%Y%m%d)
```

### 查找特定类型的成功结果:
```bash
# 查找所有3关节的成功结构
grep "num_links.*3" ./experiment_results/session_*/successful_results.csv

# 查找fitness > 0.7的结果
awk -F',' '$13 > 0.7 {print $0}' ./experiment_results/session_*/successful_results.csv
```

## 📈 **实时监控文件**

在训练过程中，这些文件会实时更新:
- `./experiment_results/session_YYYYMMDD_HHMMSS/results.json` - 实时添加新结果
- `./overnight_experiment.log` - 实时训练日志
- `./map_elites_experiments/individual_XXXXX/` - 新的个体目录

## 🔧 **文件管理建议**

### 定期清理:
```bash
# 删除旧的个体训练文件 (保留最近的)
find ./map_elites_experiments -name "individual_*" -mtime +7 -exec rm -rf {} \;

# 压缩旧的实验结果
tar -czf old_experiments_$(date +%Y%m%d).tar.gz ./experiment_results/session_202509*
```

### 快速统计:
```bash
# 统计总实验数
find ./experiment_results -name "results.json" -exec wc -l {} \; | awk '{sum+=$1} END {print "总实验数:", sum-NF}'

# 统计成功实验数  
find ./experiment_results -name "successful_results.json" -exec wc -l {} \; | awk '{sum+=$1} END {print "成功实验数:", sum-NF}'
```

---

**总结**: 你的所有成功结果都保存在 `./experiment_results/session_*/` 目录中，使用 `view_results.py` 工具可以方便地查看和分析这些结果！

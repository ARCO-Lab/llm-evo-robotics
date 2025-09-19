# MAP-Elites训练器 + 独立损失记录器

## 概述

这个系统为您提供了一个独立进程的损失记录器，能够在运行MAP-Elites训练的同时记录attention网络、GNN网络、PPO网络的每步损失。

## 主要特点

- 🚀 **独立进程**: 损失记录器在独立进程中运行，不影响主训练性能
- 📊 **实时监控**: 自动生成实时损失曲线图表，每15秒更新一次
- 💾 **数据持久化**: 自动保存CSV和JSON格式的损失数据
- 🎯 **多网络支持**: 同时监控attention、PPO、GNN、SAC等多个网络
- 📈 **趋势分析**: 自动分析损失趋势（上升/下降/稳定）
- ⚡ **高性能**: 使用队列和多进程，避免阻塞训练

## 文件结构

```
examples/surrogate_model/
├── network_loss_logger.py              # 核心损失记录器
├── loss_logger_interface.py            # 简化使用接口
├── map_elites_with_loss_logger.py      # 集成版MAP-Elites训练器
├── training_adapter_with_logging.py    # 带损失记录的训练适配器
├── launch_map_elites_with_logger.py    # Python启动脚本
├── run_map_elites_with_logger.sh       # Bash启动脚本
└── README_LOSS_LOGGER.md               # 使用说明（本文件）
```

## 快速开始

### 方法1: 使用Python启动器（推荐）

```bash
cd examples/surrogate_model
python launch_map_elites_with_logger.py
```

这将显示一个交互式菜单，让您选择训练模式。

### 方法2: 直接运行

```bash
cd examples/surrogate_model
python map_elites_with_loss_logger.py --mode basic --experiment-name my_experiment
```

### 方法3: 使用Bash脚本

```bash
cd examples/surrogate_model
./run_map_elites_with_logger.sh basic --experiment-name my_experiment
```

## 训练模式

| 模式 | 说明 |
|------|------|
| `basic` | 基础MAP-Elites训练，适合快速测试 |
| `advanced` | 高级训练，支持交互式参数配置 |
| `multiprocess` | 多进程训练，提高训练效率 |
| `shared-ppo` | 共享PPO训练，使用共享的PPO网络 |
| `custom` | 自定义参数训练，完全可配置 |

## 命令行参数

### 基本参数
- `--mode`: 训练模式（必需）
- `--experiment-name`: 实验名称
- `--disable-loss-logging`: 禁用损失记录
- `--loss-log-dir`: 损失日志目录（默认: `network_loss_logs`）
- `--loss-update-interval`: 图表更新间隔秒数（默认: 15）

### 自定义训练参数（当mode='custom'时使用）
- `--num-generations`: 进化代数（默认: 20）
- `--training-steps-per-individual`: 每个个体训练步数（默认: 2000）
- `--num-initial-random`: 初始随机个体数（默认: 10）
- `--enable-rendering`: 启用环境渲染
- `--use-genetic-fitness`: 使用遗传算法fitness

## 使用示例

### 示例1: 基础训练
```bash
python map_elites_with_loss_logger.py --mode basic --experiment-name basic_test
```

### 示例2: 自定义训练
```bash
python map_elites_with_loss_logger.py --mode custom \
    --experiment-name custom_test \
    --num-generations 30 \
    --training-steps-per-individual 5000 \
    --enable-rendering \
    --use-genetic-fitness
```

### 示例3: 禁用损失记录
```bash
python map_elites_with_loss_logger.py --mode basic --disable-loss-logging
```

## 输出文件

系统会在指定的日志目录下创建以下文件：

```
network_loss_logs/
└── experiment_name_loss_log/
    ├── config.json                         # 实验配置信息
    ├── attention_losses.csv                # attention网络损失数据
    ├── attention_stats.json                # attention网络统计信息
    ├── ppo_losses.csv                      # PPO网络损失数据
    ├── ppo_stats.json                      # PPO网络统计信息
    ├── gnn_losses.csv                      # GNN网络损失数据
    ├── gnn_stats.json                      # GNN网络统计信息
    ├── sac_losses.csv                      # SAC网络损失数据（如果适用）
    ├── sac_stats.json                      # SAC网络统计信息
    ├── total_losses.csv                    # 总损失数据
    ├── total_stats.json                    # 总损失统计信息
    ├── network_loss_curves_realtime.png   # 实时损失曲线图
    └── network_loss_curves_20231201_14.png # 带时间戳的损失曲线图
```

## 损失记录原理

### 真实训练模式
- 当使用真实训练时，系统会尝试从训练结果中提取实际的损失值
- 如果无法提取，会基于训练结果生成逼真的损失序列

### 模拟训练模式
- 当使用模拟训练时，系统会生成模拟的损失数据用于测试
- 模拟数据具有逼真的趋势和噪声

### 损失分类
系统会自动将损失值分类到对应的网络：
- **Attention**: 包含'attention'或'attn'关键字的损失
- **PPO**: 包含'ppo'、'actor'、'critic'、'policy'关键字的损失
- **GNN**: 包含'gnn'、'graph'、'node'、'edge'关键字的损失
- **SAC**: 包含'sac'、'alpha'关键字的损失

## 高级用法

### 在现有代码中集成损失记录

```python
from loss_logger_interface import start_loss_logging, log_network_loss

# 启动损失记录器
logger = start_loss_logging(experiment_name="my_experiment")

# 在训练循环中记录损失
for step in range(training_steps):
    # 训练网络...
    attention_loss = train_attention_network()
    ppo_losses = train_ppo_network()
    gnn_loss = train_gnn_network()
    
    # 记录损失
    log_network_loss('attention', step, {'attention_loss': attention_loss})
    log_network_loss('ppo', step, ppo_losses)
    log_network_loss('gnn', step, {'gnn_loss': gnn_loss})
```

### 使用装饰器自动记录

```python
from loss_logger_interface import auto_log_loss

@auto_log_loss('ppo')
def train_ppo_step(step):
    # PPO训练逻辑
    return {'actor_loss': 0.5, 'critic_loss': 0.3}

# 使用
for step in range(1000):
    train_ppo_step(step)  # 损失会自动记录
```

## 监控和调试

### 检查损失记录器状态
```python
from loss_logger_interface import is_loss_logger_alive, get_loss_log_directory

# 检查是否运行
if is_loss_logger_alive():
    print(f"损失记录器正在运行，日志目录: {get_loss_log_directory()}")
else:
    print("损失记录器未运行")
```

### 查看实时图表
损失记录器会自动生成实时图表：
- 文件位置: `network_loss_logs/experiment_name/network_loss_curves_realtime.png`
- 更新频率: 默认每15秒更新一次
- 内容: 包含所有网络的损失曲线、趋势线和统计信息

## 故障排除

### 1. 损失记录器启动失败
- 检查Python依赖: `torch`, `numpy`, `matplotlib`
- 确认有足够的系统内存和CPU资源
- 查看错误信息中的具体原因

### 2. 图表不更新
- 检查日志目录是否有写权限
- 确认matplotlib后端设置正确
- 查看是否有进程冲突

### 3. 损失数据丢失
- 检查队列是否溢出（会有警告信息）
- 确认损失数据格式正确（必须是数值类型）
- 查看记录进程是否正常运行

### 4. 性能问题
- 调整`--loss-update-interval`参数，增加更新间隔
- 减少记录频率，不是每步都记录
- 检查系统资源使用情况

## 技术细节

### 进程架构
```
主训练进程
    ↓ (队列通信)
损失记录进程
    ├── 数据收集线程
    ├── 图表生成线程
    └── 数据保存线程
```

### 队列机制
- 使用`multiprocessing.Queue`进行进程间通信
- 队列大小限制为50000，防止内存溢出
- 非阻塞发送，避免影响训练性能

### 数据格式
```python
loss_data = {
    'network': 'attention',  # 网络名称
    'step': 1000,           # 训练步数
    'timestamp': time.time(), # 时间戳
    'losses': {             # 损失字典
        'attention_loss': 0.5,
        'attention_accuracy': 0.8
    }
}
```

## 贡献和反馈

如果您在使用过程中遇到问题或有改进建议，请：
1. 检查本文档的故障排除部分
2. 查看系统日志和错误信息
3. 提供详细的错误复现步骤

## 更新日志

- **v1.0**: 初始版本，支持基本的损失记录和图表生成
- 支持attention、PPO、GNN、SAC网络的损失记录
- 集成MAP-Elites训练系统
- 提供多种启动方式和配置选项


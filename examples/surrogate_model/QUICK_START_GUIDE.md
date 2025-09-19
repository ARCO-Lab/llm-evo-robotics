# 快速开始指南 - MAP-Elites训练器 + 损失记录器

## 🚀 快速启动

### 方法1: 使用Python启动器（推荐）

```bash
cd examples/surrogate_model
python launch_map_elites_with_logger.py
```

选择训练模式后系统将自动启动训练和损失记录。

### 方法2: 直接命令行启动

```bash
cd examples/surrogate_model
python map_elites_with_loss_logger.py --mode basic --experiment-name my_test
```

### 方法3: 仅测试损失记录器

```bash
cd examples/surrogate_model
python test_loss_logger.py
```

## 📊 输出结果

训练完成后，您可以在以下位置找到结果：

### 损失记录文件
```
network_loss_logs/
└── your_experiment_name_loss_log/
    ├── network_loss_curves_realtime.png  # 实时损失曲线图
    ├── attention_losses.csv              # attention网络损失数据
    ├── ppo_losses.csv                    # PPO网络损失数据
    ├── gnn_losses.csv                    # GNN网络损失数据
    └── config.json                       # 实验配置
```

### MAP-Elites训练结果
```
map_elites_results/  # 或您指定的保存目录
├── best_individuals/
├── archive_data/
└── training_logs/
```

## 🎯 主要特点

- ✅ **独立进程记录**: 损失记录器在独立进程中运行，不影响训练性能
- ✅ **实时图表**: 每15秒自动更新损失曲线图
- ✅ **多网络监控**: 同时记录attention、PPO、GNN、SAC等网络的损失
- ✅ **数据持久化**: 自动保存CSV格式的损失数据
- ✅ **趋势分析**: 自动分析损失趋势（上升/下降/稳定）

## 🔧 自定义配置

如果需要自定义参数，使用以下命令：

```bash
python map_elites_with_loss_logger.py --mode custom \
    --experiment-name custom_experiment \
    --num-generations 50 \
    --training-steps-per-individual 10000 \
    --loss-update-interval 30 \
    --enable-rendering
```

## 📈 监控训练进度

1. **实时图表**: 查看 `network_loss_logs/experiment_name/network_loss_curves_realtime.png`
2. **CSV数据**: 使用Excel或Python分析 `*_losses.csv` 文件
3. **控制台输出**: 观察训练过程中的损失打印信息

## ⚠️ 注意事项

1. 确保已安装依赖: `torch`, `numpy`, `matplotlib`
2. 训练过程中不要删除日志目录
3. 如果出现中文字体警告，可以忽略（不影响功能）
4. 按Ctrl+C可以安全中断训练

## 🐛 故障排除

如果遇到问题：

1. **检查依赖**: 运行 `python test_loss_logger.py` 测试系统
2. **查看日志**: 检查控制台输出的错误信息
3. **清理进程**: 如果进程卡住，重启终端
4. **权限问题**: 确保对日志目录有写权限

## 📞 获取帮助

运行以下命令获取完整参数列表：

```bash
python map_elites_with_loss_logger.py --help
```

## 🎉 开始使用

现在您可以开始使用这个系统了！建议先运行测试确保一切正常：

```bash
python test_loss_logger.py
```

然后启动您的第一个训练：

```bash
python launch_map_elites_with_logger.py
```

祝您训练愉快！🚀


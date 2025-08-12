# 训练损失监控系统使用指南

本指南详细介绍如何使用我们为你创建的训练损失记录和监控系统。

## 🎯 **系统功能概览**

### 主要功能
- ✅ **实时损失记录** - 自动记录各网络的训练损失
- ✅ **可视化图表** - 生成损失曲线和趋势分析
- ✅ **异常监控** - 实时监控异常情况并发出警报
- ✅ **多格式导出** - 支持CSV、JSON、PNG等多种格式
- ✅ **统计分析** - 提供详细的训练统计报告

### 脚本文件说明
```
training_logger.py          # 核心训练监控系统
enhanced_train.py          # 集成监控的增强训练脚本
analyze_existing_logs.py    # 分析现有日志的工具脚本
```

---

## 🚀 **方法1: 使用增强版训练脚本**

### 基本使用
```bash
# 进入虚拟环境
source venv/bin/activate

# 运行增强版训练（Reacher2D）
cd examples/surrogate_model
python enhanced_train.py --test-reacher2d
```

### 自定义配置
```bash
# 自定义训练参数
python enhanced_train.py --test-reacher2d \
    --num-processes 4 \
    --lr 1e-3 \
    --seed 123
```

### 输出结果
运行后会在以下位置生成完整的训练记录：
```
./trained_models/reacher2d/enhanced_test/[timestamp]/
├── training_logs/
│   └── reacher2d_sac_[timestamp]/
│       ├── training_log.csv         # CSV格式的详细记录
│       ├── training_log.json        # JSON格式的数据
│       ├── training_curves_*.png    # 损失曲线图
│       ├── training_report.txt      # 训练总结报告
│       └── config.json              # 实验配置
└── best_models/                     # 保存的模型文件
    ├── final_model_step_*.pth
    └── checkpoint_step_*.pth
```

---

## 🔧 **方法2: 手动集成到现有训练脚本**

### 步骤1: 导入监控系统
```python
from training_logger import TrainingLogger, RealTimeMonitor

# 初始化logger
experiment_name = f"my_experiment_{time.strftime('%Y%m%d_%H%M%S')}"
logger = TrainingLogger(
    log_dir="training_logs",
    experiment_name=experiment_name
)

# 设置监控阈值
monitor = RealTimeMonitor(logger, alert_thresholds={
    'critic_loss': {'max': 50.0, 'nan_check': True},
    'actor_loss': {'max': 10.0, 'nan_check': True},
    'alpha_loss': {'max': 5.0, 'nan_check': True},
})
```

### 步骤2: 在训练循环中记录损失
```python
for step in range(num_steps):
    # ... 训练代码 ...
    
    # 更新网络并获取损失
    if should_update:
        metrics = sac.update()  # 返回损失字典
        
        if metrics:
            # 添加额外信息
            enhanced_metrics = metrics.copy()
            enhanced_metrics.update({
                'step': step,
                'buffer_size': len(sac.memory),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # 记录到监控系统
            logger.log_step(step, enhanced_metrics, episode=episode_count)
            alerts = monitor.check_alerts(step, enhanced_metrics)
            
            # 定期打印统计和生成图表
            if step % 100 == 0:
                logger.print_current_stats(step, detailed=True)
            
            if step % 1000 == 0:
                logger.plot_losses(recent_steps=2000, show=False)
```

### 步骤3: 训练结束时生成报告
```python
# 训练完成后
logger.generate_report()
logger.plot_losses(show=False)
print(f"📊 完整训练日志: {logger.experiment_dir}")
```

---

## 📊 **方法3: 分析现有训练日志**

如果你已经有训练输出的文本日志，可以用分析脚本提取损失信息：

```bash
# 分析现有的训练日志文件
python analyze_existing_logs.py \
    --log-file /path/to/training_output.log \
    --output-dir ./analysis_results

# 示例：分析上次训练的日志
python analyze_existing_logs.py \
    --log-file ../../trained_models/reacher2d/test/*/logs.txt \
    --output-dir ./log_analysis
```

---

## 📈 **生成的图表和报告**

### 1. 训练曲线图 (`training_curves_*.png`)
包含6个子图：
- **SAC Losses**: Critic Loss, Actor Loss, Alpha Loss
- **Q Values**: Q1 Mean, Q2 Mean  
- **Policy Metrics**: Alpha, Entropy Term, Q Term
- **Episode Metrics**: Episode Reward, Episode Length
- **Loss Trends**: 移动平均趋势线
- **Learning Progress**: 早期vs后期性能对比

### 2. 训练报告 (`training_report.txt`)
包含：
- 实验基本信息（步数、时间、速度）
- 各网络损失的详细统计
- Replay Buffer使用情况
- 学习稳定性分析

### 3. 数据文件
- **CSV格式** (`training_log.csv`): 适合Excel分析
- **JSON格式** (`training_log.json`): 适合程序化处理
- **Pickle格式** (`training_logger.pkl`): 可重新加载完整Logger对象

---

## 🚨 **实时监控和警报**

### 监控指标
系统会自动监控以下异常情况：
- **NaN/Inf值**: 检测数值异常
- **损失爆炸**: 超出设定阈值
- **趋势异常**: 持续上升的损失

### 警报示例
```
🚨 Step 1500 监控警报:
   ⚠️ critic_loss 超出最大阈值: 15.2341 > 10.0
   📈 actor_loss 持续上升趋势，当前均值: 2.3456
```

### 自定义阈值
```python
custom_thresholds = {
    'critic_loss': {'max': 20.0, 'min': 0.0, 'nan_check': True},
    'actor_loss': {'max': 5.0, 'nan_check': True},
    'alpha': {'min': 0.01, 'max': 1.0},
}
monitor = RealTimeMonitor(logger, alert_thresholds=custom_thresholds)
```

---

## 💡 **最佳实践建议**

### 1. 实验命名
```python
# 使用描述性的实验名称
experiment_name = f"reacher2d_sac_lr{args.lr}_bs{batch_size}_{timestamp}"
```

### 2. 记录频率
```python
# 损失记录：每次更新都记录
logger.log_step(step, metrics)

# 图表生成：适中频率避免性能影响
if step % 1000 == 0:
    logger.plot_losses(recent_steps=2000, show=False)

# 统计打印：查看训练进度
if step % 100 == 0:
    logger.print_current_stats(step)
```

### 3. 存储管理
```python
# 定期保存避免数据丢失
logger.save_logs()  # 每100步自动保存

# 大型实验建议设置更大的保存间隔
logger.save_interval = 500  # 每500步保存一次
```

---

## 🔍 **故障排除**

### 常见问题

**Q: 图表无法显示**
```bash
# 设置非交互式后端
export MPLBACKEND=Agg
python your_script.py
```

**Q: 内存使用过多**
```python
# 减少recent_losses保存的数据量
logger.max_recent_size = 50  # 默认100
```

**Q: Pickle序列化失败**
```python
# 避免使用lambda函数，已在当前版本修复
```

### 调试模式
```python
# 启用详细日志
logger.debug_mode = True
logger.print_current_stats(step, detailed=True)
```

---

## 📝 **示例配置文件**

创建 `training_config.py`:
```python
# 训练监控配置
LOGGING_CONFIG = {
    'log_dir': 'training_logs',
    'save_interval': 100,
    'plot_interval': 1000,
    'alert_thresholds': {
        'critic_loss': {'max': 50.0, 'nan_check': True},
        'actor_loss': {'max': 10.0, 'nan_check': True},
        'alpha_loss': {'max': 5.0, 'nan_check': True},
    }
}
```

---

## 🎊 **总结**

你现在有了一个完整的训练损失监控系统！

**快速开始**:
```bash
# 1. 直接使用增强版训练脚本
python enhanced_train.py --test-reacher2d

# 2. 或者分析现有日志
python analyze_existing_logs.py --log-file your_log.txt

# 3. 查看结果
ls training_logs/  # 查看生成的所有记录
```

这个系统将帮助你：
- 📊 **可视化训练过程** - 清晰看到损失变化
- 🚨 **及时发现问题** - 自动监控异常情况  
- 📈 **分析训练效果** - 详细的统计报告
- 💾 **保存训练记录** - 多格式数据导出

享受你的训练监控之旅！🚀 
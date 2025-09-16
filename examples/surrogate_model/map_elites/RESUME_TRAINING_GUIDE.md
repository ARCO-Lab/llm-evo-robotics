# 共享PPO模型恢复训练指南

## 概述

现在MAP-Elites训练器支持从之前中断的训练中恢复，继续使用已保存的共享PPO模型。这个功能特别适合长时间的实验，可以避免因为意外中断而丢失训练进度。

## 功能特性

### 🔄 自动模型保存
- 共享PPO模型每次更新后都会自动保存
- 每5次更新会创建一个带时间戳的备份文件
- 模型保存位置：`./map_elites_shared_ppo_results/shared_ppo_model.pth`

### 🚀 智能恢复
- 自动检测已保存的模型文件
- 恢复模型参数、优化器状态和训练计数
- 从上次中断的地方继续训练

### 📊 训练状态追踪
- 记录模型更新次数
- 保存完整的训练状态（包括优化器状态）
- 支持跨会话的训练进度恢复

## 使用方法

### 1. 开始新训练
```bash
# 开始新的共享PPO训练
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared
```

### 2. 恢复中断的训练
```bash
# 从已保存的模型继续训练
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume
```

### 3. 组合使用参数
```bash
# 恢复训练 + 禁用可视化
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume --no-render

# 恢复训练 + 静默模式
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume --silent

# 恢复训练 + 禁用可视化 + 静默模式
python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume --no-render --silent
```

## 工作流程示例

### 场景：长时间实验被意外中断

1. **启动初始训练**
   ```bash
   python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared
   ```
   
   输出：
   ```
   🚀 MAP-Elites + 共享PPO训练
   🆕 创建新的PPO模型
   ✅ 共享PPO训练器启动成功
   ```

2. **训练过程中意外中断**（Ctrl+C 或系统重启）
   ```
   ⚠️ 训练被用户中断
   💾 模型已保存 (更新次数: 15) -> ./map_elites_shared_ppo_results/shared_ppo_model.pth
   ```

3. **恢复训练**
   ```bash
   python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume
   ```
   
   输出：
   ```
   🔧 检测到 --resume 参数，将尝试加载已保存的模型继续训练
   🔄 将从已保存的模型继续训练: ./map_elites_shared_ppo_results/shared_ppo_model.pth
   🔄 正在加载已保存的模型...
   ✅ 成功加载模型 - 已完成 15 次更新
   ```

## 模型文件结构

### 主模型文件
```
./map_elites_shared_ppo_results/shared_ppo_model.pth
```

包含：
- `actor`: Actor网络参数
- `critic`: Critic网络参数  
- `actor_optimizer`: Actor优化器状态
- `critic_optimizer`: Critic优化器状态
- `update_count`: 更新次数计数

### 备份文件
```
./map_elites_shared_ppo_results/shared_ppo_model_backup_20250916_143025.pth
./map_elites_shared_ppo_results/shared_ppo_model_backup_20250916_143127.pth
...
```

每5次更新自动创建，用于额外的安全保障。

## 安全特性

### 🛡️ 智能检测
- 系统会自动检测是否存在已保存的模型
- 如果不使用 `--resume` 参数，会警告用户可能覆盖已有模型
- 使用 `--resume` 但模型不存在时，会自动开始新训练

### 💾 多重备份
- 主模型文件每次更新都保存
- 定期创建带时间戳的备份文件
- 确保即使主文件损坏也能恢复

### 🔧 错误处理
- 如果模型加载失败，自动回退到新模型初始化
- 详细的错误信息和状态报告
- 优雅的降级处理

## 注意事项

### ⚠️ 重要提醒
1. **模型兼容性**：确保恢复训练时使用相同的模型配置
2. **路径一致性**：保存目录需要与之前训练时一致
3. **参数匹配**：观察维度和动作维度需要匹配

### 💡 最佳实践
1. **定期备份**：手动复制重要的模型文件到安全位置
2. **监控日志**：关注模型更新次数和训练状态
3. **参数记录**：记录训练配置以便后续恢复

## 示例输出

### 新训练开始
```
🚀 MAP-Elites + 共享PPO训练
🆕 创建新的PPO模型
📊 检测到观察维度: 14, 动作维度: 3
✅ 简化PPO模型初始化完成
💾 模型已保存 (更新次数: 1) -> ./map_elites_shared_ppo_results/shared_ppo_model.pth
```

### 恢复训练
```
🔧 检测到 --resume 参数，将尝试加载已保存的模型继续训练
🔄 将从已保存的模型继续训练: ./map_elites_shared_ppo_results/shared_ppo_model.pth
🔍 发现已保存的模型: ./map_elites_shared_ppo_results/shared_ppo_model.pth
🔄 正在加载已保存的模型...
✅ 成功加载模型 - 已完成 15 次更新
📊 模型参数:
   观察维度: 14
   动作维度: 3
   隐藏层维度: 256
```

## 故障排除

### 问题：模型加载失败
**症状**：`⚠️ 加载模型失败，使用随机初始化`
**解决**：
1. 检查模型文件是否存在且未损坏
2. 确认模型配置参数一致
3. 使用备份文件恢复

### 问题：找不到模型文件
**症状**：`⚠️ 使用了 --resume 参数但未找到模型文件`
**解决**：
1. 确认保存目录路径正确
2. 检查是否有权限访问文件
3. 确认之前的训练确实保存了模型

这个恢复功能让你的长时间实验更加可靠，再也不用担心训练中断导致的进度丢失！🎉

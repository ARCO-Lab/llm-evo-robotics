# 渲染控制完整指南

## 🎨 启用渲染的方法

### 方法1：使用参数控制（推荐）
```bash
# 启用渲染
python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_with_render \
    --mode basic \
    --training-steps 500 \
    --num-generations 1 \
    --individuals-per-generation 2 \
    --enable-rendering

# 禁用渲染（默认）
python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_no_render \
    --mode basic \
    --training-steps 500 \
    --num-generations 1 \
    --individuals-per-generation 2
```

### 方法2：使用环境变量控制
```bash
# 强制启用渲染
FORCE_RENDER=1 python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_force_render \
    --mode basic \
    --training-steps 500

# 强制禁用渲染
FORCE_NO_RENDER=1 python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_force_no_render \
    --mode basic \
    --training-steps 500
```

### 方法3：多进程渲染模式
```bash
# 多进程 + 渲染（每个进程一个窗口）
python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_multiprocess_render \
    --mode multiprocess \
    --training-steps 400 \
    --num-generations 1 \
    --individuals-per-generation 4 \
    --enable-rendering
```

## 🚫 禁用渲染的方法

### 方法1：默认模式（推荐）
```bash
# 不使用--enable-rendering参数即可
python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_no_render \
    --mode basic \
    --training-steps 500
```

### 方法2：环境变量强制禁用（最可靠）
```bash
FORCE_NO_RENDER=1 python enhanced_multi_network_extractor_backup.py \
    --experiment-name test_no_render \
    --mode basic \
    --training-steps 500
```

## 🔧 渲染控制优先级

1. **环境变量 FORCE_NO_RENDER=1** - 最高优先级，强制禁用
2. **环境变量 FORCE_RENDER=1** - 强制启用
3. **--enable-rendering 参数** - 启用渲染
4. **默认行为** - 禁用渲染

## ⚠️ 注意事项

1. **多进程渲染**：会同时打开多个pygame窗口
2. **渲染性能**：启用渲染会显著降低训练速度
3. **窗口关闭**：可以按ESC键或关闭窗口来退出
4. **后台运行**：如果不需要观察，建议使用无渲染模式

## 🐛 故障排除

如果遇到渲染相关问题：

1. **进程卡住**：使用 `pkill -f enhanced` 杀掉所有进程
2. **窗口无响应**：按ESC键或使用 `FORCE_NO_RENDER=1` 强制禁用
3. **多窗口问题**：减少 `--individuals-per-generation` 数量

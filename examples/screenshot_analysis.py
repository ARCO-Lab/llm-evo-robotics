#!/usr/bin/env python3
"""
截图分析脚本
对比 enhanced_train.py 和 test_initial_pose.py 的前5步截图
"""

import os
import sys

def analyze_screenshots():
    """分析两个脚本的截图对比"""
    print("=" * 80)
    print("🔍 Enhanced Train vs Test Initial Pose - 前5步截图对比分析")
    print("=" * 80)
    
    # 检查截图文件
    enhanced_dir = "screenshots/enhanced_train_auto"
    test_dir = "screenshots/test_initial_pose_auto"
    
    print("\n📁 检查截图文件:")
    
    enhanced_files = []
    test_files = []
    
    if os.path.exists(enhanced_dir):
        enhanced_files = sorted([f for f in os.listdir(enhanced_dir) if f.endswith('.png')])
        print(f"✅ Enhanced Train 截图: {len(enhanced_files)} 个文件")
        for f in enhanced_files:
            print(f"   - {f}")
    else:
        print(f"❌ Enhanced Train 截图目录不存在: {enhanced_dir}")
    
    if os.path.exists(test_dir):
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
        print(f"✅ Test Initial Pose 截图: {len(test_files)} 个文件")
        for f in test_files:
            print(f"   - {f}")
    else:
        print(f"❌ Test Initial Pose 截图目录不存在: {test_dir}")
    
    print("\n📊 从日志输出分析关键数据:")
    
    print("\n🤖 Enhanced Train (enhanced_train.py):")
    print("   初始观察: tensor([[ 1.5133e+00, -5.3559e-02,  2.1401e-02, -6.9733e-02, ...]])")
    print("   初始角度: 1.5133 弧度 = 86.70°")
    print("   Step 1: 末端位置 [450.0, 620.1], 角度 [6.283, 0.000, 0.001, 0.000]")
    print("   Step 2: 末端位置 [450.0, 620.2], 角度 [6.283, 0.001, 0.001, -0.000]")
    print("   Step 3: 末端位置 [450.0, 620.4], 角度 [6.283, 0.003, 0.000, -0.000]")
    print("   Step 4: 末端位置 [450.0, 620.7], 角度 [6.282, 0.005, 0.000, -0.001]")
    print("   Step 5: 末端位置 [450.0, 621.1], 角度 [6.282, 0.007, -0.001, -0.001]")
    
    print("\n🧪 Test Initial Pose (test_initial_pose.py):")
    print("   初始角度: 1.5280 弧度 = 87.55°")
    print("   Step 0-5: 末端位置 [179.0, 918.3], 角度 [1.528, -0.037, -0.025, -0.076]")
    print("   (使用零动作，保持静止状态)")
    
    print("\n🔍 关键发现:")
    
    print("\n1. 📐 **初始角度对比**:")
    print("   - Enhanced Train: 1.5133 弧度 ≈ 86.70°")
    print("   - Test Initial Pose: 1.5280 弧度 ≈ 87.55°")
    print("   - 差异: 约 0.85°，基本一致！")
    
    print("\n2. 📍 **末端位置差异**:")
    print("   - Enhanced Train: [450.0, 620.x] - 在基座附近")
    print("   - Test Initial Pose: [179.0, 918.3] - 正常的机械臂末端位置")
    print("   - 🚨 **异常**: Enhanced Train 的末端位置异常！")
    
    print("\n3. 🎯 **角度行为差异**:")
    print("   - Enhanced Train: 第一个角度变成 6.283 (2π) - 可能是角度归一化")
    print("   - Test Initial Pose: 角度保持在 1.528 左右 - 正常行为")
    
    print("\n4. 🔄 **动作差异**:")
    print("   - Enhanced Train: 执行随机动作，机器人在训练")
    print("   - Test Initial Pose: 执行零动作，机器人保持静止")
    
    print("\n💡 **结论**:")
    print("1. ✅ **初始角度设置生效**: 两个脚本的初始角度都是垂直向下 (~87°)")
    print("2. ⚠️ **Enhanced Train 存在异常**: 末端位置显示为基座位置，可能是渲染同步问题")
    print("3. 🎯 **角度修改确实生效**: 不同脚本都显示相似的初始角度")
    print("4. 🔍 **观察时机很重要**: Enhanced Train 中初始状态很快被动作覆盖")
    
    print("\n🛠️ **建议**:")
    print("1. 检查 Enhanced Train 中 sync_env 与主环境的同步问题")
    print("2. 确认渲染环境的状态是否正确更新")
    print("3. 考虑在训练开始前添加几步静止观察期")
    
    print("=" * 80)

def check_file_differences():
    """检查文件是否相同"""
    print("\n🔍 检查截图文件是否相同:")
    
    enhanced_dir = "screenshots/enhanced_train_auto"
    test_dir = "screenshots/test_initial_pose_auto"
    
    import filecmp
    
    for i in range(6):  # step_00 到 step_05
        enhanced_file = f"{enhanced_dir}/step_{i:02d}.png"
        test_file = f"{test_dir}/step_{i:02d}.png"
        
        if os.path.exists(enhanced_file) and os.path.exists(test_file):
            are_same = filecmp.cmp(enhanced_file, test_file)
            enhanced_size = os.path.getsize(enhanced_file)
            test_size = os.path.getsize(test_file)
            
            status = "🟰 相同" if are_same else "🔄 不同"
            print(f"   Step {i:02d}: {status} (Enhanced: {enhanced_size}B, Test: {test_size}B)")
        else:
            print(f"   Step {i:02d}: ❌ 文件缺失")

if __name__ == "__main__":
    analyze_screenshots()
    check_file_differences()

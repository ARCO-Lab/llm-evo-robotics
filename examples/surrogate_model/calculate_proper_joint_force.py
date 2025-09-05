#!/usr/bin/env python3
"""
计算Reacher机器人的合理Joint约束力
"""

import math

def calculate_joint_forces():
    """计算合理的Joint约束力"""
    
    print("🔧 计算Reacher机器人的合理Joint约束力")
    print("=" * 50)
    
    # Reacher参数
    link_lengths = [80, 60, 50, 40]  # mm
    density = 0.1  # kg/mm
    gravity = 98.1  # m/s²
    max_torque = 100  # N·m (Motor最大扭矩)
    
    print(f"📏 Link长度: {link_lengths}")
    print(f"📦 密度: {density}")
    print(f"🌍 重力: {gravity}")
    print(f"⚡ 最大Motor扭矩: {max_torque}")
    
    # 计算每个Link的质量
    masses = []
    total_mass = 0
    for i, length in enumerate(link_lengths):
        mass = density * length * 5  # 与环境中的计算一致
        masses.append(mass)
        total_mass += mass
        print(f"   Link{i}: 长度={length}, 质量={mass:.2f}")
    
    print(f"🔢 总质量: {total_mass:.2f}")
    
    # 计算重力产生的力
    gravity_force = total_mass * gravity
    print(f"🌍 总重力: {gravity_force:.2f} N")
    
    # 计算Motor产生的最大拉力
    # 扭矩转换为线性力: F = T / r (其中r是力臂长度)
    motor_forces = []
    for i, length in enumerate(link_lengths):
        # 最坏情况：Motor扭矩全部转换为Joint拉力
        arm_length = length / 1000  # 转换为米
        motor_force = max_torque / arm_length if arm_length > 0 else 0
        motor_forces.append(motor_force)
        print(f"   Link{i} Motor力: {motor_force:.2f} N")
    
    max_motor_force = max(motor_forces)
    print(f"⚡ 最大Motor力: {max_motor_force:.2f} N")
    
    # 计算碰撞冲量 (估算)
    # 假设以5m/s速度碰撞，0.01s内停止
    collision_velocity = 5.0  # m/s
    collision_time = 0.01  # s
    collision_impulse = total_mass * collision_velocity / collision_time
    print(f"💥 估算碰撞冲量: {collision_impulse:.2f} N")
    
    # 计算总的最大外力
    total_external_force = gravity_force + max_motor_force + collision_impulse
    print(f"🔢 总外力: {total_external_force:.2f} N")
    
    # 计算推荐的Joint约束力 (添加安全系数)
    safety_factors = [2, 5, 10, 20]
    print(f"\n💡 推荐Joint约束力 (不同安全系数):")
    
    recommendations = {}
    for factor in safety_factors:
        recommended_force = total_external_force * factor
        recommendations[factor] = recommended_force
        print(f"   安全系数 {factor}x: {recommended_force:.0f} N")
    
    # 当前设置检查
    current_joint_force = 1000000  # 当前设置
    current_safety_factor = current_joint_force / total_external_force
    print(f"\n📊 当前设置分析:")
    print(f"   当前Joint约束力: {current_joint_force:,} N")
    print(f"   实际安全系数: {current_safety_factor:.1f}x")
    
    if current_safety_factor >= 10:
        print(f"   ✅ 约束力足够 (安全系数 >= 10x)")
    elif current_safety_factor >= 5:
        print(f"   ⚠️ 约束力偏低 (安全系数 5-10x)")
    else:
        print(f"   ❌ 约束力不足 (安全系数 < 5x)")
    
    return recommendations

if __name__ == "__main__":
    calculate_joint_forces()


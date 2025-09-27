#!/usr/bin/env python3
"""
Reacher 环境工厂
提供统一接口来创建不同版本的 Reacher 环境
"""

import os
import sys
from typing import Optional, Union

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

class ReacherEnvFactory:
    """
    Reacher 环境工厂类
    支持创建不同版本的 Reacher 环境
    """
    
    SUPPORTED_VERSIONS = {
        'original': 'Original Reacher2DEnv (自定义物理引擎)',
        'mujoco': 'MuJoCo Reacher Adapter (MuJoCo 物理引擎)',
        'auto': '自动选择最佳版本'
    }
    
    @staticmethod
    def create_env(version='auto', **kwargs):
        """
        创建 Reacher 环境
        
        Args:
            version (str): 环境版本 ('original', 'mujoco', 'auto')
            **kwargs: 传递给环境构造函数的参数
            
        Returns:
            Reacher 环境实例
        """
        
        print(f"🏭 Reacher 环境工厂 - 创建 {version} 版本环境")
        
        if version == 'auto':
            version = ReacherEnvFactory._auto_select_version()
            print(f"   自动选择版本: {version}")
        
        if version == 'original':
            return ReacherEnvFactory._create_original_env(**kwargs)
        elif version == 'mujoco':
            return ReacherEnvFactory._create_mujoco_env(**kwargs)
        else:
            raise ValueError(f"不支持的环境版本: {version}. 支持的版本: {list(ReacherEnvFactory.SUPPORTED_VERSIONS.keys())}")
    
    @staticmethod
    def _auto_select_version():
        """自动选择最佳环境版本"""
        # 检查 MuJoCo 是否可用
        try:
            import gymnasium as gym
            import mujoco
            # 尝试创建 MuJoCo 环境
            test_env = gym.make('Reacher-v5')
            test_env.close()
            print("   ✅ MuJoCo 环境可用，选择 MuJoCo 版本")
            return 'mujoco'
        except Exception as e:
            print(f"   ⚠️ MuJoCo 环境不可用 ({e})，回退到原始版本")
            return 'original'
    
    @staticmethod
    def _create_original_env(**kwargs):
        """创建原始 Reacher2DEnv"""
        try:
            # 动态导入原始环境
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "reacher2d_env", 
                os.path.join(os.path.dirname(__file__), "reacher2d_env.py")
            )
            reacher_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reacher_module)
            Reacher2DEnv = reacher_module.Reacher2DEnv
            
            print("   ✅ 创建原始 Reacher2DEnv")
            return Reacher2DEnv(**kwargs)
            
        except Exception as e:
            print(f"   ❌ 创建原始环境失败: {e}")
            raise
    
    @staticmethod
    def _create_mujoco_env(**kwargs):
        """创建 MuJoCo Reacher 适配器"""
        try:
            # 动态导入 MuJoCo 适配器
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "mujoco_reacher_adapter", 
                os.path.join(os.path.dirname(__file__), "mujoco_reacher_adapter.py")
            )
            adapter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter_module)
            MuJoCoReacherAdapter = adapter_module.MuJoCoReacherAdapter
            
            print("   ✅ 创建 MuJoCo Reacher 适配器")
            return MuJoCoReacherAdapter(**kwargs)
            
        except Exception as e:
            print(f"   ❌ 创建 MuJoCo 环境失败: {e}")
            print("   🔄 回退到原始环境")
            return ReacherEnvFactory._create_original_env(**kwargs)
    
    @staticmethod
    def list_versions():
        """列出所有支持的环境版本"""
        print("🔍 支持的 Reacher 环境版本:")
        for version, description in ReacherEnvFactory.SUPPORTED_VERSIONS.items():
            print(f"   {version}: {description}")
    
    @staticmethod
    def compare_versions():
        """对比不同版本的环境"""
        print("📊 环境版本对比:")
        print("=" * 60)
        
        # 尝试创建两个版本进行对比
        try:
            print("🔍 测试原始环境...")
            original_env = ReacherEnvFactory._create_original_env(render_mode=None)
            print(f"   观察空间: {original_env.observation_space}")
            print(f"   动作空间: {original_env.action_space}")
            original_env.close()
            original_available = True
        except Exception as e:
            print(f"   ❌ 原始环境不可用: {e}")
            original_available = False
        
        try:
            print("🔍 测试 MuJoCo 环境...")
            mujoco_env = ReacherEnvFactory._create_mujoco_env(render_mode=None)
            print(f"   观察空间: {mujoco_env.observation_space}")
            print(f"   动作空间: {mujoco_env.action_space}")
            mujoco_env.close()
            mujoco_available = True
        except Exception as e:
            print(f"   ❌ MuJoCo 环境不可用: {e}")
            mujoco_available = False
        
        print("\n📋 可用性总结:")
        print(f"   原始环境: {'✅ 可用' if original_available else '❌ 不可用'}")
        print(f"   MuJoCo 环境: {'✅ 可用' if mujoco_available else '❌ 不可用'}")
        
        if mujoco_available:
            print("   🎯 推荐使用 MuJoCo 版本（更好的物理仿真）")
        elif original_available:
            print("   🎯 推荐使用原始版本（稳定可靠）")
        else:
            print("   ⚠️ 没有可用的环境版本")

def create_reacher_env(version='auto', **kwargs):
    """
    便捷函数：创建 Reacher 环境
    
    Args:
        version (str): 环境版本 ('original', 'mujoco', 'auto')
        **kwargs: 传递给环境构造函数的参数
        
    Returns:
        Reacher 环境实例
    """
    return ReacherEnvFactory.create_env(version=version, **kwargs)

# 为了向后兼容，提供别名
def Reacher2DEnv(**kwargs):
    """向后兼容的 Reacher2DEnv 构造函数"""
    return create_reacher_env(version='auto', **kwargs)

if __name__ == "__main__":
    print("🎯 Reacher 环境工厂测试")
    print("=" * 60)
    
    # 列出版本
    ReacherEnvFactory.list_versions()
    print()
    
    # 对比版本
    ReacherEnvFactory.compare_versions()
    print()
    
    # 测试自动创建
    print("🚀 测试自动环境创建...")
    try:
        env = create_reacher_env(version='auto', render_mode=None)
        print("✅ 环境创建成功")
        
        # 简单测试
        obs = env.reset()
        print(f"   重置观察: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"   步进结果: obs={obs.shape}, reward={reward:.3f}, done={done}")
        
        env.close()
        print("✅ 环境测试完成")
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()


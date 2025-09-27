#!/usr/bin/env python3
"""
SB3 环境包装器
确保环境与 Stable Baselines3 完全兼容
"""

import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
from typing import Optional, Tuple, Dict, Any, Union
import warnings


class SB3CompatibleWrapper(Env):
    """
    SB3 兼容包装器
    确保环境与 Stable Baselines3 的期望完全一致
    """
    
    def __init__(self, env):
        """
        初始化包装器
        
        Args:
            env: 要包装的环境
        """
        super().__init__()
        
        self.env = env
        
        # 复制空间定义
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # 复制其他重要属性
        if hasattr(env, 'spec'):
            self.spec = env.spec
        else:
            self.spec = None
            
        # 确保有 render_mode 属性
        if hasattr(env, 'render_mode'):
            self.render_mode = env.render_mode
        else:
            self.render_mode = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境 - 确保返回 (obs, info) 格式
        """
        # 设置种子
        if seed is not None:
            self.seed(seed)
        
        # 调用底层环境的 reset
        result = self.env.reset()
        
        # 处理返回值 - 可能是 obs 或 (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            # 新 API 格式: (obs, info)
            obs, info = result
        else:
            # 旧 API 格式: 只有 obs
            obs = result
            info = {}
        
        # 确保观察值是正确的格式
        try:
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            else:
                obs = obs.astype(np.float32)
            
            # 强制确保是 float32 类型
            obs = np.array(obs, dtype=np.float32)
            
            # 确保是一维数组
            if obs.ndim > 1:
                obs = obs.flatten()
        except Exception as e:
            print(f"❌ 观察值转换失败: {e}")
            print(f"   观察值类型: {type(obs)}")
            print(f"   观察值内容: {obs}")
            if hasattr(obs, '__len__'):
                print(f"   观察值长度: {len(obs)}")
                if len(obs) > 0:
                    print(f"   第一个元素类型: {type(obs[0])}")
                    print(f"   第一个元素内容: {obs[0]}")
            raise e
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步 - 确保返回新的 Gymnasium 格式
        """
        # 调用底层环境的 step
        result = self.env.step(action)
        
        if len(result) == 4:
            # 旧格式: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = False
        elif len(result) == 5:
            # 新格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
        else:
            raise ValueError(f"Unexpected step result format: {len(result)} elements")
        
        # 确保观察值是正确的格式
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        else:
            obs = obs.astype(np.float32)
        
        # 强制确保是 float32 类型
        obs = np.array(obs, dtype=np.float32)
        
        # 确保是一维数组
        if obs.ndim > 1:
            obs = obs.flatten()
        
        # 确保奖励是标量
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        else:
            reward = float(reward)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染"""
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None
    
    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]
    
    def __getattr__(self, name):
        """转发其他属性访问到底层环境"""
        return getattr(self.env, name)


def make_sb3_compatible(env):
    """
    创建 SB3 兼容的环境
    
    Args:
        env: 原始环境
        
    Returns:
        SB3 兼容的环境
    """
    if isinstance(env, SB3CompatibleWrapper):
        return env
    
    return SB3CompatibleWrapper(env)


def test_sb3_wrapper():
    """测试 SB3 包装器"""
    print("🧪 测试 SB3 环境包装器")
    
    try:
        # 导入环境
        import sys
        import os
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from reacher_env_factory import create_reacher_env
        
        # 创建原始环境
        env = create_reacher_env(version='mujoco', render_mode=None)
        print(f"✅ 原始环境创建成功")
        
        # 包装环境
        wrapped_env = make_sb3_compatible(env)
        print(f"✅ 环境包装成功")
        
        # 测试 reset
        obs, info = wrapped_env.reset()
        print(f"✅ Reset 测试通过: obs.shape={obs.shape}, info={type(info)}")
        
        # 测试 step
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"✅ Step 测试通过: obs.shape={obs.shape}, reward={reward:.3f}")
        
        wrapped_env.close()
        print(f"✅ SB3 包装器测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_sb3_wrapper()

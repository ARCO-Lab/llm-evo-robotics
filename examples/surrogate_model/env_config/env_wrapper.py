# examples/surrogate_model/env_wrappers.py

import torch
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from a2c_ppo_acktr.envs import VecPyTorch, VecNormalize

import sys
import os

# 添加异步渲染器的导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
async_renderer_path = os.path.join(current_dir, "../../2d_reacher/envs")
sys.path.insert(0, async_renderer_path)

try:
    from async_renderer import AsyncRenderer, StateExtractor
    ASYNC_RENDER_AVAILABLE = True
    print("✅ 异步渲染器可用")
except ImportError as e:
    ASYNC_RENDER_AVAILABLE = False
    print(f"⚠️ 异步渲染器不可用: {e}")


class AsyncRenderableVecEnv:
    """支持异步渲染的向量化环境包装器"""
    
    def __init__(self, vec_env, env_params, render_env_id=0, enable_async_render=True):
        """
        Args:
            vec_env: 基础向量化环境
            env_params: 环境参数字典
            render_env_id: 要渲染的环境ID (0-based)
            enable_async_render: 是否启用异步渲染
        """
        self.vec_env = vec_env
        self.render_env_id = render_env_id
        self.enable_async_render = enable_async_render and ASYNC_RENDER_AVAILABLE
        
        # 🎨 创建异步渲染器
        if self.enable_async_render:
            self.renderer = AsyncRenderer(env_params, max_queue_size=20)
            self.renderer.start()
            print(f"🎨 异步渲染器已启动，渲染环境ID: {render_env_id}")
        else:
            self.renderer = None
            print("⚠️ 异步渲染器未启用")
        
        # 🤖 创建状态同步环境（不渲染，只用于状态跟踪）
        from reacher2d_env import Reacher2DEnv
        sync_env_params = env_params.copy()
        sync_env_params['render_mode'] = None  # 确保不渲染
        
        self.sync_env = Reacher2DEnv(**sync_env_params)
        self.step_count = 0
        
        print(f"🔄 状态同步环境已创建")
        
    def reset(self):
        """重置向量化环境"""
        obs = self.vec_env.reset()
        
        # 重置同步环境
        if self.sync_env:
            self.sync_env.reset()
            self.step_count = 0
            
            # 发送初始渲染状态
            if self.renderer:
                robot_state = StateExtractor.extract_robot_state(
                    self.sync_env, self.step_count
                )
                self.renderer.render_frame(robot_state)
        
        return obs
    
    def step(self, actions):
        """执行向量化环境步进"""
        obs, rewards, dones, infos = self.vec_env.step(actions)
        
        # 🔄 同步渲染环境状态
        if self.sync_env and self.render_env_id < len(actions):
            # 使用指定环境的动作来同步状态
            if hasattr(actions, 'cpu'):  # PyTorch tensor
                render_action = actions[self.render_env_id].cpu().numpy()
            else:  # numpy array
                render_action = actions[self.render_env_id]
            
            # 同步环境步进
            sync_obs, sync_reward, sync_done, sync_info = self.sync_env.step(render_action)
            
            # 📤 异步渲染
            if self.renderer:
                robot_state = StateExtractor.extract_robot_state(
                    self.sync_env, self.step_count
                )
                self.renderer.render_frame(robot_state)
            
            # 如果同步环境结束，重置它并同步episode计数
            if sync_done:
                self.sync_env.reset()
                # 🔧 同步episode计数到训练环境
                if hasattr(self.sync_env, 'current_episode') and hasattr(self.vec_env, 'envs'):
                    for env in self.vec_env.envs:
                        if hasattr(env, 'current_episode'):
                            env.current_episode = self.sync_env.current_episode
                            print(f"🔄 [SYNC] 同步episode计数到训练环境: Episode = {env.current_episode}")
        
        self.step_count += 1
        return obs, rewards, dones, infos
    
    def close(self):
        """关闭环境和渲染器"""
        self.vec_env.close()
        
        if self.sync_env:
            self.sync_env.close()
        
        if self.renderer:
            self.renderer.stop()
            print("🎨 异步渲染器已停止")
    
    def get_render_stats(self):
        """获取渲染统计信息"""
        if self.renderer:
            return self.renderer.get_stats()
        return {}
    
    def __getattr__(self, name):
        """代理其他属性到基础环境"""
        return getattr(self.vec_env, name)


def make_reacher2d_env(env_params, seed, rank, log_dir=None, allow_early_resets=True):
    """
    创建单个 Reacher2D 环境的 thunk 函数
    参考 RoboGrammar 的 make_env 模式
    """
    def _thunk():
        from reacher2d_env import Reacher2DEnv
        
        # 创建环境
        # env = Reacher2DEnv(
        #     num_links=env_params.get('num_links', 5),
        #     link_lengths=env_params.get('link_lengths', [80, 50, 30, 20, 10]),
        #     render_mode=env_params.get('render_mode', "human"),  # 训练环境不渲染
        #     config_path=env_params.get('config_path', None)
        # )

        env = Reacher2DEnv(
            num_links=env_params['num_links'],        # 🔧 移除默认值
            link_lengths=env_params['link_lengths'],  # 🔧 移除默认值
            render_mode=env_params.get('render_mode', "human"),
            config_path=env_params.get('config_path', None)
        )
        
        # 设置种子（每个进程不同的种子）
        env.seed(seed + rank)
        
        # 添加监控（如果需要）
        if log_dir is not None:
            from baselines import bench
            import os
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets
            )
        
        return env
    
    return _thunk

def make_reacher2d_vec_envs(env_params, seed, num_processes, gamma, log_dir, device, allow_early_resets):
    """
    创建 Reacher2D 的向量化环境
    完全模仿 RoboGrammar 的 make_vec_envs
    """
    # 创建环境 thunk 列表
    envs = [
        make_reacher2d_env(env_params, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]
    
    # 选择向量化方式
    if len(envs) > 1:
        # 多进程：使用共享内存向量化环境
        envs = ShmemVecEnv(envs, context='fork')
    else:
        # 单进程：使用虚拟向量化环境
        envs = DummyVecEnv(envs)
    
    # 添加归一化（如果需要）
    # 🔧 临时禁用VecNormalize来测试reward问题
    if False and len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)
    
    # 转换为 PyTorch 张量
    envs = VecPyTorch(envs, device)
    
    return envs

class Reacher2DEnvWrapper:
    """
    Reacher2D 环境包装器
    让 Reacher2DEnv 兼容 gymnasium 接口（如果需要）
    """
    def __init__(self, base_env):
        self.base_env = base_env
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
    
    def reset(self, **kwargs):
        """兼容不同的 reset 接口"""
        result = self.base_env.reset(**kwargs)
        if isinstance(result, tuple):
            return result  # gymnasium 格式：(obs, info)
        else:
            return result, {}  # gym 格式：obs -> (obs, {})
    
    def step(self, action):
        """统一 step 接口"""
        return self.base_env.step(action)
    
    def seed(self, seed):
        """设置随机种子"""
        if hasattr(self.base_env, 'seed'):
            return self.base_env.seed(seed)
        else:
            # 如果环境没有 seed 方法，可以设置其他随机数生成器
            np.random.seed(seed)
            return [seed]
    
    def close(self):
        """关闭环境"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()
    
    def __getattr__(self, name):
        """代理其他属性到基础环境"""
        return getattr(self.base_env, name)
    

# # 在 env_wrapper.py 文件末尾添加测试代码

# if __name__ == "__main__":
#     print("🧪 开始测试环境包装器...")
    
#     import sys
#     import os
#     import torch
#     import numpy as np
    
#     # 添加必要的路径
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     base_dir = os.path.join(current_dir, '../../../')
#     sys.path.append(base_dir)
#     sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
    
#     print(f"📁 当前目录: {current_dir}")
#     print(f"📁 基础目录: {base_dir}")
    
#     # 测试1: 基本导入
#     print("\n=== 测试1: 基本导入 ===")
#     try:
#         from reacher2d_env import Reacher2DEnv
#         print("✅ 成功导入 Reacher2DEnv")
#     except Exception as e:
#         print(f"❌ 导入 Reacher2DEnv 失败: {e}")
#         sys.exit(1)
    
#     # 测试2: 单个环境创建和包装
#     print("\n=== 测试2: 环境创建和包装 ===")

#     abs_config_path = "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"

#     print(f"🔍 使用绝对路径: {abs_config_path}")
#     print(f"🔍 文件存在: {os.path.exists(abs_config_path)}")
    
#     try:
#         # 创建基础环境
#         base_env = Reacher2DEnv(
#             num_links=3,
#             link_lengths=[80, 50, 30],
#             render_mode="human",
#             config_path = abs_config_path
#         )
#         print("✅ 基础环境创建成功")
        
#         # 包装环境
#         wrapped_env = Reacher2DEnvWrapper(base_env)
#         print("✅ 环境包装成功")
        
#         print(f"   动作空间: {wrapped_env.action_space}")
#         print(f"   观察空间: {wrapped_env.observation_space}")
#         print(f"   关节数量: {wrapped_env.action_space.shape[0]}")
        
#         # 测试重置
#         obs, info = wrapped_env.reset()
#         print(f"✅ 重置成功，观察维度: {obs.shape}, 信息类型: {type(info)}")
        
#         # 测试步进
#         action = np.random.uniform(-1, 1, wrapped_env.action_space.shape[0])
#         obs, reward, terminated, truncated, info = wrapped_env.step(action)
#         print(f"✅ 步进成功")
#         print(f"   观察维度: {obs.shape}")
#         print(f"   奖励: {reward:.3f}")
#         print(f"   结束状态: terminated={terminated}, truncated={truncated}")
        
#         # 测试种子设置
#         seed_result = wrapped_env.seed(123)
#         print(f"✅ 种子设置成功: {seed_result}")
        
#         wrapped_env.close()
#         print("✅ 环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ 环境包装测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试3: Thunk 创建
#     print("\n=== 测试3: Thunk 创建 ===")
#     try:
#         env_params = {
#             'num_links': 5,
#             'link_lengths': [80, 50, 30, 20, 10],
#             'config_path': None,
#             'render_mode': "human"
#         }
        
#         # 创建 thunk
#         thunk = make_reacher2d_env(env_params, seed=42, rank=0)
#         print("✅ Thunk 创建成功")
        
#         # 从 thunk 创建环境
#         env = thunk()
#         print("✅ 从 thunk 创建环境成功")
        
#         # 测试基本功能
#         obs, info = env.reset()
#         print(f"✅ Thunk 环境重置成功，观察维度: {obs.shape}")
        
#         action = np.random.uniform(-2, 2, env.action_space.shape[0])
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"✅ Thunk 环境步进成功，奖励: {reward:.3f}")
        
#         env.close()
#         print("✅ Thunk 环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ Thunk 测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试4: 向量化环境创建（单进程）
#     print("\n=== 测试4: 向量化环境（单进程）===")
#     try:
#         # 检查依赖
#         try:
#             from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
#             from a2c_ppo_acktr.envs import VecPyTorch
#             vec_env_available = True
#             print("✅ 向量化环境依赖可用")
#         except ImportError as e:
#             print(f"⚠️  向量化环境依赖不可用: {e}")
#             vec_env_available = False
        
#         if vec_env_available:
#             env_params = {
#                 'num_links': 3,
#                 'link_lengths': [80, 50, 30],
#                 'config_path': abs_config_path,
#                 'render_mode': "human"
#             }
            
#             device = torch.device('cpu')
            
#             # 创建向量化环境
#             envs = make_reacher2d_vec_envs(
#                 env_params=env_params,
#                 seed=42,
#                 num_processes=1,  # 单进程测试
#                 gamma=0.99,
#                 log_dir=None,
#                 device=device,
#                 allow_early_resets=False
#             )
#             print("✅ 单进程向量化环境创建成功")
            
#             print(f"   环境数量: {envs.num_envs}")
#             print(f"   动作空间: {envs.action_space}")
#             print(f"   观察空间: {envs.observation_space}")
            
#             # 测试向量化操作
#             obs = envs.reset()
#             print(f"✅ 向量化重置成功，观察形状: {obs.shape}")
            
#             actions = torch.randn(1, envs.action_space.shape[0])
#             obs, rewards, dones, infos = envs.step(actions)
#             print(f"✅ 向量化步进成功")
#             print(f"   观察形状: {obs.shape}")
#             print(f"   奖励形状: {rewards.shape}")
#             print(f"   完成形状: {dones.shape}")
#             print(f"   奖励值: {rewards[0].item():.3f}")
            
#             # 测试多步执行
#             print("🔄 执行10步测试...")
#             print("🎥 创建渲染环境...")
#             render_env = Reacher2DEnv(
#                 num_links=3,
#                 link_lengths=[80, 50, 30],
#                 render_mode="human",
#                 config_path=abs_config_path  # 使用相同的配置
#             )
#             render_obs = render_env.reset()

#             for i in range(5000):
#                 actions = torch.randn(1, envs.action_space.shape[0])
#                 obs, rewards, dones, infos = envs.step(actions)
#                 render_env.render()
#                 if i % 20 == 0:
#                     print(f"   步骤 {i}: 奖励 {rewards[0].item():.3f}")

#                     # 处理pygame事件（避免窗口无响应）
#                 import pygame
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         break
                
#                 if terminated or truncated:
#                     obs, info = wrapped_env.reset()
#                 if dones:
#                     render_obs = render_env.reset()
                    
#                 # 添加小延迟以便观察
#                 import time
#                 time.sleep(0.05)  # 20 FPS

#             print("✅ 渲染测试完成")
            
#             envs.close()
#             print("✅ 向量化环境关闭成功")
#         else:
#             print("⏭️  跳过向量化环境测试")
        
#     except Exception as e:
#         print(f"❌ 向量化环境测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试5: 多进程环境（如果支持）
#     print("\n=== 测试5: 多进程环境 ===")
#     try:
#         if vec_env_available:
#             print("🚀 尝试创建2进程环境...")
            
#             env_params = {
#                 'num_links': 3,
#                 'link_lengths': [80, 50, 30],
#                 'config_path': "configs/reacher_with_zigzag_obstacles.yaml",
#                 'render_mode': "human"
#             }
            
#             device = torch.device('cpu')
            
#             # 创建多进程向量化环境
#             envs = make_reacher2d_vec_envs(
#                 env_params=env_params,
#                 seed=42,
#                 num_processes=5,  # 2进程测试
#                 gamma=0.99,
#                 log_dir=None,
#                 device=device,
#                 allow_early_resets=False
#             )
#             print("✅ 多进程向量化环境创建成功")
            
#             # 测试并行操作
#             obs = envs.reset()
#             print(f"✅ 多进程重置成功，观察形状: {obs.shape}")
            
#             # 并行执行几步
#             import time
#             start_time = time.time()
#             for i in range(5000):
#                 actions = torch.randn(5, envs.action_space.shape[0]) * 0.5
#                 obs, rewards, dones, infos = envs.step(actions)

#                 print(f"   步骤 {i}: 奖励 {rewards.numpy()}, obs: {obs.shape}")
            
#             elapsed = time.time() - start_time
#             print(f"✅ 5步并行执行完成，用时: {elapsed:.3f}秒")
            
#             envs.close()
#             print("✅ 多进程环境关闭成功")
#         else:
#             print("⏭️  跳过多进程环境测试（依赖不可用）")
        
#     except Exception as e:
#         print(f"❌ 多进程环境测试失败: {e}")
#         print("   这可能是正常的，取决于系统支持情况")
#         # 不打印完整错误，因为多进程可能在某些系统上不工作
    
#     # 测试总结
#     print("\n🎉 测试完成！")
#     print("=" * 50)
#     print("如果看到这里，说明基本功能都正常工作。")
#     print("如果有任何 ❌ 错误，请检查相应的依赖和路径设置。")
#     print("⚠️  警告通常是可以忽略的（表示某些高级功能不可用）。")





#     print("🔄 执行可视化测试...")

#     # 创建单独的可视化环境
#     vis_env = Reacher2DEnv(
#         num_links=4,  # 减少关节数，运动更明显
#         link_lengths=[80, 80, 80, 60],  # 增加连杆长度
#         render_mode="human",
#         config_path=abs_config_path
#     )

#     # 调整物理参数让运动更明显
#     vis_env.max_torque = 100.0  # 大幅增加最大扭矩
#     vis_env.space.damping = 0.85  # 减少阻尼让运动更自由
#     vis_env.dt = 1/25.0  # 稍大的时间步长

#     # 减少body质量让它们更容易运动
#     for body in vis_env.bodies:
#         body.mass = body.mass * 0.4  # 减少质量
#         body.moment = body.moment * 0.4  # 减少转动惯量

#     print("🎥 开始大幅度可视化（按ESC退出）...")
#     vis_obs = vis_env.reset()

#     import pygame
#     import time

#     try:
#         for i in range(2000):
#             # 处理事件
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     break
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         break
            
#             # 生成整个时间段都大幅度的动作
#             t = i * 0.01  # 统一的时间频率
            
#             # 多层叠加的大幅度正弦波，创造复杂而明显的运动
#             action = np.array([
#                 # 第一关节：主要大幅摆动
#                 60 * np.sin(t) + 25 * np.sin(t * 2.3),
                
#                 # 第二关节：跟随摆动，稍有延迟
#                 50 * np.sin(t + 0.5) + 20 * np.sin(t * 1.7),
                
#                 # 第三关节：更快的振荡
#                 45 * np.sin(t * 1.2 + 1) + 15 * np.sin(t * 3.1),
                
#                 # 第四关节：高频小幅叠加
#                 40 * np.sin(t * 0.8 + 1.5) + 12 * np.sin(t * 4.2)
#             ])
            
#             # 确保动作在范围内
#             action = np.clip(action, -80, 80)  # 扩大扭矩限制
            
#             # 执行动作
#             vis_obs, vis_reward, vis_done, vis_info = vis_env.step(action)
            
#             # 渲染
#             vis_env.render()
            
#             if i % 50 == 0:
#                 end_pos = vis_env._get_end_effector_position()
#                 print(f"   步骤 {i}: 奖励 {vis_reward:.3f}, 末端位置 ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
            
#             if vis_done:
#                 vis_obs = vis_env.reset()
            
#             time.sleep(0.06)  # 稍慢的帧率让运动更明显

#     except KeyboardInterrupt:
#         print("\n⏹️ 用户停止可视化")

#     finally:
#         vis_env.close()
#         print("✅ 可视化环境关闭")


# # 🎯 便捷函数：自动选择最佳环境创建方式
# def make_smart_reacher2d_vec_envs(env_params, seed, num_processes, gamma, log_dir, device,
#                                 allow_early_resets=True, prefer_async_render=True):
#     """
#     智能创建Reacher2D向量化环境
#     - 单进程 + 渲染模式 → 普通向量化环境
#     - 多进程 + 渲染模式 → 异步渲染向量化环境
#     - 无渲染模式 → 普通向量化环境
#     """
    
#     needs_render = env_params.get('render_mode') == 'human'
#     is_multiprocess = num_processes > 1
    
#     if needs_render and is_multiprocess and prefer_async_render and ASYNC_RENDER_AVAILABLE:
#         print("🚀 使用异步渲染多进程环境")
#         return make_async_renderable_vec_envs(
#             env_params=env_params,
#             seed=seed,
#             num_processes=num_processes,
#             gamma=gamma,
#             log_dir=log_dir,
#             device=device,
#             allow_early_resets=allow_early_resets,
#             render_env_id=0,
#             enable_async_render=True
#         )
#     else:
#         if needs_render and is_multiprocess:
#             print("⚠️ 多进程环境强制禁用渲染（异步渲染不可用）")
#             env_params = env_params.copy()
#             env_params['render_mode'] = None
        
#         print("🏃 使用标准向量化环境")
#         return make_reacher2d_vec_envs(
#             env_params=env_params,
#             seed=seed,
#             num_processes=num_processes,
#             gamma=gamma,
#             log_dir=log_dir,
#             device=device,
#             allow_early_resets=allow_early_resets
#         )


# # 🧪 在现有测试代码中添加异步渲染测试
# def test_async_render_multiprocess():
#     """测试异步渲染多进程环境"""
#     print("\n=== 测试6: 异步渲染多进程环境 ===")
    
#     if not ASYNC_RENDER_AVAILABLE:
#         print("⏭️ 跳过异步渲染测试（依赖不可用）")
#         return
    
#     try:
#         # 环境参数
#         env_params = {
#             'num_links': 4,
#             'link_lengths': [80, 80, 80, 60],
#             'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
#             'render_mode': 'human'  # 这会触发异步渲染
#         }
        
#         device = torch.device('cpu')
        
#         # 创建异步渲染多进程环境
#         envs = make_async_renderable_vec_envs(
#             env_params=env_params,
#             seed=42,
#             num_processes=4,  # 4进程
#             gamma=0.99,
#             log_dir=None,
#             device=device,
#             allow_early_resets=False,
#             render_env_id=0,
#             enable_async_render=True
#         )
#         print("✅ 异步渲染多进程环境创建成功")
        
#         # 测试并行训练 + 异步渲染
#         obs = envs.reset()
#         print(f"✅ 环境重置成功，观察形状: {obs.shape}")
        
#         print("🎮 开始异步渲染训练（在渲染窗口中按ESC退出）...")
        
#         import time
#         start_time = time.time()
        
#         for step in range(2000):  # 运行2000步
#             # 4个进程的随机动作
#             actions = torch.randn(4, envs.action_space.shape[0]) * 2.0
            
#             # 并行步进
#             obs, rewards, dones, infos = envs.step(actions)
            
#             # 定期打印统计
#             if step % 200 == 0:
#                 render_stats = envs.get_render_stats()
#                 elapsed = time.time() - start_time
#                 print(f"   步骤 {step}: 训练FPS={step/elapsed:.1f}, "
#                       f"渲染FPS={render_stats.get('fps', 0):.1f}, "
#                       f"丢帧率={render_stats.get('drop_rate', 0):.1f}%")
            
#             # 检查渲染进程是否还活着
#             if hasattr(envs, 'renderer') and envs.renderer:
#                 if not envs.renderer.render_process.is_alive():
#                     print("🔴 渲染进程已退出，停止训练")
#                     break
            
#             # 小延迟避免过度占用CPU
#             if step % 50 == 0:
#                 time.sleep(0.01)
        
#         elapsed = time.time() - start_time
#         final_stats = envs.get_render_stats()
        
#         print(f"✅ 异步渲染测试完成")
#         print(f"   总用时: {elapsed:.2f}秒")
#         print(f"   平均训练FPS: {2000/elapsed:.1f}")
#         print(f"   平均渲染FPS: {final_stats.get('fps', 0):.1f}")
#         print(f"   总丢帧率: {final_stats.get('drop_rate', 0):.1f}%")
        
#         envs.close()
#         print("✅ 异步渲染环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ 异步渲染测试失败: {e}")
#         import traceback
#         traceback.print_exc()


# # 修改 env_wrapper.py 中的现有测试代码
# if __name__ == "__main__":
#     print("🧪 开始测试环境包装器...")
    
#     import sys
#     import os
#     import torch
#     import numpy as np
    
#     # 添加必要的路径
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     base_dir = os.path.join(current_dir, '../../../')
#     sys.path.append(base_dir)
#     sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
    
#     print(f"📁 当前目录: {current_dir}")
#     print(f"📁 基础目录: {base_dir}")
    
#     # 测试1: 基本导入
#     print("\n=== 测试1: 基本导入 ===")
#     try:
#         from reacher2d_env import Reacher2DEnv
#         print("✅ 成功导入 Reacher2DEnv")
#     except Exception as e:
#         print(f"❌ 导入 Reacher2DEnv 失败: {e}")
#         sys.exit(1)
    
#     # 测试2: 单个环境创建和包装
#     print("\n=== 测试2: 环境创建和包装 ===")

#     abs_config_path = "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"

#     print(f"🔍 使用绝对路径: {abs_config_path}")
#     print(f"🔍 文件存在: {os.path.exists(abs_config_path)}")
    
#     try:
#         # 创建基础环境
#         base_env = Reacher2DEnv(
#             num_links=3,
#             link_lengths=[80, 50, 30],
#             render_mode="human",
#             config_path = abs_config_path
#         )
#         print("✅ 基础环境创建成功")
        
#         # 包装环境
#         wrapped_env = Reacher2DEnvWrapper(base_env)
#         print("✅ 环境包装成功")
        
#         print(f"   动作空间: {wrapped_env.action_space}")
#         print(f"   观察空间: {wrapped_env.observation_space}")
#         print(f"   关节数量: {wrapped_env.action_space.shape[0]}")
        
#         # 测试重置
#         obs, info = wrapped_env.reset()
#         print(f"✅ 重置成功，观察维度: {obs.shape}, 信息类型: {type(info)}")
        
#         # 测试步进
#         action = np.random.uniform(-1, 1, wrapped_env.action_space.shape[0])
#         obs, reward, terminated, truncated, info = wrapped_env.step(action)
#         print(f"✅ 步进成功")
#         print(f"   观察维度: {obs.shape}")
#         print(f"   奖励: {reward:.3f}")
#         print(f"   结束状态: terminated={terminated}, truncated={truncated}")
        
#         # 测试种子设置
#         seed_result = wrapped_env.seed(123)
#         print(f"✅ 种子设置成功: {seed_result}")
        
#         wrapped_env.close()
#         print("✅ 环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ 环境包装测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试3: Thunk 创建
#     print("\n=== 测试3: Thunk 创建 ===")
#     try:
#         env_params = {
#             'num_links': 5,
#             'link_lengths': [80, 50, 30, 20, 10],
#             'config_path': None,
#             'render_mode': "human"
#         }
        
#         # 创建 thunk
#         thunk = make_reacher2d_env(env_params, seed=42, rank=0)
#         print("✅ Thunk 创建成功")
        
#         # 从 thunk 创建环境
#         env = thunk()
#         print("✅ 从 thunk 创建环境成功")
        
#         # 测试基本功能
#         obs, info = env.reset()
#         print(f"✅ Thunk 环境重置成功，观察维度: {obs.shape}")
        
#         action = np.random.uniform(-2, 2, env.action_space.shape[0])
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"✅ Thunk 环境步进成功，奖励: {reward:.3f}")
        
#         env.close()
#         print("✅ Thunk 环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ Thunk 测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试4: 向量化环境创建（单进程）
#     print("\n=== 测试4: 向量化环境（单进程）===")
#     try:
#         # 检查依赖
#         try:
#             from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
#             from a2c_ppo_acktr.envs import VecPyTorch
#             vec_env_available = True
#             print("✅ 向量化环境依赖可用")
#         except ImportError as e:
#             print(f"⚠️  向量化环境依赖不可用: {e}")
#             vec_env_available = False
        
#         if vec_env_available:
#             env_params = {
#                 'num_links': 3,
#                 'link_lengths': [80, 50, 30],
#                 'config_path': abs_config_path,
#                 'render_mode': "human"
#             }
            
#             device = torch.device('cpu')
            
#             # 创建向量化环境
#             envs = make_reacher2d_vec_envs(
#                 env_params=env_params,
#                 seed=42,
#                 num_processes=1,  # 单进程测试
#                 gamma=0.99,
#                 log_dir=None,
#                 device=device,
#                 allow_early_resets=False
#             )
#             print("✅ 单进程向量化环境创建成功")
            
#             print(f"   环境数量: {envs.num_envs}")
#             print(f"   动作空间: {envs.action_space}")
#             print(f"   观察空间: {envs.observation_space}")
            
#             # 测试向量化操作
#             obs = envs.reset()
#             print(f"✅ 向量化重置成功，观察形状: {obs.shape}")
            
#             actions = torch.randn(1, envs.action_space.shape[0])
#             obs, rewards, dones, infos = envs.step(actions)
#             print(f"✅ 向量化步进成功")
#             print(f"   观察形状: {obs.shape}")
#             print(f"   奖励形状: {rewards.shape}")
#             print(f"   完成形状: {dones.shape}")
#             print(f"   奖励值: {rewards[0].item():.3f}")
            
#             # 测试多步执行
#             print("🔄 执行10步测试...")
#             print("🎥 创建渲染环境...")
#             render_env = Reacher2DEnv(
#                 num_links=3,
#                 link_lengths=[80, 50, 30],
#                 render_mode="human",
#                 config_path=abs_config_path  # 使用相同的配置
#             )
#             render_obs = render_env.reset()

#             for i in range(5000):
#                 actions = torch.randn(1, envs.action_space.shape[0])
#                 obs, rewards, dones, infos = envs.step(actions)
#                 render_env.render()
#                 if i % 20 == 0:
#                     print(f"   步骤 {i}: 奖励 {rewards[0].item():.3f}")

#                     # 处理pygame事件（避免窗口无响应）
#                 import pygame
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         break
                
#                 if terminated or truncated:
#                     obs, info = wrapped_env.reset()
#                 if dones:
#                     render_obs = render_env.reset()
                    
#                 # 添加小延迟以便观察
#                 import time
#                 time.sleep(0.05)  # 20 FPS

#             print("✅ 渲染测试完成")
            
#             envs.close()
#             print("✅ 向量化环境关闭成功")
#         else:
#             print("⏭️  跳过向量化环境测试")
        
#     except Exception as e:
#         print(f"❌ 向量化环境测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试5: 多进程环境（如果支持）
#     print("\n=== 测试5: 多进程环境 ===")
#     try:
#         if vec_env_available:
#             print("🚀 尝试创建2进程环境...")
            
#             env_params = {
#                 'num_links': 3,
#                 'link_lengths': [80, 50, 30],
#                 'config_path': "configs/reacher_with_zigzag_obstacles.yaml",
#                 'render_mode': "human"
#             }
            
#             device = torch.device('cpu')
            
#             # 创建多进程向量化环境
#             envs = make_reacher2d_vec_envs(
#                 env_params=env_params,
#                 seed=42,
#                 num_processes=5,  # 2进程测试
#                 gamma=0.99,
#                 log_dir=None,
#                 device=device,
#                 allow_early_resets=False
#             )
#             print("✅ 多进程向量化环境创建成功")
            
#             # 测试并行操作
#             obs = envs.reset()
#             print(f"✅ 多进程重置成功，观察形状: {obs.shape}")
            
#             # 并行执行几步
#             import time
#             start_time = time.time()
#             for i in range(5000):
#                 actions = torch.randn(5, envs.action_space.shape[0]) * 0.5
#                 obs, rewards, dones, infos = envs.step(actions)

#                 print(f"   步骤 {i}: 奖励 {rewards.numpy()}, obs: {obs.shape}")
            
#             elapsed = time.time() - start_time
#             print(f"✅ 5步并行执行完成，用时: {elapsed:.3f}秒")
            
#             envs.close()
#             print("✅ 多进程环境关闭成功")
#         else:
#             print("⏭️  跳过多进程环境测试（依赖不可用）")
        
#     except Exception as e:
#         print(f"❌ 多进程环境测试失败: {e}")
#         print("   这可能是正常的，取决于系统支持情况")
#         # 不打印完整错误，因为多进程可能在某些系统上不工作
    
#     # 测试总结
#     print("\n🎉 测试完成！")
#     print("=" * 50)
#     print("如果看到这里，说明基本功能都正常工作。")
#     print("如果有任何 ❌ 错误，请检查相应的依赖和路径设置。")
#     print("⚠️  警告通常是可以忽略的（表示某些高级功能不可用）。")





#     print("🔄 执行可视化测试...")

#     # 创建单独的可视化环境
#     vis_env = Reacher2DEnv(
#         num_links=4,  # 减少关节数，运动更明显
#         link_lengths=[80, 80, 80, 60],  # 增加连杆长度
#         render_mode="human",
#         config_path=abs_config_path
#     )

#     # 调整物理参数让运动更明显
#     vis_env.max_torque = 100.0  # 大幅增加最大扭矩
#     vis_env.space.damping = 0.85  # 减少阻尼让运动更自由
#     vis_env.dt = 1/25.0  # 稍大的时间步长

#     # 减少body质量让它们更容易运动
#     for body in vis_env.bodies:
#         body.mass = body.mass * 0.4  # 减少质量
#         body.moment = body.moment * 0.4  # 减少转动惯量

#     print("🎥 开始大幅度可视化（按ESC退出）...")
#     vis_obs = vis_env.reset()

#     import pygame
#     import time

#     try:
#         for i in range(2000):
#             # 处理事件
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     break
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         break
            
#             # 生成整个时间段都大幅度的动作
#             t = i * 0.01  # 统一的时间频率
            
#             # 多层叠加的大幅度正弦波，创造复杂而明显的运动
#             action = np.array([
#                 # 第一关节：主要大幅摆动
#                 60 * np.sin(t) + 25 * np.sin(t * 2.3),
                
#                 # 第二关节：跟随摆动，稍有延迟
#                 50 * np.sin(t + 0.5) + 20 * np.sin(t * 1.7),
                
#                 # 第三关节：更快的振荡
#                 45 * np.sin(t * 1.2 + 1) + 15 * np.sin(t * 3.1),
                
#                 # 第四关节：高频小幅叠加
#                 40 * np.sin(t * 0.8 + 1.5) + 12 * np.sin(t * 4.2)
#             ])
            
#             # 确保动作在范围内
#             action = np.clip(action, -80, 80)  # 扩大扭矩限制
            
#             # 执行动作
#             vis_obs, vis_reward, vis_done, vis_info = vis_env.step(action)
            
#             # 渲染
#             vis_env.render()
            
#             if i % 50 == 0:
#                 end_pos = vis_env._get_end_effector_position()
#                 print(f"   步骤 {i}: 奖励 {vis_reward:.3f}, 末端位置 ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
            
#             if vis_done:
#                 vis_obs = vis_env.reset()
            
#             time.sleep(0.06)  # 稍慢的帧率让运动更明显

#     except KeyboardInterrupt:
#         print("\n⏹️ 用户停止可视化")

#     finally:
#         vis_env.close()
#         print("✅ 可视化环境关闭")


# 🎯 便捷函数：自动选择最佳环境创建方式



def make_smart_reacher2d_vec_envs(env_params, seed, num_processes, gamma, log_dir, device,
                                allow_early_resets=True, prefer_async_render=True):
    """
    智能创建Reacher2D向量化环境
    - 单进程 + 渲染模式 → 普通向量化环境
    - 多进程 + 渲染模式 → 异步渲染向量化环境
    - 无渲染模式 → 普通向量化环境
    """
    
    needs_render = env_params.get('render_mode') == 'human'
    is_multiprocess = num_processes > 1
    
    if needs_render and is_multiprocess and prefer_async_render and ASYNC_RENDER_AVAILABLE:
        print("🚀 使用异步渲染多进程环境")
        
        # 🔧 直接在这里实现异步渲染逻辑，不调用外部函数
        # 确保训练环境不渲染（避免多进程渲染冲突）
        train_env_params = env_params.copy()
        train_env_params['render_mode'] = None
        print("⚠️ 多进程训练环境自动禁用渲染，使用异步渲染器代替")
        
        # 创建基础向量化环境
        base_vec_env = make_reacher2d_vec_envs(
            env_params=train_env_params,
            seed=seed,
            num_processes=num_processes,
            gamma=gamma,
            log_dir=log_dir,
            device=device,
            allow_early_resets=allow_early_resets
        )
        
        # 包装异步渲染功能
        async_env = AsyncRenderableVecEnv(
            vec_env=base_vec_env,
            env_params=env_params,  # 使用原始参数（包含渲染模式）
            render_env_id=0,
            enable_async_render=True
        )
        
        return async_env
        
    else:
        if needs_render and is_multiprocess:
            print("⚠️ 多进程环境强制禁用渲染（异步渲染不可用）")
            env_params = env_params.copy()
            env_params['render_mode'] = None
        
        print("🏃 使用标准向量化环境")
        return make_reacher2d_vec_envs(
            env_params=env_params,
            seed=seed,
            num_processes=num_processes,
            gamma=gamma,
            log_dir=log_dir,
            device=device,
            allow_early_resets=allow_early_resets
        )

# 🧪 在现有测试代码中添加异步渲染测试
def test_async_render_multiprocess():
    """测试异步渲染多进程环境"""
    print("\n=== 测试6: 异步渲染多进程环境 ===")
    
    if not ASYNC_RENDER_AVAILABLE:
        print("⏭️ 跳过异步渲染测试（依赖不可用）")
        return
    
    try:
        import torch
        import time
        
        # 环境参数
        env_params = {
            'num_links': 4,
            'link_lengths': [80, 80, 80, 60],
            'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
            'render_mode': 'human'  # 这会触发异步渲染
        }
        
        device = torch.device('cpu')
        
        # 使用智能环境创建函数
        envs = make_smart_reacher2d_vec_envs(
            env_params=env_params,
            seed=42,
            num_processes=4,  # 4进程
            gamma=0.99,
            log_dir=None,
            device=device,
            allow_early_resets=False,
            prefer_async_render=True  # 优先使用异步渲染
        )
        print("✅ 智能环境创建成功")
        print(f"   环境类型: {type(envs).__name__}")
        print(f"   是否异步渲染: {hasattr(envs, 'renderer') and envs.renderer is not None}")
        
        # 测试并行训练 + 异步渲染
        obs = envs.reset()
        print(f"✅ 环境重置成功，观察形状: {obs.shape}")
        
        # 只有当确实是异步渲染环境时才进行长时间测试
        if hasattr(envs, 'renderer') and envs.renderer:
            print("🎮 开始异步渲染训练（在渲染窗口中按ESC退出）...")
            
            start_time = time.time()
            
            for step in range(1000):  # 运行1000步
                # 4个进程的随机动作
                actions = torch.randn(4, envs.action_space.shape[0]) * 2.0
                
                # 并行步进
                obs, rewards, dones, infos = envs.step(actions)
                
                # 定期打印统计
                if step % 100 == 0:
                    render_stats = envs.get_render_stats()
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        train_fps = step / elapsed
                    else:
                        train_fps = 0
                    print(f"   步骤 {step}: 训练FPS={train_fps:.1f}, "
                          f"渲染FPS={render_stats.get('fps', 0):.1f}, "
                          f"丢帧率={render_stats.get('drop_rate', 0):.1f}%")
                
                # 检查渲染进程是否还活着
                if not envs.renderer.render_process.is_alive():
                    print("🔴 渲染进程已退出，停止训练")
                    break
                
                # 小延迟避免过度占用CPU
                if step % 20 == 0:
                    time.sleep(0.01)
            
            elapsed = time.time() - start_time
            final_stats = envs.get_render_stats()
            
            print(f"✅ 异步渲染测试完成")
            print(f"   总用时: {elapsed:.2f}秒")
            if elapsed > 0:
                print(f"   平均训练FPS: {1000/elapsed:.1f}")
            print(f"   平均渲染FPS: {final_stats.get('fps', 0):.1f}")
            print(f"   总丢帧率: {final_stats.get('drop_rate', 0):.1f}%")
        else:
            print("⚠️ 未检测到异步渲染，只进行基本功能测试")
            # 简单的功能测试
            for step in range(10):
                actions = torch.randn(4, envs.action_space.shape[0])
                obs, rewards, dones, infos = envs.step(actions)
                if step % 5 == 0:
                    print(f"   基本测试步骤 {step}: ✅")
            print("✅ 基本功能测试完成")
        
        envs.close()
        print("✅ 环境关闭成功")
        
    except Exception as e:
        print(f"❌ 异步渲染测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_async_render_multiprocess()


# # 修改 env_wrapper.py 中的现有测试代码
# if __name__ == "__main__":
#     print("🧪 开始测试环境包装器...")
    
#     import sys
#     import os
#     import torch
#     import numpy as np
    
#     # 添加必要的路径
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     base_dir = os.path.join(current_dir, '../../../')
#     sys.path.append(base_dir)
#     sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
    
#     print(f"📁 当前目录: {current_dir}")
#     print(f"📁 基础目录: {base_dir}")
    
#     # 测试1: 基本导入
#     print("\n=== 测试1: 基本导入 ===")
#     try:
#         from reacher2d_env import Reacher2DEnv
#         print("✅ 成功导入 Reacher2DEnv")
#     except Exception as e:
#         print(f"❌ 导入 Reacher2DEnv 失败: {e}")
#         sys.exit(1)
    
#     # 测试2: 单个环境创建和包装
#     print("\n=== 测试2: 环境创建和包装 ===")

#     abs_config_path = "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"

#     print(f"🔍 使用绝对路径: {abs_config_path}")
#     print(f"🔍 文件存在: {os.path.exists(abs_config_path)}")
    
#     try:
#         # 创建基础环境
#         base_env = Reacher2DEnv(
#             num_links=3,
#             link_lengths=[80, 50, 30],
#             render_mode="human",
#             config_path = abs_config_path
#         )
#         print("✅ 基础环境创建成功")
        
#         # 包装环境
#         wrapped_env = Reacher2DEnvWrapper(base_env)
#         print("✅ 环境包装成功")
        
#         print(f"   动作空间: {wrapped_env.action_space}")
#         print(f"   观察空间: {wrapped_env.observation_space}")
#         print(f"   关节数量: {wrapped_env.action_space.shape[0]}")
        
#         # 测试重置
#         obs, info = wrapped_env.reset()
#         print(f"✅ 重置成功，观察维度: {obs.shape}, 信息类型: {type(info)}")
        
#         # 测试步进
#         action = np.random.uniform(-1, 1, wrapped_env.action_space.shape[0])
#         obs, reward, terminated, truncated, info = wrapped_env.step(action)
#         print(f"✅ 步进成功")
#         print(f"   观察维度: {obs.shape}")
#         print(f"   奖励: {reward:.3f}")
#         print(f"   结束状态: terminated={terminated}, truncated={truncated}")
        
#         # 测试种子设置
#         seed_result = wrapped_env.seed(123)
#         print(f"✅ 种子设置成功: {seed_result}")
        
#         wrapped_env.close()
#         print("✅ 环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ 环境包装测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试3: Thunk 创建
#     print("\n=== 测试3: Thunk 创建 ===")
#     try:
#         env_params = {
#             'num_links': 5,
#             'link_lengths': [80, 50, 30, 20, 10],
#             'config_path': None,
#             'render_mode': "human"
#         }
        
#         # 创建 thunk
#         thunk = make_reacher2d_env(env_params, seed=42, rank=0)
#         print("✅ Thunk 创建成功")
        
#         # 从 thunk 创建环境
#         env = thunk()
#         print("✅ 从 thunk 创建环境成功")
        
#         # 测试基本功能
#         obs, info = env.reset()
#         print(f"✅ Thunk 环境重置成功，观察维度: {obs.shape}")
        
#         action = np.random.uniform(-2, 2, env.action_space.shape[0])
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"✅ Thunk 环境步进成功，奖励: {reward:.3f}")
        
#         env.close()
#         print("✅ Thunk 环境关闭成功")
        
#     except Exception as e:
#         print(f"❌ Thunk 测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试4: 向量化环境创建（单进程）
#     print("\n=== 测试4: 向量化环境（单进程）===")
#     try:
#         # 检查依赖
#         try:
#             from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
#             from a2c_ppo_acktr.envs import VecPyTorch
#             vec_env_available = True
#             print("✅ 向量化环境依赖可用")
#         except ImportError as e:
#             print(f"⚠️  向量化环境依赖不可用: {e}")
#             vec_env_available = False
        
#         if vec_env_available:
#             env_params = {
#                 'num_links': 3,
#                 'link_lengths': [80, 50, 30],
#                 'config_path': abs_config_path,
#                 'render_mode': "human"
#             }
            
#             device = torch.device('cpu')
            
#             # 创建向量化环境
#             envs = make_reacher2d_vec_envs(
#                 env_params=env_params,
#                 seed=42,
#                 num_processes=1,  # 单进程测试
#                 gamma=0.99,
#                 log_dir=None,
#                 device=device,
#                 allow_early_resets=False
#             )
#             print("✅ 单进程向量化环境创建成功")
            
#             print(f"   环境数量: {envs.num_envs}")
#             print(f"   动作空间: {envs.action_space}")
#             print(f"   观察空间: {envs.observation_space}")
            
#             # 测试向量化操作
#             obs = envs.reset()
#             print(f"✅ 向量化重置成功，观察形状: {obs.shape}")
            
#             actions = torch.randn(1, envs.action_space.shape[0])
#             obs, rewards, dones, infos = envs.step(actions)
#             print(f"✅ 向量化步进成功")
#             print(f"   观察形状: {obs.shape}")
#             print(f"   奖励形状: {rewards.shape}")
#             print(f"   完成形状: {dones.shape}")
#             print(f"   奖励值: {rewards[0].item():.3f}")
            
#             # 测试多步执行
#             print("🔄 执行10步测试...")
#             print("🎥 创建渲染环境...")
#             render_env = Reacher2DEnv(
#                 num_links=3,
#                 link_lengths=[80, 50, 30],
#                 render_mode="human",
#                 config_path=abs_config_path  # 使用相同的配置
#             )
#             render_obs = render_env.reset()

#             for i in range(5000):
#                 actions = torch.randn(1, envs.action_space.shape[0])
#                 obs, rewards, dones, infos = envs.step(actions)
#                 render_env.render()
#                 if i % 20 == 0:
#                     print(f"   步骤 {i}: 奖励 {rewards[0].item():.3f}")

#                     # 处理pygame事件（避免窗口无响应）
#                 import pygame
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         break
                
#                 if terminated or truncated:
#                     obs, info = wrapped_env.reset()
#                 if dones:
#                     render_obs = render_env.reset()
                    
#                 # 添加小延迟以便观察
#                 import time
#                 time.sleep(0.05)  # 20 FPS

#             print("✅ 渲染测试完成")
            
#             envs.close()
#             print("✅ 向量化环境关闭成功")
#         else:
#             print("⏭️  跳过向量化环境测试")
        
#     except Exception as e:
#         print(f"❌ 向量化环境测试失败: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # 测试5: 多进程环境（如果支持）
#     print("\n=== 测试5: 多进程环境 ===")
#     try:
#         if vec_env_available:
#             print("🚀 尝试创建2进程环境...")
            
#             env_params = {
#                 'num_links': 3,
#                 'link_lengths': [80, 50, 30],
#                 'config_path': "configs/reacher_with_zigzag_obstacles.yaml",
#                 'render_mode': "human"
#             }
            
#             device = torch.device('cpu')
            
#             # 创建多进程向量化环境
#             envs = make_reacher2d_vec_envs(
#                 env_params=env_params,
#                 seed=42,
#                 num_processes=5,  # 2进程测试
#                 gamma=0.99,
#                 log_dir=None,
#                 device=device,
#                 allow_early_resets=False
#             )
#             print("✅ 多进程向量化环境创建成功")
            
#             # 测试并行操作
#             obs = envs.reset()
#             print(f"✅ 多进程重置成功，观察形状: {obs.shape}")
            
#             # 并行执行几步
#             import time
#             start_time = time.time()
#             for i in range(5000):
#                 actions = torch.randn(5, envs.action_space.shape[0]) * 0.5
#                 obs, rewards, dones, infos = envs.step(actions)

#                 print(f"   步骤 {i}: 奖励 {rewards.numpy()}, obs: {obs.shape}")
            
#             elapsed = time.time() - start_time
#             print(f"✅ 5步并行执行完成，用时: {elapsed:.3f}秒")
            
#             envs.close()
#             print("✅ 多进程环境关闭成功")
#         else:
#             print("⏭️  跳过多进程环境测试（依赖不可用）")
        
#     except Exception as e:
#         print(f"❌ 多进程环境测试失败: {e}")
#         print("   这可能是正常的，取决于系统支持情况")
#         # 不打印完整错误，因为多进程可能在某些系统上不工作
    
#     # 测试总结
#     print("\n🎉 测试完成！")
#     print("=" * 50)
#     print("如果看到这里，说明基本功能都正常工作。")
#     print("如果有任何 ❌ 错误，请检查相应的依赖和路径设置。")
#     print("⚠️  警告通常是可以忽略的（表示某些高级功能不可用）。")





#     print("🔄 执行可视化测试...")

#     # 创建单独的可视化环境
#     vis_env = Reacher2DEnv(
#         num_links=4,  # 减少关节数，运动更明显
#         link_lengths=[80, 80, 80, 60],  # 增加连杆长度
#         render_mode="human",
#         config_path=abs_config_path
#     )

#     # 调整物理参数让运动更明显
#     vis_env.max_torque = 100.0  # 大幅增加最大扭矩
#     vis_env.space.damping = 0.85  # 减少阻尼让运动更自由
#     vis_env.dt = 1/25.0  # 稍大的时间步长

#     # 减少body质量让它们更容易运动
#     for body in vis_env.bodies:
#         body.mass = body.mass * 0.4  # 减少质量
#         body.moment = body.moment * 0.4  # 减少转动惯量

#     print("🎥 开始大幅度可视化（按ESC退出）...")
#     vis_obs = vis_env.reset()

#     import pygame
#     import time

#     try:
#         for i in range(2000):
#             # 处理事件
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     break
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         break
            
#             # 生成整个时间段都大幅度的动作
#             t = i * 0.01  # 统一的时间频率
            
#             # 多层叠加的大幅度正弦波，创造复杂而明显的运动
#             action = np.array([
#                 # 第一关节：主要大幅摆动
#                 60 * np.sin(t) + 25 * np.sin(t * 2.3),
                
#                 # 第二关节：跟随摆动，稍有延迟
#                 50 * np.sin(t + 0.5) + 20 * np.sin(t * 1.7),
                
#                 # 第三关节：更快的振荡
#                 45 * np.sin(t * 1.2 + 1) + 15 * np.sin(t * 3.1),
                
#                 # 第四关节：高频小幅叠加
#                 40 * np.sin(t * 0.8 + 1.5) + 12 * np.sin(t * 4.2)
#             ])
            
#             # 确保动作在范围内
#             action = np.clip(action, -80, 80)  # 扩大扭矩限制
            
#             # 执行动作
#             vis_obs, vis_reward, vis_done, vis_info = vis_env.step(action)
            
#             # 渲染
#             vis_env.render()
            
#             if i % 50 == 0:
#                 end_pos = vis_env._get_end_effector_position()
#                 print(f"   步骤 {i}: 奖励 {vis_reward:.3f}, 末端位置 ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
            
#             if vis_done:
#                 vis_obs = vis_env.reset()
            
#             time.sleep(0.06)  # 稍慢的帧率让运动更明显

#     except KeyboardInterrupt:
#         print("\n⏹️ 用户停止可视化")

#     finally:
#         vis_env.close()
#         print("✅ 可视化环境关闭")
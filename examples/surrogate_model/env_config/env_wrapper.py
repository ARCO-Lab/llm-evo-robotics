# examples/surrogate_model/env_wrappers.py

import torch
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from a2c_ppo_acktr.envs import VecPyTorch, VecNormalize




def make_reacher2d_env(env_params, seed, rank, log_dir=None, allow_early_resets=True):
    """
    创建单个 Reacher2D 环境的 thunk 函数
    参考 RoboGrammar 的 make_env 模式
    """
    def _thunk():
        from reacher2d_env import Reacher2DEnv
        
        # 创建环境
        env = Reacher2DEnv(
            num_links=env_params.get('num_links', 5),
            link_lengths=env_params.get('link_lengths', [80, 50, 30, 20, 10]),
            render_mode=env_params.get('render_mode', "human"),  # 训练环境不渲染
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
    if len(envs.observation_space.shape) == 1:
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
    

# 在 env_wrapper.py 文件末尾添加测试代码

if __name__ == "__main__":
    print("🧪 开始测试环境包装器...")
    
    import sys
    import os
    import torch
    import numpy as np
    
    # 添加必要的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '../../../')
    sys.path.append(base_dir)
    sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
    
    print(f"📁 当前目录: {current_dir}")
    print(f"📁 基础目录: {base_dir}")
    
    # 测试1: 基本导入
    print("\n=== 测试1: 基本导入 ===")
    try:
        from reacher2d_env import Reacher2DEnv
        print("✅ 成功导入 Reacher2DEnv")
    except Exception as e:
        print(f"❌ 导入 Reacher2DEnv 失败: {e}")
        sys.exit(1)
    
    # 测试2: 单个环境创建和包装
    print("\n=== 测试2: 环境创建和包装 ===")

    abs_config_path = "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"

    print(f"🔍 使用绝对路径: {abs_config_path}")
    print(f"🔍 文件存在: {os.path.exists(abs_config_path)}")
    
    try:
        # 创建基础环境
        base_env = Reacher2DEnv(
            num_links=3,
            link_lengths=[80, 50, 30],
            render_mode="human",
            config_path = abs_config_path
        )
        print("✅ 基础环境创建成功")
        
        # 包装环境
        wrapped_env = Reacher2DEnvWrapper(base_env)
        print("✅ 环境包装成功")
        
        print(f"   动作空间: {wrapped_env.action_space}")
        print(f"   观察空间: {wrapped_env.observation_space}")
        print(f"   关节数量: {wrapped_env.action_space.shape[0]}")
        
        # 测试重置
        obs, info = wrapped_env.reset()
        print(f"✅ 重置成功，观察维度: {obs.shape}, 信息类型: {type(info)}")
        
        # 测试步进
        action = np.random.uniform(-1, 1, wrapped_env.action_space.shape[0])
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"✅ 步进成功")
        print(f"   观察维度: {obs.shape}")
        print(f"   奖励: {reward:.3f}")
        print(f"   结束状态: terminated={terminated}, truncated={truncated}")
        
        # 测试种子设置
        seed_result = wrapped_env.seed(123)
        print(f"✅ 种子设置成功: {seed_result}")
        
        wrapped_env.close()
        print("✅ 环境关闭成功")
        
    except Exception as e:
        print(f"❌ 环境包装测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: Thunk 创建
    print("\n=== 测试3: Thunk 创建 ===")
    try:
        env_params = {
            'num_links': 5,
            'link_lengths': [80, 50, 30, 20, 10],
            'config_path': None,
            'render_mode': "human"
        }
        
        # 创建 thunk
        thunk = make_reacher2d_env(env_params, seed=42, rank=0)
        print("✅ Thunk 创建成功")
        
        # 从 thunk 创建环境
        env = thunk()
        print("✅ 从 thunk 创建环境成功")
        
        # 测试基本功能
        obs, info = env.reset()
        print(f"✅ Thunk 环境重置成功，观察维度: {obs.shape}")
        
        action = np.random.uniform(-2, 2, env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Thunk 环境步进成功，奖励: {reward:.3f}")
        
        env.close()
        print("✅ Thunk 环境关闭成功")
        
    except Exception as e:
        print(f"❌ Thunk 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 向量化环境创建（单进程）
    print("\n=== 测试4: 向量化环境（单进程）===")
    try:
        # 检查依赖
        try:
            from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
            from a2c_ppo_acktr.envs import VecPyTorch
            vec_env_available = True
            print("✅ 向量化环境依赖可用")
        except ImportError as e:
            print(f"⚠️  向量化环境依赖不可用: {e}")
            vec_env_available = False
        
        if vec_env_available:
            env_params = {
                'num_links': 3,
                'link_lengths': [80, 50, 30],
                'config_path': abs_config_path,
                'render_mode': "human"
            }
            
            device = torch.device('cpu')
            
            # 创建向量化环境
            envs = make_reacher2d_vec_envs(
                env_params=env_params,
                seed=42,
                num_processes=1,  # 单进程测试
                gamma=0.99,
                log_dir=None,
                device=device,
                allow_early_resets=False
            )
            print("✅ 单进程向量化环境创建成功")
            
            print(f"   环境数量: {envs.num_envs}")
            print(f"   动作空间: {envs.action_space}")
            print(f"   观察空间: {envs.observation_space}")
            
            # 测试向量化操作
            obs = envs.reset()
            print(f"✅ 向量化重置成功，观察形状: {obs.shape}")
            
            actions = torch.randn(1, envs.action_space.shape[0])
            obs, rewards, dones, infos = envs.step(actions)
            print(f"✅ 向量化步进成功")
            print(f"   观察形状: {obs.shape}")
            print(f"   奖励形状: {rewards.shape}")
            print(f"   完成形状: {dones.shape}")
            print(f"   奖励值: {rewards[0].item():.3f}")
            
            # 测试多步执行
            print("🔄 执行10步测试...")
            print("🎥 创建渲染环境...")
            render_env = Reacher2DEnv(
                num_links=3,
                link_lengths=[80, 50, 30],
                render_mode="human",
                config_path=abs_config_path  # 使用相同的配置
            )
            render_obs = render_env.reset()

            for i in range(5000):
                actions = torch.randn(1, envs.action_space.shape[0])
                obs, rewards, dones, infos = envs.step(actions)
                render_env.render()
                if i % 20 == 0:
                    print(f"   步骤 {i}: 奖励 {rewards[0].item():.3f}")

                    # 处理pygame事件（避免窗口无响应）
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                
                if terminated or truncated:
                    obs, info = wrapped_env.reset()
                if dones:
                    render_obs = render_env.reset()
                    
                # 添加小延迟以便观察
                import time
                time.sleep(0.05)  # 20 FPS

            print("✅ 渲染测试完成")
            
            envs.close()
            print("✅ 向量化环境关闭成功")
        else:
            print("⏭️  跳过向量化环境测试")
        
    except Exception as e:
        print(f"❌ 向量化环境测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试5: 多进程环境（如果支持）
    print("\n=== 测试5: 多进程环境 ===")
    try:
        if vec_env_available:
            print("🚀 尝试创建2进程环境...")
            
            env_params = {
                'num_links': 3,
                'link_lengths': [80, 50, 30],
                'config_path': "configs/reacher_with_zigzag_obstacles.yaml",
                'render_mode': "human"
            }
            
            device = torch.device('cpu')
            
            # 创建多进程向量化环境
            envs = make_reacher2d_vec_envs(
                env_params=env_params,
                seed=42,
                num_processes=5,  # 2进程测试
                gamma=0.99,
                log_dir=None,
                device=device,
                allow_early_resets=False
            )
            print("✅ 多进程向量化环境创建成功")
            
            # 测试并行操作
            obs = envs.reset()
            print(f"✅ 多进程重置成功，观察形状: {obs.shape}")
            
            # 并行执行几步
            import time
            start_time = time.time()
            for i in range(5000):
                actions = torch.randn(5, envs.action_space.shape[0]) * 0.5
                obs, rewards, dones, infos = envs.step(actions)

                print(f"   步骤 {i}: 奖励 {rewards.numpy()}, obs: {obs.shape}")
            
            elapsed = time.time() - start_time
            print(f"✅ 5步并行执行完成，用时: {elapsed:.3f}秒")
            
            envs.close()
            print("✅ 多进程环境关闭成功")
        else:
            print("⏭️  跳过多进程环境测试（依赖不可用）")
        
    except Exception as e:
        print(f"❌ 多进程环境测试失败: {e}")
        print("   这可能是正常的，取决于系统支持情况")
        # 不打印完整错误，因为多进程可能在某些系统上不工作
    
    # 测试总结
    print("\n🎉 测试完成！")
    print("=" * 50)
    print("如果看到这里，说明基本功能都正常工作。")
    print("如果有任何 ❌ 错误，请检查相应的依赖和路径设置。")
    print("⚠️  警告通常是可以忽略的（表示某些高级功能不可用）。")





    print("🔄 执行可视化测试...")

    # 创建单独的可视化环境
    vis_env = Reacher2DEnv(
        num_links=4,  # 减少关节数，运动更明显
        link_lengths=[80, 80, 80, 60],  # 增加连杆长度
        render_mode="human",
        config_path=abs_config_path
    )

    # 调整物理参数让运动更明显
    vis_env.max_torque = 100.0  # 大幅增加最大扭矩
    vis_env.space.damping = 0.85  # 减少阻尼让运动更自由
    vis_env.dt = 1/25.0  # 稍大的时间步长

    # 减少body质量让它们更容易运动
    for body in vis_env.bodies:
        body.mass = body.mass * 0.4  # 减少质量
        body.moment = body.moment * 0.4  # 减少转动惯量

    print("🎥 开始大幅度可视化（按ESC退出）...")
    vis_obs = vis_env.reset()

    import pygame
    import time

    try:
        for i in range(2000):
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        break
            
            # 生成整个时间段都大幅度的动作
            t = i * 0.01  # 统一的时间频率
            
            # 多层叠加的大幅度正弦波，创造复杂而明显的运动
            action = np.array([
                # 第一关节：主要大幅摆动
                60 * np.sin(t) + 25 * np.sin(t * 2.3),
                
                # 第二关节：跟随摆动，稍有延迟
                50 * np.sin(t + 0.5) + 20 * np.sin(t * 1.7),
                
                # 第三关节：更快的振荡
                45 * np.sin(t * 1.2 + 1) + 15 * np.sin(t * 3.1),
                
                # 第四关节：高频小幅叠加
                40 * np.sin(t * 0.8 + 1.5) + 12 * np.sin(t * 4.2)
            ])
            
            # 确保动作在范围内
            action = np.clip(action, -80, 80)  # 扩大扭矩限制
            
            # 执行动作
            vis_obs, vis_reward, vis_done, vis_info = vis_env.step(action)
            
            # 渲染
            vis_env.render()
            
            if i % 50 == 0:
                end_pos = vis_env._get_end_effector_position()
                print(f"   步骤 {i}: 奖励 {vis_reward:.3f}, 末端位置 ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
            
            if vis_done:
                vis_obs = vis_env.reset()
            
            time.sleep(0.06)  # 稍慢的帧率让运动更明显

    except KeyboardInterrupt:
        print("\n⏹️ 用户停止可视化")

    finally:
        vis_env.close()
        print("✅ 可视化环境关闭")
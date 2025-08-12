# async_renderer.py
import multiprocessing as mp
import threading
import time
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from queue import Queue, Empty
import pickle

class AsyncRenderer:
    """异步渲染器 - 在独立进程中处理渲染，不阻塞训练"""
    
    def __init__(self, env_params, max_queue_size=10):
        """
        初始化异步渲染器
        
        Args:
            env_params: 环境参数（num_links, link_lengths等）
            max_queue_size: 渲染队列最大长度
        """
        self.env_params = env_params
        self.max_queue_size = max_queue_size
        
        # 🔄 进程间通信
        self.render_queue = mp.Queue(maxsize=max_queue_size)
        self.control_queue = mp.Queue()  # 控制信号队列
        self.render_process = None
        
        # 📊 性能统计
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        
    def start(self):
        """启动异步渲染进程"""
        if self.render_process and self.render_process.is_alive():
            print("⚠️ 渲染进程已经在运行")
            return
            
        print("🎨 启动异步渲染进程...")
        self.render_process = mp.Process(
            target=self._render_worker,
            args=(self.render_queue, self.control_queue, self.env_params),
            daemon=True  # 主进程结束时自动结束
        )
        self.render_process.start()
        self.start_time = time.time()
        print(f"✅ 渲染进程已启动 (PID: {self.render_process.pid})")
        
    def render_frame(self, robot_state):
        """
        发送一帧数据到渲染进程
        
        Args:
            robot_state: 包含机器人状态的字典
                - body_positions: [(x, y, angle), ...] 每个body的位置和角度
                - goal_pos: [x, y] 目标位置
                - step_count: 当前步数（可选）
        """
        if not self.render_process or not self.render_process.is_alive():
            return
            
        try:
            # 🚀 非阻塞发送：队列满时丢弃帧而不是等待
            self.render_queue.put_nowait(robot_state)
            self.frame_count += 1
        except:
            # 队列满，丢弃帧
            self.dropped_frames += 1
            
    def get_stats(self):
        """获取渲染统计信息"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            drop_rate = self.dropped_frames / max(1, self.frame_count + self.dropped_frames)
            
            return {
                'fps': fps,
                'total_frames': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'drop_rate': drop_rate * 100,
                'queue_size': self.render_queue.qsize() if hasattr(self.render_queue, 'qsize') else 'unknown'
            }
        return {}
        
    def stop(self):
        """停止异步渲染进程"""
        if self.render_process and self.render_process.is_alive():
            print("🛑 停止渲染进程...")
            
            # 发送停止信号
            try:
                self.control_queue.put_nowait('STOP')
            except:
                pass
                
            # 等待进程结束
            self.render_process.join(timeout=3)
            
            if self.render_process.is_alive():
                print("⚠️ 渲染进程未响应，强制终止")
                self.render_process.terminate()
                self.render_process.join(timeout=1)
                
            print("✅ 渲染进程已停止")
            
            # 打印统计信息
            stats = self.get_stats()
            print(f"📊 渲染统计: 总帧数={stats.get('total_frames', 0)}, "
                  f"丢帧率={stats.get('drop_rate', 0):.1f}%, "
                  f"平均FPS={stats.get('fps', 0):.1f}")
                  
    # @staticmethod
    # def _render_worker(render_queue, control_queue, env_params):
    #     """渲染工作进程 - 在独立进程中运行"""
    #     try:
    #         # 🎮 初始化Pygame（在渲染进程中）
    #         pygame.init()
    #         screen = pygame.display.set_mode((1200, 1200))
    #         pygame.display.set_caption("Reacher2D 异步渲染 - 按ESC退出")
    #         clock = pygame.time.Clock()
            
    #         # 🎨 初始化绘制参数
    #         WHITE = (255, 255, 255)
    #         RED = (255, 0, 0)
    #         BLACK = (0, 0, 0)
    #         BLUE = (0, 0, 255)
    #         GREEN = (0, 255, 0)
    #         GRAY = (128, 128, 128)
    #         ORANGE = (255, 165, 0)  # 障碍物颜色
    #         PURPLE = (128, 0, 128)  # 锚点颜色
            
    #         print("🎨 渲染进程初始化完成")
            
    #         frame_count = 0
    #         last_stats_time = time.time()
    #         running = True
    #         last_robot_state = None  # 保存最后一个状态用于连续渲染
            
    #         while running:
    #             try:
    #                 # 🎮 关键修复：处理Pygame事件
    #                 for event in pygame.event.get():
    #                     if event.type == pygame.QUIT:
    #                         print("🔴 用户关闭窗口")
    #                         running = False
    #                         break
    #                     elif event.type == pygame.KEYDOWN:
    #                         if event.key == pygame.K_ESCAPE:
    #                             print("🔴 用户按ESC退出")
    #                             running = False
    #                             break
                    
    #                 if not running:
    #                     break
                    
    #                 # 🛑 检查控制信号
    #                 try:
    #                     control_signal = control_queue.get_nowait()
    #                     if control_signal == 'STOP':
    #                         print("🔴 收到停止信号")
    #                         break
    #                 except:
    #                     pass
                    
    #                 # 🎞️ 获取渲染数据
    #                 new_data = False
    #                 try:
    #                     robot_state = render_queue.get(timeout=0.016)  # 60FPS的超时
    #                     last_robot_state = robot_state
    #                     new_data = True
    #                 except:
    #                     # 没有新数据，使用最后一个状态继续渲染
    #                     robot_state = last_robot_state
                    
    #                 # 🖼️ 渲染帧（即使没有新数据也要渲染，保持窗口响应）
    #                 screen.fill(WHITE)
                    
    #                 if robot_state:
    #                     # 🚧 绘制障碍物（在最底层）
    #                     if 'obstacles' in robot_state:
    #                         for obstacle in robot_state['obstacles']:
    #                             if obstacle['type'] == 'segment':
    #                                 points = obstacle['points']
    #                                 p1 = (int(points[0][0]), int(points[0][1]))
    #                                 p2 = (int(points[1][0]), int(points[1][1]))
    #                                 radius = int(obstacle.get('radius', 3))
                                    
    #                                 # 绘制粗线表示障碍物
    #                                 pygame.draw.line(screen, ORANGE, p1, p2, radius * 2)
                                    
    #                                 # 在端点绘制圆形
    #                                 pygame.draw.circle(screen, ORANGE, p1, radius)
    #                                 pygame.draw.circle(screen, ORANGE, p2, radius)
                        
    #                     # 🏠 绘制锚点
    #                     if 'anchor_point' in robot_state:
    #                         anchor = robot_state['anchor_point']
    #                         pygame.draw.circle(screen, PURPLE, 
    #                                          (int(anchor[0]), int(anchor[1])), 8)
    #                         # 绘制锚点标识
    #                         font = pygame.font.Font(None, 24)
    #                         text = font.render("ANCHOR", True, PURPLE)
    #                         screen.blit(text, (int(anchor[0]) + 10, int(anchor[1]) - 10))
                        
    #                     # 🎯 绘制目标点
    #                     if 'goal_pos' in robot_state:
    #                         goal_pos = robot_state['goal_pos']
    #                         goal_radius = robot_state.get('goal_radius', 10)
    #                         pygame.draw.circle(screen, RED, 
    #                                          (int(goal_pos[0]), int(goal_pos[1])), int(goal_radius))
    #                         # 绘制目标标识
    #                         font = pygame.font.Font(None, 24)
    #                         text = font.render("GOAL", True, RED)
    #                         screen.blit(text, (int(goal_pos[0]) + 15, int(goal_pos[1]) - 10))
                        
    #                     # 🤖 绘制机器人links
    #                     if 'body_positions' in robot_state:
    #                         positions = robot_state['body_positions']
    #                         link_lengths = env_params.get('link_lengths', [60] * len(positions))
                            
    #                         for i, (pos, length) in enumerate(zip(positions, link_lengths)):
    #                             x, y, angle = pos
                                
    #                             # 计算link的两个端点
    #                             end_x = x + length * np.cos(angle)
    #                             end_y = y + length * np.sin(angle)
                                
    #                             # 绘制link（粗线）
    #                             pygame.draw.line(screen, BLACK, 
    #                                            (int(x), int(y)), (int(end_x), int(end_y)), 8)
                                
    #                             # 绘制关节（圆点）
    #                             color = BLUE if i == 0 else GREEN
    #                             pygame.draw.circle(screen, color, (int(x), int(y)), 6)
                                
    #                             # 绘制关节编号
    #                             font = pygame.font.Font(None, 20)
    #                             text = font.render(str(i), True, WHITE)
    #                             text_rect = text.get_rect(center=(int(x), int(y)))
    #                             screen.blit(text, text_rect)
                        
    #                     # 📊 绘制信息文本
    #                     if 'step_count' in robot_state:
    #                         font = pygame.font.Font(None, 36)
    #                         text = font.render(f"Step: {robot_state['step_count']}", 
    #                                          True, BLACK)
    #                         screen.blit(text, (10, 10))
                            
    #                         # 显示数据状态
    #                         status_text = "NEW DATA" if new_data else "REPEATING"
    #                         status_color = GREEN if new_data else GRAY
    #                         status_surface = font.render(status_text, True, status_color)
    #                         screen.blit(status_surface, (10, 50))
                            
    #                         # 显示障碍物数量
    #                         if 'obstacles' in robot_state:
    #                             obstacle_count = len(robot_state['obstacles'])
    #                             obstacle_text = f"Obstacles: {obstacle_count}"
    #                             obstacle_surface = font.render(obstacle_text, True, ORANGE)
    #                             screen.blit(obstacle_surface, (10, 90))
    #                 else:
    #                     # 没有数据时显示等待信息
    #                     font = pygame.font.Font(None, 48)
    #                     text = font.render("等待数据...", True, GRAY)
    #                     text_rect = text.get_rect(center=(600, 600))
    #                     screen.blit(text, text_rect)
                    
    #                 # 🔄 更新显示
    #                 pygame.display.flip()
    #                 clock.tick(60)  # 限制60FPS
                    
    #                 frame_count += 1
                    
    #                 # 📊 定期打印统计
    #                 if frame_count % 300 == 0:  # 每5秒
    #                     current_time = time.time()
    #                     elapsed = current_time - last_stats_time
    #                     fps = 300 / elapsed if elapsed > 0 else 0
    #                     queue_size = render_queue.qsize() if hasattr(render_queue, 'qsize') else 'unknown'
    #                     print(f"🎨 渲染进程: 帧数={frame_count}, FPS={fps:.1f}, 队列={queue_size}")
    #                     last_stats_time = current_time
                        
    #             except Exception as e:
    #                 print(f"❌ 渲染进程错误: {e}")
    #                 import traceback
    #                 traceback.print_exc()
    #                 time.sleep(0.1)  # 短暂等待避免错误循环
    #                 continue
                    
    #     except Exception as e:
    #         print(f"❌ 渲染进程启动失败: {e}")
    #         import traceback
    #         traceback.print_exc()
    #     finally:
    #         try:
    #             pygame.quit()
    #             print("🎨 渲染进程已清理")
    #         except:
    #             pass

    @staticmethod
    def _render_worker(render_queue, control_queue, env_params):
        """混合渲染：原生机器人 + 自定义信息 - 修复Link不动问题"""
        try:
            # 🤖 创建Reacher2DEnv实例
            from reacher2d_env import Reacher2DEnv
            import numpy as np
            
            render_env_params = env_params.copy()
            render_env_params['render_mode'] = 'human'
            render_env = Reacher2DEnv(**render_env_params)
            
            # 获取pygame组件
            screen = render_env.screen
            clock = render_env.clock
            
            print("🎨 混合渲染进程初始化完成 - 修复Link同步问题")
            
            frame_count = 0
            last_stats_time = time.time()
            running = True
            last_robot_state = None
            
            while running:
                try:
                    # 🎮 关键修复：处理Pygame事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("🔴 用户关闭窗口")
                            running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("🔴 用户按ESC退出")
                                running = False
                                break
                    
                    if not running:
                        break
                    
                    # 🛑 检查控制信号
                    try:
                        control_signal = control_queue.get_nowait()
                        if control_signal == 'STOP':
                            print("🔴 收到停止信号")
                            break
                    except:
                        pass
                    
                    # 🎞️ 获取渲染数据
                    new_data = False
                    try:
                        robot_state = render_queue.get(timeout=0.016)  # 60FPS的超时
                        last_robot_state = robot_state
                        new_data = True
                    except:
                        # 没有新数据，使用最后一个状态继续渲染
                        robot_state = last_robot_state
                    
                    # 🔑 关键修复：正确同步环境状态
                    if robot_state and 'body_positions' in robot_state:
                        positions = robot_state['body_positions']
                        
                        # 同步body位置和角度到渲染环境
                        for i, (x, y, angle) in enumerate(positions):
                            if i < len(render_env.bodies):
                                body = render_env.bodies[i]
                                # 设置新的位置和角度
                                body.position = (x, y)
                                body.angle = angle
                                
                                # 🔧 关键修复：手动更新body关联的所有shape
                                for shape in body.shapes:
                                    # 强制更新shape的缓存边界框和变换
                                    shape.cache_bb()
                        
                        # 🔧 另一种方法：执行一个微小的物理步进来更新所有形状
                        # 保存当前速度
                        velocities = []
                        angular_velocities = []
                        for body in render_env.bodies:
                            velocities.append(body.velocity)
                            angular_velocities.append(body.angular_velocity)
                            # 暂时清零速度，避免位置漂移
                            body.velocity = (0, 0)
                            body.angular_velocity = 0
                        
                        # 执行微小物理步进以更新shape位置
                        render_env.space.step(0.001)  # 非常小的时间步长
                        
                        # 恢复速度（保持静态显示）
                        for i, body in enumerate(render_env.bodies):
                            body.velocity = velocities[i]
                            body.angular_velocity = angular_velocities[i]
                        
                        # 同步目标位置（如果有变化）
                        if 'goal_pos' in robot_state:
                            render_env.goal_pos = np.array(robot_state['goal_pos'])
                    
                    # 🎨 使用原生PyMunk渲染风格
                    screen.fill((255, 255, 255))  # 白色背景
                    
                    # 绘制目标点（与原生风格一致）
                    if hasattr(render_env, 'goal_pos') and render_env.goal_pos is not None:
                        pygame.draw.circle(screen, (255, 0, 0), render_env.goal_pos.astype(int), 10)
                    
                    # 🎯 绘制安全区域（可选调试，与原生一致）
                    if hasattr(render_env, 'bodies') and len(render_env.bodies) > 0:
                        for body in render_env.bodies:
                            pos = (int(body.position[0]), int(body.position[1]))
                            # 绘制安全半径（浅蓝色圆圈）
                            pygame.draw.circle(screen, (173, 216, 230), pos, 30, 1)
                    
                    # 🔑 关键：使用PyMunk原生debug_draw渲染机器人和障碍物
                    render_env.space.debug_draw(render_env.draw_options)
                    
                    # 📊 添加自定义信息覆盖层（不影响原生渲染）
                    if robot_state:
                        # 显示步数
                        if 'step_count' in robot_state:
                            font = pygame.font.Font(None, 36)
                            text = font.render(f"Step: {robot_state['step_count']}", True, (0, 0, 0))
                            screen.blit(text, (10, 10))
                            
                            # 显示数据状态
                            status_text = "NEW DATA" if new_data else "REPEATING"
                            status_color = (0, 255, 0) if new_data else (128, 128, 128)
                            status_surface = font.render(status_text, True, status_color)
                            screen.blit(status_surface, (10, 50))
                            
                            # 🔧 新增：显示同步状态调试信息
                            if 'body_positions' in robot_state:
                                positions = robot_state['body_positions']
                                sync_text = f"Bodies synced: {len(positions)}"
                                sync_surface = font.render(sync_text, True, (0, 0, 255))
                                screen.blit(sync_surface, (10, 90))
                                
                                # 显示第一个body的状态作为调试
                                if len(positions) > 0:
                                    x, y, angle = positions[0]
                                    pos_text = f"Body0: ({x:.1f}, {y:.1f}, {angle:.2f})"
                                    pos_surface = pygame.font.Font(None, 24).render(pos_text, True, (128, 0, 128))
                                    screen.blit(pos_surface, (10, 130))
                    
                    else:
                        # 没有数据时显示等待信息
                        font = pygame.font.Font(None, 48)
                        text = font.render("等待数据...", True, (128, 128, 128))
                        text_rect = text.get_rect(center=(600, 600))
                        screen.blit(text, text_rect)
                    
                    # 🔄 更新显示
                    pygame.display.flip()
                    clock.tick(60)  # 限制60FPS
                    
                    frame_count += 1
                    
                    # 📊 定期打印统计
                    if frame_count % 300 == 0:  # 每5秒
                        current_time = time.time()
                        elapsed = current_time - last_stats_time
                        fps = 300 / elapsed if elapsed > 0 else 0
                        queue_size = render_queue.qsize() if hasattr(render_queue, 'qsize') else 'unknown'
                        print(f"🎨 混合渲染进程: 帧数={frame_count}, FPS={fps:.1f}, 队列={queue_size}")
                        last_stats_time = current_time
                        
                except Exception as e:
                    print(f"❌ 渲染进程错误: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)  # 短暂等待避免错误循环
                    continue
                    
        except Exception as e:
            print(f"❌ 渲染进程启动失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if 'render_env' in locals():
                    render_env.close()
                print("🎨 混合渲染进程已清理")
            except:
                pass

    def __del__(self):
        """析构函数：确保进程被正确关闭"""
        self.stop()


# 📦 环境状态提取器
class StateExtractor:
    """从Reacher2D环境中提取渲染所需的状态"""
    
    @staticmethod
    def extract_robot_state(env, step_count=None):
        """
        从环境中提取机器人状态
        
        Args:
            env: Reacher2DEnv实例
            step_count: 当前步数（可选）
            
        Returns:
            包含渲染信息的字典
        """
        try:
            # 🤖 提取body位置和角度
            body_positions = []
            for body in env.bodies:
                x, y = body.position
                angle = body.angle
                body_positions.append((float(x), float(y), float(angle)))
            
            # 🎯 目标位置
            goal_pos = [float(env.goal_pos[0]), float(env.goal_pos[1])]
            goal_radius = getattr(env, 'goal_radius', 10)  # 默认半径10
            
            # 🚧 提取障碍物信息
            obstacles = []
            if hasattr(env, 'obstacles') and env.obstacles:
                for obstacle in env.obstacles:
                    if hasattr(obstacle, 'a') and hasattr(obstacle, 'b'):
                        # Segment障碍物
                        obstacles.append({
                            'type': 'segment',
                            'points': [
                                [float(obstacle.a[0]), float(obstacle.a[1])],
                                [float(obstacle.b[0]), float(obstacle.b[1])]
                            ],
                            'radius': float(getattr(obstacle, 'radius', 3.0))
                        })
            
            # 🏠 提取锚点位置
            anchor_point = [float(env.anchor_point[0]), float(env.anchor_point[1])]
            
            state = {
                'body_positions': body_positions,
                'goal_pos': goal_pos,
                'goal_radius': goal_radius,
                'obstacles': obstacles,
                'anchor_point': anchor_point,
            }
            
            if step_count is not None:
                state['step_count'] = step_count
                
            return state
            
        except Exception as e:
            print(f"❌ 状态提取失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
        



# 🎯 使用示例和测试
if __name__ == "__main__":
    from reacher2d_env import Reacher2DEnv
    import numpy as np 
    
    print("🚀 测试异步渲染器")
    
    # 环境参数
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': None,  # 🔧 训练环境不渲染！
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    # 🎨 创建异步渲染器
    renderer = AsyncRenderer(env_params)
    renderer.start()
    
    # 给渲染进程一点启动时间
    time.sleep(1)
    
    # 🤖 创建训练环境（无渲染）
    env = Reacher2DEnv(**env_params)
    obs = env.reset()
    
    print("🏃 开始训练循环...")
    print("🎮 在渲染窗口中按ESC或关闭窗口可以退出")
    
    try:
        for step in range(100000):
            # 🎲 随机动作
            action = np.random.uniform(-10, 10, env.num_links)
            
            # ⚡ 环境步进（快速）
            obs, reward, done, info = env.step(action)
            
            # 📤 异步发送渲染数据（瞬间完成）
            robot_state = StateExtractor.extract_robot_state(env, step)
            renderer.render_frame(robot_state)
            
            # 🔄 重置检查
            if done:
                obs = env.reset()
            
            # 📊 定期打印训练统计
            if step % 1000 == 0:
                stats = renderer.get_stats()
                print(f"训练步数: {step}, 渲染FPS: {stats.get('fps', 0):.1f}, "
                      f"丢帧率: {stats.get('drop_rate', 0):.1f}%")
            
            # 🛑 检查渲染进程是否还活着
            if not renderer.render_process.is_alive():
                print("🔴 渲染进程已退出，停止训练")
                break
                
            # 🐌 稍微放慢训练速度，让渲染跟得上
            if step % 10 == 0:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断训练")
        
    finally:
        # 🛑 清理资源
        env.close()
        renderer.stop()
        print("🎉 异步渲染测试完成！")


"""
GCN骨架重建GIF动画可视化器 - 内存优化版本
基于gcn_skeleton_visualizer.py，生成多帧骨架重建的GIF动画
展示原始骨架序列和重建骨架序列的对比动画
集成分组损失和关节权重优化的可视化
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import random
import gc
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.GCNSkeletonTokenizer import GCNSkeletonTokenizer
from utils.config import cfg_from_yaml_file
from utils.logger import print_log

def optimize_memory():
    """优化内存设置"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 限制CPU线程数
    os.environ['OMP_NUM_THREADS'] = '2'
    torch.set_num_threads(2)

class GCNSkeletonGifVisualizer:
    """GCN骨架重建GIF动画可视化器 - 内存优化版本"""
    
    def __init__(self, model_path, config_path):
        # 应用内存优化
        optimize_memory()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载配置
        self.config = cfg_from_yaml_file(config_path)
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # NTU RGB+D 25关节点连接关系
        self.skeleton_edges = [
            # 头部和脊柱连接
            (3, 2), (2, 20), (20, 1), (1, 0),
            # 左臂连接
            (20, 4), (4, 5), (5, 6), (6, 22), (6, 7), (7, 21),
            # 右臂连接
            (20, 8), (8, 9), (9, 10), (10, 24), (10, 11), (11, 23),
            # 左腿连接
            (0, 12), (12, 13), (13, 14), (14, 15),
            # 右腿连接
            (0, 16), (16, 17), (17, 18), (18, 19)
        ]
    
    def _get_joint_weights(self):
        """获取关节重要性权重"""
        weights = np.ones(25)
        # 重要关节权重x2 (与模型中的设置一致)
        head_joints = [3, 2]  # 头部
        hand_joints = [6, 7, 21, 22, 10, 11, 23, 24]  # 手部
        foot_joints = [14, 15, 18, 19]  # 脚部
        important_joints = head_joints + hand_joints + foot_joints
        weights[important_joints] = 2.0
        return weights
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        print(f"Loading model: {model_path}")

        # 创建模型实例
        model = GCNSkeletonTokenizer(
            num_tokens=640,  # 5个语义组 × 128个token
            token_dim=256,
            in_channels=64,
            hidden_channels=128
        ).to(self.device)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)

        # 检查checkpoint结构并加载正确的state_dict
        if 'base_model' in checkpoint:
            state_dict = checkpoint['base_model']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 处理DataParallel模型的module.前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 移除'module.'前缀
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict)

        model.eval()
        print("Model loaded successfully!")
        return model
    
    def normalize_skeleton(self, skeleton):
        """标准化骨架数据 - 与训练时保持一致"""
        # 使用与训练数据集相同的标准化方法
        centroid = np.mean(skeleton, axis=0)
        centered = skeleton - centroid
        
        # 使用最大距离进行缩放（与训练时一致）
        distances = np.sqrt(np.sum(centered**2, axis=1))
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
            
        return normalized
    
    def align_skeleton_orientation(self, skeleton):
        """对齐骨架方向，减少旋转导致的重建错误"""
        # 计算主要身体轴向（从骨盆到头部）
        # NTU RGB+D关节点索引：0=骨盆中心, 3=头顶
        if len(skeleton) >= 4:
            pelvis = skeleton[0]  # 骨盆中心
            head = skeleton[3]   # 头顶
            
            # 计算身体主轴
            body_axis = head - pelvis
            body_axis_norm = np.linalg.norm(body_axis)
            
            if body_axis_norm > 1e-6:
                # 将身体主轴对齐到Y轴正方向
                target_axis = np.array([0, 1, 0])
                body_axis_normalized = body_axis / body_axis_norm
                
                # 计算旋转角度
                cos_angle = np.dot(body_axis_normalized, target_axis)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
                # 如果身体轴向与Y轴相反（倒立），进行180度旋转
                if cos_angle < -0.5:  # 角度大于120度，认为是倒立
                    # 绕X轴旋转180度
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    skeleton = np.dot(skeleton, rotation_matrix.T)
        
        return skeleton
    
    def reconstruct_skeleton_sequence(self, skeleton_sequence):
        """重建骨架序列 - 增强版，返回更多诊断信息"""
        reconstructed_sequence = []
        normalized_sequence = []
        base_reconstructed_sequence = []  # 新增：基础重建（无残差）
        token_sequences = []  # 新增：token序列
        residual_info = []  # 新增：残差信息

        with torch.no_grad():
            for skeleton in skeleton_sequence:
                # skeleton是已经转换为(x,y,z)格式的可视化数据
                # 需要转换回(x,z,y)格式用于模型推理
                skeleton_for_model = skeleton[:, [0, 2, 1]]  # [x,y,z] -> [x,z,y]

                # 先对齐骨架方向，减少旋转导致的重建错误
                aligned = self.align_skeleton_orientation(skeleton_for_model)

                # 使用与训练时一致的标准化方法
                normalized = self.normalize_skeleton(aligned)
                skeleton_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)

                output = self.model(skeleton_tensor, return_recon=True)

                if 'reconstructed' in output:
                    reconstructed = output['reconstructed'].cpu().numpy()[0]
                    # 对重建的骨架进行坐标转换: (x,z,y) -> (x,y,z) 以匹配可视化
                    reconstructed = reconstructed[:, [0, 2, 1]]  # [x,z,y] -> [x,y,z]
                else:
                    # normalized是(x,z,y)格式，需要转换为(x,y,z)
                    reconstructed = normalized[:, [0, 2, 1]]

                # 获取基础重建（无残差）
                if 'base_reconstructed' in output:
                    base_recon = output['base_reconstructed'].cpu().numpy()[0]
                    base_recon = base_recon[:, [0, 2, 1]]  # [x,z,y] -> [x,y,z]
                else:
                    base_recon = reconstructed.copy()
                
                # 获取token序列
                if 'token_sequence' in output:
                    tokens = output['token_sequence'].cpu().numpy()[0]
                else:
                    tokens = None
                
                # 获取残差信息
                residual_scale = None
                if 'residual_scale' in output:
                    residual_scale = float(output['residual_scale'].cpu().numpy())

                # 返回的normalized也需要转换为可视化格式
                normalized_for_vis = normalized[:, [0, 2, 1]]

                reconstructed_sequence.append(reconstructed)
                normalized_sequence.append(normalized_for_vis)
                base_reconstructed_sequence.append(base_recon)
                token_sequences.append(tokens)
                residual_info.append(residual_scale)

        return {
            'reconstructed': reconstructed_sequence,
            'normalized': normalized_sequence,
            'base_reconstructed': base_reconstructed_sequence,
            'tokens': token_sequences,
            'residual_scale': residual_info[0] if residual_info else None
        }
    
    def read_skeleton_file(self, filepath):
        """读取.skeleton文件，返回多帧数据"""
        try:
            with open(filepath, 'r') as f:
                # 读取帧数
                frame_count = int(f.readline().strip())
                
                frames_data = []
                for frame_idx in range(frame_count):
                    # 读取人体数量
                    body_count = int(f.readline().strip())
                    
                    frame_skeletons = []
                    for body_idx in range(body_count):
                        # 读取人体信息行
                        body_info = f.readline().strip()
                        
                        # 读取关节数量
                        joint_count = int(f.readline().strip())
                        
                        # 读取关节数据
                        joints = []
                        for joint_idx in range(joint_count):
                            joint_line = f.readline().strip().split()
                            if len(joint_line) >= 3:
                                # NTU RGB+D坐标转换: (x,z,y) -> (x,y,z) 仅用于可视化
                                x, z, y = float(joint_line[0]), float(joint_line[1]), float(joint_line[2])
                                joints.append([x, y, z])
                        
                        if len(joints) == 25:  # NTU有25个关节
                            frame_skeletons.append(np.array(joints))
                    
                    if frame_skeletons:
                        # 如果有多个人体，选择第一个
                        frames_data.append(frame_skeletons[0])
                
                return frames_data if frames_data else None
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
    
    def plot_skeleton_3d(self, ax, skeleton, title, color='blue', show_joint_weights=False):
        """在3D子图中绘制骨架，支持关节权重可视化"""
        ax.clear()
        
        # 绘制骨骼连接（先绘制连接线，这样关节点会在上层）
        for edge in self.skeleton_edges:
            if edge[0] < len(skeleton) and edge[1] < len(skeleton):
                start = skeleton[edge[0]]
                end = skeleton[edge[1]]
                # 只绘制非零关节的连接
                if not (np.all(start == 0) or np.all(end == 0)):
                    # 根据连接的关节权重调整线条样式
                    if show_joint_weights and hasattr(self, 'joint_weights'):
                        weight = max(self.joint_weights[edge[0]], self.joint_weights[edge[1]])
                        if weight > 1.0:  # 连接重要关节
                            line_color = 'red'
                            line_width = 5.0
                            alpha = 0.9
                        else:
                            line_color = color
                            line_width = 4.0
                            alpha = 0.8
                    else:
                        line_color = color
                        line_width = 4.0
                        alpha = 0.8
                    
                    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                             color=line_color, alpha=alpha, linewidth=line_width)
        
        # 绘制关节点（在连接线之后绘制，确保在上层）
        if show_joint_weights and hasattr(self, 'joint_weights'):
            # 根据关节权重分别绘制
            important_indices = np.where(self.joint_weights > 1.0)[0]
            normal_indices = np.where(self.joint_weights == 1.0)[0]
            
            # 绘制普通关节
            if len(normal_indices) > 0:
                ax.scatter(skeleton[normal_indices, 0], skeleton[normal_indices, 1], skeleton[normal_indices, 2], 
                          c=color, s=25, alpha=0.9, edgecolors='white', linewidth=0.5)
            
            # 绘制重要关节（更大、更显眼）
            if len(important_indices) > 0:
                ax.scatter(skeleton[important_indices, 0], skeleton[important_indices, 1], skeleton[important_indices, 2], 
                          c='red', s=50, alpha=1.0, edgecolors='black', linewidth=1.0, marker='o')
        else:
            # 标准绘制
            ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], 
                      c=color, s=25, alpha=0.9, edgecolors='white', linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        
        # 设置固定的显示范围以保持动画稳定
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([-0.6, 0.6])
        
        # 设置网格和背景
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # 设置坐标轴线的颜色和透明度
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # 设置更好的观察视角，确保骨架看起来是站立的
        # elev: 仰角（垂直角度），azim: 方位角（水平角度）
        ax.view_init(elev=15, azim=45)  # 稍微从上往下看，45度角观察

    def _get_test_set_files(self, data_dir):
        """获取测试集文件列表，使用与训练时相同的分割逻辑"""
        # 获取所有skeleton文件
        all_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.skeleton'):
                all_files.append(file)

        # 过滤有效动作（与训练时一致）
        excluded_actions = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                           106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                           117, 118, 119, 120]

        valid_files = []
        for file in all_files:
            action_id = self._parse_action_id(file)
            if action_id not in excluded_actions:
                valid_files.append(file)

        # 使用与训练时相同的数据分割逻辑
        np.random.seed(42)  # 与训练时相同的随机种子

        # 按动作类别进行分层抽样
        files_by_action = {}
        for file in valid_files:
            action_id = self._parse_action_id(file)
            if action_id not in files_by_action:
                files_by_action[action_id] = []
            files_by_action[action_id].append(file)

        test_files = []
        train_ratio, val_ratio = 0.8, 0.15  # 与配置文件一致

        for action_id, action_files in files_by_action.items():
            n_files = len(action_files)
            if n_files == 0:
                continue

            # 计算分割点
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)

            # 随机打乱文件（使用相同种子确保一致性）
            np.random.shuffle(action_files)

            # 获取测试集文件
            test_files.extend(action_files[n_train+n_val:])

        return test_files

    def _parse_action_id(self, filename):
        """从文件名解析动作ID"""
        try:
            action_part = filename.split('A')[1].split('.')[0]
            return int(action_part)
        except:
            return 0

    def create_skeleton_gif(self, recon_results, sample_name, save_path, fps=10, show_joint_weights=True, show_codebook_only=False):
        """创建骨架重建对比的GIF动画 - 增强版，可展示纯码本重建
        
        Args:
            recon_results: reconstruct_skeleton_sequence返回的字典
            show_codebook_only: 是否展示纯码本重建（不含残差）
        """
        original_sequence = recon_results['normalized']
        reconstructed_sequence = recon_results['reconstructed']
        base_reconstructed_sequence = recon_results['base_reconstructed']
        residual_scale = recon_results['residual_scale']
        
        num_frames = min(len(original_sequence), len(reconstructed_sequence))
        
        # 预计算所有帧的损失信息以提升性能
        frame_losses = []
        total_mse = 0.0
        weighted_mse = 0.0
        base_mse = 0.0  # 纯码本重建的MSE
        
        for i in range(num_frames):
            # 计算最终重建的MSE误差
            mse_error = np.mean((original_sequence[i] - reconstructed_sequence[i]) ** 2)
            
            # 计算纯码本重建的MSE误差
            base_mse_error = np.mean((original_sequence[i] - base_reconstructed_sequence[i]) ** 2)
            
            # 计算加权MSE误差（基于关节重要性）
            if hasattr(self, 'joint_weights'):
                joint_errors = np.mean((original_sequence[i] - reconstructed_sequence[i]) ** 2, axis=1)
                weighted_error = np.mean(joint_errors * self.joint_weights)
            else:
                weighted_error = mse_error
            
            frame_losses.append({
                'mse': mse_error,
                'weighted_mse': weighted_error,
                'base_mse': base_mse_error
            })
            total_mse += mse_error
            weighted_mse += weighted_error
            base_mse += base_mse_error
        
        avg_mse = total_mse / num_frames
        avg_weighted_mse = weighted_mse / num_frames
        avg_base_mse = base_mse / num_frames

        # 创建图形 - 根据模式选择布局
        if show_codebook_only:
            # 3列布局：原始 | 纯码本重建 | 最终重建（含残差）
            fig = plt.figure(figsize=(24, 8))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')
            
            # 设置总标题
            title_text = f'GCN Skeleton Reconstruction Analysis - {sample_name}\n'
            title_text += f'Codebook Only MSE: {avg_base_mse:.4f} | '
            title_text += f'Final MSE (w/ Residual): {avg_mse:.4f} | '
            title_text += f'Residual Scale: {residual_scale:.4f}' if residual_scale else ''
            if show_joint_weights:
                title_text += '\nRed Joints: Important (2x weight)'
        else:
            # 2列布局：原始 | 最终重建
            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            
            # 设置总标题
            title_text = f'GCN Skeleton Reconstruction - {sample_name}\n'
            title_text += f'Avg MSE: {avg_mse:.4f} | Weighted MSE: {avg_weighted_mse:.4f}'
            if residual_scale is not None:
                title_text += f' | Residual Scale: {residual_scale:.4f}'
            if show_joint_weights:
                title_text += '\nRed: Important Joints (2x weight)'
        
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.95)

        def animate(frame_idx):
            """动画更新函数 - 增强版"""
            try:
                # 获取预计算的损失
                losses = frame_losses[frame_idx]
                
                # 绘制原始骨架
                self.plot_skeleton_3d(ax1, original_sequence[frame_idx],
                                    f'Frame {frame_idx+1}/{num_frames} - Original', 
                                    'dodgerblue', show_joint_weights=show_joint_weights)

                if show_codebook_only:
                    # 绘制纯码本重建
                    base_title = f'Frame {frame_idx+1}/{num_frames} - Codebook Only\n' + \
                                f'MSE: {losses["base_mse"]:.4f}'
                    self.plot_skeleton_3d(ax2, base_reconstructed_sequence[frame_idx],
                                        base_title, 'orange', show_joint_weights=show_joint_weights)
                    
                    # 绘制最终重建（含残差）
                    final_title = f'Frame {frame_idx+1}/{num_frames} - Final (w/ Residual)\n' + \
                                 f'MSE: {losses["mse"]:.4f} | Weighted: {losses["weighted_mse"]:.4f}'
                    self.plot_skeleton_3d(ax3, reconstructed_sequence[frame_idx],
                                        final_title, 'crimson', show_joint_weights=show_joint_weights)
                    return ax1, ax2, ax3
                else:
                    # 绘制最终重建骨架，显示损失信息
                    recon_title = f'Frame {frame_idx+1}/{num_frames} - Reconstructed\n' + \
                                 f'MSE: {losses["mse"]:.4f} | Weighted: {losses["weighted_mse"]:.4f}'
                    self.plot_skeleton_3d(ax2, reconstructed_sequence[frame_idx],
                                        recon_title, 'crimson', show_joint_weights=show_joint_weights)
                    return ax1, ax2
            except Exception as e:
                print(f"Error in animation frame {frame_idx}: {e}")
                if show_codebook_only:
                    return ax1, ax2, ax3
                else:
                    return ax1, ax2

        # 创建动画
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, blit=False, repeat=True)

        # 保存为GIF - 内存优化
        try:
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=120)  # 降低DPI节省内存
            plt.close()
            
            # 强制内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"✅ Saved GIF animation: {save_path}")
            if show_codebook_only:
                print(f"   Codebook Only MSE: {avg_base_mse:.4f}")
                print(f"   Final MSE (w/ Residual): {avg_mse:.4f}")
                print(f"   Residual Contribution: {(1 - avg_mse/avg_base_mse)*100:.1f}% improvement")
            else:
                print(f"   Average MSE: {avg_mse:.4f}, Weighted MSE: {avg_weighted_mse:.4f}")
            return True
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")
            plt.close()
            return False

    def process_sample_to_gif(self, skeleton_file, save_dir, max_frames=30, fps=8, show_codebook_only=False):
        """处理单个样本生成GIF
        
        Args:
            show_codebook_only: 是否生成包含纯码本重建的三列对比GIF
        """
        # 读取骨架序列
        frames_data = self.read_skeleton_file(skeleton_file)
        if frames_data is None or len(frames_data) == 0:
            print(f"Failed to parse {skeleton_file}")
            return False

        # 限制帧数以控制GIF大小
        if len(frames_data) > max_frames:
            # 均匀采样
            indices = np.linspace(0, len(frames_data)-1, max_frames, dtype=int)
            frames_data = [frames_data[i] for i in indices]

        # 重建骨架序列（返回包含base_reconstructed的字典）
        recon_results = self.reconstruct_skeleton_sequence(frames_data)

        # 生成GIF文件名
        sample_name = os.path.splitext(os.path.basename(skeleton_file))[0]
        if show_codebook_only:
            gif_path = os.path.join(save_dir, f'gcn_codebook_analysis_{sample_name}.gif')
        else:
            gif_path = os.path.join(save_dir, f'gcn_animation_{sample_name}.gif')

        # 创建GIF动画
        self.create_skeleton_gif(recon_results, sample_name, gif_path, fps, 
                                show_joint_weights=True, show_codebook_only=show_codebook_only)

        return True

    def process_samples_to_gifs(self, data_dir, save_dir, num_samples=5, max_frames=30, fps=8, 
                               use_test_set_only=True, show_codebook_only=False):
        """处理多个样本生成GIF动画
        
        Args:
            show_codebook_only: 是否生成包含纯码本重建的三列对比GIF
        """
        print(f"Starting GCN skeleton reconstruction GIF generation...")
        print(f"Data: {data_dir}")
        print(f"Output: {save_dir}")
        print(f"Max frames per GIF: {max_frames}, FPS: {fps}")
        print(f"Mode: {'Codebook Analysis (3-column)' if show_codebook_only else 'Standard Reconstruction (2-column)'}")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        if use_test_set_only:
            # 使用与训练时相同的数据分割逻辑，只选择测试集样本
            test_files = self._get_test_set_files(data_dir)
            print(f"Found {len(test_files)} test set skeleton files")

            if not test_files:
                print("No test set skeleton files found")
                return

            # 从测试集中随机选择样本
            selected_files = [os.path.join(data_dir, f) for f in test_files]
            random.seed(42)  # 固定随机种子以获得可重复的结果
            selected_files = random.sample(selected_files, min(num_samples, len(selected_files)))
            print(f"Using TEST SET ONLY: {len(selected_files)} samples selected from test set")
        else:
            # 原始方法：从整个数据集随机选择（不推荐用于评估）
            skeleton_files = []
            for file in os.listdir(data_dir):
                if file.endswith('.skeleton'):
                    skeleton_files.append(os.path.join(data_dir, file))

            print(f"Found {len(skeleton_files)} skeleton files")

            # 随机选择样本
            random.seed(42)  # 固定随机种子以获得可重复的结果
            selected_files = random.sample(skeleton_files, min(num_samples, len(skeleton_files)))
            print(f"WARNING: Using samples from ENTIRE dataset (may include training data)")

        # 处理每个样本
        success_count = 0
        for i, skeleton_file in enumerate(selected_files):
            print(f"Processing sample {i+1}/{len(selected_files)}: {os.path.basename(skeleton_file)}")

            if self.process_sample_to_gif(skeleton_file, save_dir, max_frames, fps, show_codebook_only):
                success_count += 1

        print(f"\nGIF generation completed!")
        print(f"Successfully generated {success_count}/{len(selected_files)} GIF animations")
        print(f"All GIFs saved to: {save_dir}")


def main():
    """主函数"""
    # 配置参数
    model_path = '/home/uo/myProject/CRSkeleton/experiments/gcn_skeleton_memory_optimized/NTU_models/default/ckpt-best.pth'
    config_path = '/home/uo/myProject/CRSkeleton/cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml'
    data_dir = '/home/uo/myProject/HumanPoint-BERT/data/NTU-RGB+D'
    
    # 创建可视化器
    visualizer = GCNSkeletonGifVisualizer(model_path, config_path)

    # 模式1：标准重建可视化（2列布局：原始 vs 重建）
    print("\n" + "="*80)
    print("MODE 1: Standard Reconstruction Visualization")
    print("="*80)
    save_dir_standard = '/home/uo/myProject/CRSkeleton/visualizations/1_gcn/gif_standard_reconstruction'
    visualizer.process_samples_to_gifs(
        data_dir=data_dir,
        save_dir=save_dir_standard,
        num_samples=5,          # 生成5个样本的GIF
        max_frames=20,          # 每个GIF最多20帧
        fps=6,                  # 6帧每秒
        use_test_set_only=True, # 只使用测试集样本
        show_codebook_only=False  # 标准2列布局
    )

    # 模式2：码本分析可视化（3列布局：原始 vs 纯码本重建 vs 最终重建）
    print("\n" + "="*80)
    print("MODE 2: Codebook Analysis Visualization")
    print("="*80)
    save_dir_codebook = '/home/uo/myProject/CRSkeleton/visualizations/1_gcn/gif_codebook_analysis'
    visualizer.process_samples_to_gifs(
        data_dir=data_dir,
        save_dir=save_dir_codebook,
        num_samples=5,          # 生成5个样本的GIF
        max_frames=20,          # 每个GIF最多20帧
        fps=6,                  # 6帧每秒
        use_test_set_only=True, # 只使用测试集样本
        show_codebook_only=True   # 3列布局，展示码本重构效果
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GCN骨架重建可视化工具
简化版本：显示原始骨架和重建骨架的多帧对比，每个样本生成一个PNG文件
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import random
from pathlib import Path

# 设置matplotlib支持，避免字体问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from models.GCNSkeletonTokenizer import GCNSkeletonTokenizer
    from utils.config import cfg_from_yaml_file
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class SimpleGCNVisualizer:
    """简化的GCN骨架可视化器"""
    
    def __init__(self, model_path, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path, config_path)
        
        # NTU RGB+D 25关节点连接关系
        self.skeleton_edges = [
            (3, 2), (2, 20), (20, 1), (1, 0),  # 头部和脊柱
            (20, 4), (4, 5), (5, 6), (6, 22), (6, 7), (7, 21),  # 左臂
            (20, 8), (8, 9), (9, 10), (10, 24), (10, 11), (11, 23),  # 右臂
            (0, 12), (12, 13), (13, 14), (14, 15),  # 左腿
            (0, 16), (16, 17), (17, 18), (18, 19)   # 右腿
        ]
        
    def _load_model(self, model_path, config_path):
        """加载训练好的模型"""
        print(f"Loading model: {model_path}")
        
        # 加载配置
        config = cfg_from_yaml_file(config_path)
        
        # 创建模型
        model = GCNSkeletonTokenizer(config.model)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'base_model' in checkpoint:
            state_dict = checkpoint['base_model']
        else:
            state_dict = checkpoint
            
        # 移除module.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
        
    def parse_skeleton_file(self, file_path, max_frames=6):
        """解析skeleton文件，提取多帧数据"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 1:
                return None
                
            num_frames = int(lines[0].strip())
            if num_frames == 0:
                return None
                
            frames = []
            line_idx = 1
            
            # 提取前max_frames帧或所有帧（取较小值）
            frames_to_extract = min(max_frames, num_frames)
            
            for frame_i in range(frames_to_extract):
                if line_idx >= len(lines):
                    break
                    
                # 读取人体数量
                num_bodies = int(lines[line_idx].strip())
                line_idx += 1
                
                if num_bodies == 0:
                    # 跳过空帧
                    continue
                    
                # 读取第一个人体的数据
                body_info = lines[line_idx].strip().split()
                line_idx += 1
                
                # 读取关节数量
                num_joints = int(lines[line_idx].strip())
                line_idx += 1
                
                if num_joints != 25:
                    # 跳过不符合要求的帧
                    for j in range(num_joints):
                        line_idx += 1
                    continue
                    
                # 读取关节数据
                joints = []
                for j in range(num_joints):
                    if line_idx >= len(lines):
                        break
                    joint_data = lines[line_idx].strip().split()
                    line_idx += 1
                    
                    # NTU RGB+D坐标转换: (x,z,y) -> (x,y,z) 仅用于可视化
                    x, z, y = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                    joints.append([x, y, z])
                    
                if len(joints) == 25:
                    frames.append(np.array(joints, dtype=np.float32))
                    
            return frames if len(frames) > 0 else None
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
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
    
    def reconstruct_skeleton(self, skeleton):
        """重建单帧骨架"""
        with torch.no_grad():
            # skeleton是已经转换为(x,y,z)格式的可视化数据
            # 需要转换回(x,z,y)格式用于模型推理
            skeleton_for_model = skeleton[:, [0, 2, 1]]  # [x,y,z] -> [x,z,y]

            # 先对齐骨架方向，减少旋转导致的重建错误
            aligned = self.align_skeleton_orientation(skeleton_for_model)

            # 使用与训练时一致的标准化方法
            normalized = self.normalize_skeleton(aligned)
            skeleton_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

            output = self.model(skeleton_tensor, return_recon=True)

            if 'reconstructed' in output:
                reconstructed = output['reconstructed'].cpu().numpy()[0]
                # 对重建的骨架进行坐标转换: (x,z,y) -> (x,y,z) 以匹配可视化
                reconstructed = reconstructed[:, [0, 2, 1]]  # [x,z,y] -> [x,y,z]
            else:
                # normalized是(x,z,y)格式，需要转换为(x,y,z)
                reconstructed = normalized[:, [0, 2, 1]]

            # 返回的normalized也需要转换为可视化格式
            normalized_for_vis = normalized[:, [0, 2, 1]]
            return reconstructed, normalized_for_vis
    
    def plot_skeleton_3d(self, ax, skeleton, title, color='blue'):
        """在3D子图中绘制骨架"""
        # 绘制骨骼连接（先绘制连接线，这样关节点会在上层）
        for edge in self.skeleton_edges:
            if edge[0] < len(skeleton) and edge[1] < len(skeleton):
                start = skeleton[edge[0]]
                end = skeleton[edge[1]]
                # 只绘制非零关节的连接
                if not (np.all(start == 0) or np.all(end == 0)):
                    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                             color=color, alpha=0.8, linewidth=4.0)  # 增加线宽

        # 绘制关节点（在连接线之后绘制，确保在上层）
        ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
                  c=color, s=25, alpha=0.9, edgecolors='white', linewidth=0.5)  # 减小关节点，添加白色边框

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        # 计算骨架的实际范围并设置合适的显示范围
        valid_joints = skeleton[~np.all(skeleton == 0, axis=1)]
        if len(valid_joints) > 0:
            # 计算骨架的边界
            x_range = valid_joints[:, 0].max() - valid_joints[:, 0].min()
            y_range = valid_joints[:, 1].max() - valid_joints[:, 1].min()
            z_range = valid_joints[:, 2].max() - valid_joints[:, 2].min()

            # 使用更小的范围倍数，让骨架显示更大
            max_range = max(x_range, y_range, z_range) * 0.6  # 从0.75减少到0.6
            if max_range < 0.08:  # 减小最小范围阈值
                max_range = 0.3

            # 计算中心点
            x_center = valid_joints[:, 0].mean()
            y_center = valid_joints[:, 1].mean()
            z_center = valid_joints[:, 2].mean()

            # 设置坐标范围以骨架为中心，给Y轴（垂直方向）更多空间
            ax.set_xlim([x_center - max_range, x_center + max_range])
            ax.set_ylim([y_center - max_range*1.2, y_center + max_range*1.2])  # Y轴稍微放大
            ax.set_zlim([z_center - max_range, z_center + max_range])

            # 确保坐标轴比例相等，避免骨架变形
            ax.set_box_aspect([1, 1.2, 1])  # Y轴稍微拉长
        else:
            # 如果没有有效关节，使用更小的默认范围
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([-0.3, 0.3])

        # 设置刻度标签
        ax.tick_params(labelsize=8)

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
    
    def visualize_sample(self, frames, sample_name, save_path):
        """可视化单个样本的多帧数据"""
        num_frames = len(frames)

        # 创建图形 - 2行显示，每行显示多帧，增大图像尺寸
        fig = plt.figure(figsize=(4.5 * num_frames, 12))  # 增大图像尺寸

        for i, frame in enumerate(frames):
            # 重建当前帧
            reconstructed, normalized_original = self.reconstruct_skeleton(frame)

            # 计算重建误差
            mse_error = np.mean((normalized_original - reconstructed) ** 2)

            # 原始骨架 (上排)
            ax1 = fig.add_subplot(2, num_frames, i + 1, projection='3d')
            self.plot_skeleton_3d(ax1, normalized_original, f'Frame {i+1} - Original', 'dodgerblue')

            # 重建骨架 (下排)
            ax2 = fig.add_subplot(2, num_frames, i + 1 + num_frames, projection='3d')
            self.plot_skeleton_3d(ax2, reconstructed, f'Frame {i+1} - Reconstructed\\nMSE: {mse_error:.4f}', 'crimson')

        plt.suptitle(f'GCN Skeleton Reconstruction - {sample_name}', fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # 为标题留出空间

        # 保存图像，提高DPI以获得更清晰的图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')  # 提高DPI，设置白色背景
        plt.close()

        print(f"Saved visualization: {save_path}")
    
    def process_samples(self, data_dir, save_dir, num_samples=15, use_test_set_only=True):
        """处理多个样本"""
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
            selected_files = random.sample(selected_files, min(num_samples, len(selected_files)))
            print(f"Using TEST SET ONLY: {len(selected_files)} samples selected from test set")
        else:
            # 原始方法：从整个数据集随机选择（不推荐用于评估）
            skeleton_files = glob.glob(os.path.join(data_dir, "**/*.skeleton"), recursive=True)
            print(f"Found {len(skeleton_files)} skeleton files")

            if not skeleton_files:
                print("No skeleton files found")
                return

            # 随机选择样本
            selected_files = random.sample(skeleton_files, min(num_samples, len(skeleton_files)))
            print(f"WARNING: Using samples from ENTIRE dataset (may include training data)")
        
        for i, file_path in enumerate(selected_files):
            print(f"Processing sample {i+1}/{len(selected_files)}: {os.path.basename(file_path)}")
            
            # 解析多帧数据
            frames = self.parse_skeleton_file(file_path, max_frames=6)
            if frames is None or len(frames) == 0:
                print(f"Failed to parse {file_path}")
                continue
            
            # 生成样本名称
            sample_name = os.path.basename(file_path).replace('.skeleton', '')
            
            # 生成保存路径
            save_path = os.path.join(save_dir, f'gcn_reconstruction_sample_{i+1}_{sample_name}.png')
            
            # 可视化
            self.visualize_sample(frames, sample_name, save_path)
        
        print(f"\\nAll visualizations saved to: {save_dir}")

def main():
    # 配置参数
    model_path = "/home/uo/myProject/CRSkeleton/experiments/gcn_skeleton_memory_optimized/NTU_models/default/ckpt-best.pth"
    config_path = "/home/uo/myProject/CRSkeleton/cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml"
    data_dir = "/home/uo/myProject/HumanPoint-BERT/data/NTU-RGB+D"
    save_dir = "/home/uo/myProject/CRSkeleton/visualizations/1_gcn/results_oltest_100"
    
    # 设置随机种子
    random.seed(42)
    
    # 检查文件存在性
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
        
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
        
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Starting GCN skeleton reconstruction visualization...")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {save_dir}")
    
    # 创建可视化器并处理样本
    try:
        visualizer = SimpleGCNVisualizer(model_path, config_path)
        # 使用测试集样本进行可视化（确保评估公正性）
        visualizer.process_samples(data_dir, save_dir, num_samples=15, use_test_set_only=True)
        print("\\nVisualization completed successfully!")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

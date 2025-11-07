#!/usr/bin/env python3
"""
独立可视化窗口模块
为标注工具提供3D可视化支持
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import time
import matplotlib.font_manager as fm

# 配置matplotlib避免字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from current font.*')

# 设置matplotlib参数
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class VisualizationWindow:
    """独立的可视化窗口类"""
    
    def __init__(self):
        self.window = None
        self.figure = None
        self.canvas = None
        self.current_data = None
        self.data_type = None
        self.sample_info = {}
        self.dataset_type = "auto"  # auto, mars, ntu
        
        # MARS数据集：Microsoft Kinect 19关节点骨架连接定义 
        # 参考vis_gif_skeleton_extractor.py中的正确连接方式
        self.mars_skeleton_connections = [
            (2, 3),   # head-neck
            (2, 18),  # neck-spineshoulder
            (18, 4),  # spineshoulder-leftshoulder
            (4, 5),   # leftshoulder-leftelbow
            (5, 6),   # leftelbow-leftwrist
            (18, 7),  # spineshoulder-rightshoulder
            (7, 8),   # rightshoulder-rightelbow
            (8, 9),   # rightelbow-rightwrist
            (18, 1),  # spineshoulder-spinemid
            (1, 0),   # spinemid-spinebase
            (0, 10),  # spinebase-hipleft
            (10, 11), # hipleft-kneeleft
            (11, 12), # kneeleft-ankleleft
            (12, 13), # ankleleft-footleft
            (0, 14),  # spinebase-hipright
            (14, 15), # hipright-kneeright
            (15, 16), # kneeright-ankleright
            (16, 17)  # ankleright-footright
        ]
        
        # NTU RGB+D数据集：25关节点连接关系
        # 参考 tools/analyze_ntu_skeleton.py 的标准定义
        self.ntu_skeleton_connections = [
            # 躯干和头部
            (3, 2),   # 头顶 - 颈部
            (2, 20),  # 颈部 - 上躯干
            (20, 1),  # 上躯干 - 躯干中
            (1, 0),   # 躯干中 - 躯干下
            
            # 左上肢
            (20, 4),  # 上躯干 - 左肩
            (4, 5),   # 左肩 - 左肘
            (5, 6),   # 左肘 - 左腕
            (6, 22),  # 左腕 - 左手指1
            (6, 7),   # 左腕 - 左手
            (7, 21),  # 左手 - 左手指2
            
            # 右上肢
            (20, 8),  # 上躯干 - 右肩
            (8, 9),   # 右肩 - 右肘
            (9, 10),  # 右肘 - 右腕
            (10, 24), # 右腕 - 右手指1
            (10, 11), # 右腕 - 右手
            (11, 23), # 右手 - 右手指2
            
            # 左下肢
            (0, 12),  # 躯干下 - 左髋
            (12, 13), # 左髋 - 左膝
            (13, 14), # 左膝 - 左踝
            (14, 15), # 左踝 - 左脚
            
            # 右下肢
            (0, 16),  # 躯干下 - 右髋
            (16, 17), # 右髋 - 右膝
            (17, 18), # 右膝 - 右踝
            (18, 19), # 右踝 - 右脚
        ]
        
        # 🐛 调试输出：确认加载的连接定义
        print(f"🐛 [VisualizationWindow] NTU骨架连接数: {len(self.ntu_skeleton_connections)}")
        print(f"🐛 [VisualizationWindow] 前3个连接: {self.ntu_skeleton_connections[:3]}")
        print(f"🐛 [VisualizationWindow] 后3个连接: {self.ntu_skeleton_connections[-3:]}")
        
    def create_window(self):
        """创建可视化窗口"""
        if self.window is not None:
            try:
                self.window.destroy()
            except:
                pass
            self.window = None
            
        # 使用独立的Tk窗口而不是Toplevel
        self.window = tk.Tk()
        self.window.title("Skeleton Data Visualization")
        self.window.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="Sample Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=4, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X)
        
        # 可视化区域
        viz_frame = ttk.LabelFrame(main_frame, text="3D Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(control_frame, text="Refresh View", command=self.refresh_view).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Close Window", command=self.close_window).pack(side=tk.RIGHT)
        
        # 保持窗口在前台
        self.window.attributes('-topmost', True)
        self.window.focus_set()
        
    def show_sample(self, sample_data, sample_info=None):
        """显示样本数据"""
        try:
            if self.window is None or not self.window.winfo_exists():
                self.create_window()
        except tk.TclError:
            # 窗口已被销毁，重新创建
            self.window = None
            self.create_window()
            
        self.current_data = sample_data
        self.sample_info = sample_info or {}
        
        # 更新信息显示
        self.update_info_display()
        
        # 获取文件路径用于数据集类型检测
        file_path = self.sample_info.get('filename') or self.sample_info.get('file_path')
        
        # 判断数据类型并可视化
        if 'point_cloud_data' in sample_data:
            self.visualize_point_cloud(sample_data['point_cloud_data'])
        elif 'radar_data' in sample_data:
            self.visualize_radar_data(sample_data['radar_data'])
        elif 'skeleton_data' in sample_data:
            self.visualize_skeleton_data(sample_data['skeleton_data'], file_path)
        elif 'extracted' in sample_data or 'reconstructed' in sample_data:
            # MARS Token 数据集格式: extracted/reconstructed 骨架
            skeleton = sample_data.get('reconstructed', sample_data.get('extracted'))
            self.visualize_skeleton_data(skeleton, file_path)
        else:
            self.show_placeholder()
            
    def update_info_display(self):
        """更新信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        info_lines = []
        info_lines.append(f"File: {self.sample_info.get('filename', 'unknown')}")
        info_lines.append(f"Tokens: {self.sample_info.get('tokens', [])}")
        
        if 'ground_truth_action' in self.sample_info:
            info_lines.append(f"Ground Truth: {self.sample_info['ground_truth_action']}")
            
        if 'source' in self.sample_info:
            info_lines.append(f"Source: {self.sample_info['source']}")
            
        self.info_text.insert(1.0, '\n'.join(info_lines))
        self.info_text.config(state=tk.DISABLED)
        
    def visualize_point_cloud(self, point_cloud_data):
        """可视化点云数据"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        # 基本点云显示 - 交换y和z轴，使z轴成为竖直方向
        x, y, z = point_cloud_data[:, 0], point_cloud_data[:, 2], point_cloud_data[:, 1]
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
        
        # 添加颜色条
        self.figure.colorbar(scatter, ax=ax, shrink=0.8)
        
        # 分区域显示（5个身体部位）
        points_per_part = len(point_cloud_data) // 5
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        labels = ['Head&Neck', 'Left Arm', 'Right Arm', 'Left Leg', 'Right Leg']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            start_idx = i * points_per_part
            end_idx = start_idx + points_per_part if i < 4 else len(point_cloud_data)
            part_points = point_cloud_data[start_idx:end_idx]
            
            if len(part_points) > 0:
                # 计算中心点 - 交换y和z坐标
                center_orig = np.mean(part_points, axis=0)
                center = [center_orig[0], center_orig[2], center_orig[1]]  # 交换y和z
                ax.scatter(*center, c=color, s=100, marker='o', 
                          label=f'{label} (Center)', edgecolors='black', linewidth=1)
        
        ax.set_xlabel('X (Horizontal)')
        ax.set_ylabel('Y (Depth)') 
        ax.set_zlabel('Z (Vertical)')
        ax.set_title(f'Point Cloud Visualization ({len(point_cloud_data)} points) - Z-axis Vertical')
        ax.legend()
        
        # 设置相等的坐标轴比例
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.canvas.draw()
        
    def visualize_radar_data(self, radar_data):
        """可视化雷达数据"""
        self.figure.clear()
        
        if len(radar_data.shape) == 3 and radar_data.shape[2] == 5:
            # 多通道雷达数据 (8, 8, 5)
            for i in range(5):
                ax = self.figure.add_subplot(2, 3, i+1)
                im = ax.imshow(radar_data[:, :, i], cmap='jet', aspect='auto')
                ax.set_title(f'通道 {i+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                self.figure.colorbar(im, ax=ax)
        else:
            # 单通道或其他格式
            ax = self.figure.add_subplot(111)
            if len(radar_data.shape) == 2:
                im = ax.imshow(radar_data, cmap='jet', aspect='auto')
                self.figure.colorbar(im, ax=ax)
            else:
                # 1D数据显示为波形
                ax.plot(radar_data.flatten())
                ax.set_title('雷达信号')
                ax.set_xlabel('采样点')
                ax.set_ylabel('幅值')
        
        self.figure.suptitle('MARS雷达特征图可视化\n📡 这是雷达传感器的5通道特征数据，不是骨架数据\n💡 MARS数据集记录的是雷达信号，需要通过AI模型转换为骨架', fontsize=10)
        self.canvas.draw()
        
    def detect_dataset_type(self, skeleton_data, file_path=None, sample_info=None):
        """自动检测数据集类型"""
        if self.dataset_type != "auto":
            return self.dataset_type
        
        # 🔥 优先基于关节点数量判断（最可靠的依据）
        # MARS_recon_tokens虽然source='mars_tokens'，但实际是25关节的NTU数据！
        if len(skeleton_data) == 19:
            print("🐛 [DatasetDetect] 19关节 → MARS")
            return "mars"
        elif len(skeleton_data) == 25:
            print("🐛 [DatasetDetect] 25关节 → NTU")
            return "ntu"
        
        # 其次检查样本信息中的source字段
        if sample_info:
            source = sample_info.get('source', '').lower()
            if 'ntu' in source:
                print("🐛 [DatasetDetect] source含'ntu' → NTU")
                return "ntu"
            elif 'mars' in source and 'token' not in source:
                # 排除 mars_tokens（它是NTU数据）
                print("🐛 [DatasetDetect] source含'mars'(非token) → MARS")
                return "mars"
        
        # 基于文件路径判断
        if file_path:
            path_lower = file_path.lower()
            if 'ntu' in path_lower or 'nturgbd' in path_lower:
                print("🐛 [DatasetDetect] 路径含'ntu' → NTU")
                return "ntu"
            elif 'mars' in path_lower and 'token' not in path_lower:
                print("🐛 [DatasetDetect] 路径含'mars'(非token) → MARS")
                return "mars"
        
        # 基于文件名模式判断（NTU文件名通常包含A[action_id]）
        if file_path:
            import re
            # NTU文件名模式：S001C001P001R001A001.skeleton
            if re.search(r'S\d+C\d+P\d+R\d+A\d+', file_path):
                print("🐛 [DatasetDetect] 文件名NTU模式 → NTU")
                return "ntu"
        
        # 默认使用MARS（但实际上前面的关节数量判断应该已经处理了）
        print("🐛 [DatasetDetect] 默认 → MARS")
        return "mars"
    
    def get_skeleton_connections(self, dataset_type):
        """获取对应数据集的骨架连接关系"""
        if dataset_type == "ntu":
            return self.ntu_skeleton_connections
        else:  # mars 或其他
            return self.mars_skeleton_connections
    
    def normalize_skeleton_for_mars(self, skeleton_data):
        """MARS数据集的骨架标准化（参考vis_gif_skeleton_extractor.py）"""
        # 使用与训练时一致的标准化方法
        centroid = np.mean(skeleton_data, axis=0)
        centered = skeleton_data - centroid
        
        # 使用最大距离进行缩放
        distances = np.sqrt(np.sum(centered**2, axis=1))
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
            
        return normalized
    
    def align_skeleton_for_ntu(self, skeleton_data):
        """NTU数据集的骨架对齐（参考gcn_skeleton_gif_visualizer.py）"""
        # 计算主要身体轴向（从骨盆到头部）
        # NTU RGB+D关节点索引：0=骨盆中心, 3=头顶
        if len(skeleton_data) >= 4:
            pelvis = skeleton_data[0]  # 骨盆中心
            head = skeleton_data[3]   # 头顶
            
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
                    skeleton_data = np.dot(skeleton_data, rotation_matrix.T)
        
        return skeleton_data
    
    def get_joint_weights_ntu(self):
        """获取NTU关节重要性权重（参考gcn_skeleton_gif_visualizer.py）"""
        weights = np.ones(25)
        # 重要关节权重x2
        head_joints = [3, 2]  # 头部
        hand_joints = [6, 7, 21, 22, 10, 11, 23, 24]  # 手部
        foot_joints = [14, 15, 18, 19]  # 脚部
        important_joints = head_joints + hand_joints + foot_joints
        weights[important_joints] = 2.0
        return weights
    
    def visualize_skeleton_data(self, skeleton_data, file_path=None):
        """可视化骨架数据"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        if len(skeleton_data.shape) == 2 and skeleton_data.shape[1] == 3:
            # 检测数据集类型
            dataset_type = self.detect_dataset_type(skeleton_data, file_path, self.sample_info)
            connections = self.get_skeleton_connections(dataset_type)
            
            # 根据数据集类型进行预处理
            if dataset_type == "mars":
                processed_skeleton = self.normalize_skeleton_for_mars(skeleton_data)
                title = f"MARS Skeleton Data (19 joints)"
                joint_weights = None
                # MARS数据：直接使用原坐标系，Z轴已经是竖直方向
                x, y, z = processed_skeleton[:, 0], processed_skeleton[:, 1], processed_skeleton[:, 2]
            else:  # NTU
                processed_skeleton = self.align_skeleton_for_ntu(skeleton_data.copy())
                title = f"NTU RGB+D Skeleton Data (25 joints)"
                joint_weights = self.get_joint_weights_ntu()
                # NTU数据：Z轴是竖直方向，但数据是倒立的（头部Z值小于脚部）
                # 翻转Z轴让骨架正立显示
                x, y, z = processed_skeleton[:, 0], processed_skeleton[:, 1], -processed_skeleton[:, 2]
            
            # 根据关节重要性使用不同颜色和大小
            if joint_weights is not None:
                # NTU数据集：重要关节用红色，普通关节用蓝色
                important_indices = np.where(joint_weights > 1.0)[0]
                normal_indices = np.where(joint_weights == 1.0)[0]
                
                if len(normal_indices) > 0:
                    ax.scatter(x[normal_indices], y[normal_indices], z[normal_indices], 
                             c='blue', s=20, alpha=0.7, label='Normal Joints')  # 从50→20
                
                if len(important_indices) > 0:
                    ax.scatter(x[important_indices], y[important_indices], z[important_indices], 
                             c='red', s=30, alpha=0.9, label='Important Joints', edgecolors='black')  # 从80→30
            else:
                # MARS数据集：统一蓝色
                ax.scatter(x, y, z, c='blue', s=25, alpha=0.7)  # 从60→25
            
            # 绘制骨架连接线
            for connection in connections:
                if connection[0] < len(processed_skeleton) and connection[1] < len(processed_skeleton):
                    # 使用与关节点相同的坐标变换
                    if dataset_type == "ntu":
                        # NTU数据：翻转Z轴
                        start_orig, end_orig = processed_skeleton[connection[0]], processed_skeleton[connection[1]]
                        start = [start_orig[0], start_orig[1], -start_orig[2]]  # 翻转Z
                        end = [end_orig[0], end_orig[1], -end_orig[2]]          # 翻转Z
                    else:
                        # MARS数据：直接使用原坐标
                        start, end = processed_skeleton[connection[0]], processed_skeleton[connection[1]]
                    
                    # 根据连接的关节重要性调整线条样式
                    if joint_weights is not None:
                        weight = max(joint_weights[connection[0]], joint_weights[connection[1]])
                        if weight > 1.0:  # 连接重要关节
                            line_color = 'red'
                            line_width = 3.0
                            alpha = 0.9
                        else:
                            line_color = 'blue'
                            line_width = 2.0
                            alpha = 0.7
                    else:
                        line_color = 'blue'
                        line_width = 2.0
                        alpha = 0.8
                    
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                           color=line_color, alpha=alpha, linewidth=line_width)
            
            # 标注关节点编号（可选）- 使用与关节点相同的坐标变换
            for i in range(len(processed_skeleton)):
                if dataset_type == "ntu":
                    # NTU数据：翻转Z轴
                    joint_orig = processed_skeleton[i]
                    joint_pos = [joint_orig[0], joint_orig[1], -joint_orig[2]]
                else:
                    # MARS数据：直接使用原坐标
                    joint_pos = processed_skeleton[i]
                
                ax.text(joint_pos[0], joint_pos[1], joint_pos[2], str(i), fontsize=6, alpha=0.7)
        
            # 设置坐标轴
            ax.set_xlabel('X (Left-Right)')
            ax.set_ylabel('Y (Front-Back)') 
            ax.set_zlabel('Z (Up-Down)')
            ax.set_title(title + " - Z-axis Vertical")
            
            # 添加图例（如果有关节权重信息）
            if joint_weights is not None:
                ax.legend()
            
            # 设置固定的显示范围以保持动画稳定
            ax.set_xlim([-0.6, 0.6])
            ax.set_ylim([-0.6, 0.6])
            ax.set_zlim([-0.6, 0.6])
            
            # 设置更好的观察视角，确保骨架看起来是站立的
            if dataset_type == "ntu":
                ax.view_init(elev=15, azim=45)  # NTU: 稍微从上往下看，45度角观察
            else:
                ax.view_init(elev=10, azim=30)  # MARS: 不同的视角
        
        self.canvas.draw()
        
    def show_placeholder(self):
        """显示占位图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No visualization data available\nor unsupported data format', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
        
    def refresh_view(self):
        """刷新视图"""
        if self.current_data:
            self.show_sample(self.current_data, self.sample_info)
            
    def reset_view(self):
        """重置视角"""
        if hasattr(self.figure, 'gca'):
            ax = self.figure.gca()
            if hasattr(ax, 'view_init'):
                ax.view_init(elev=20, azim=45)
                self.canvas.draw()
                
    def save_image(self):
        """保存图片"""
        if self.figure:
            filename = f"visualization_{int(time.time())}.png"
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ 图片已保存: {filename}")
            
    def close_window(self):
        """关闭窗口"""
        if self.window:
            self.window.destroy()
            self.window = None
            
    def is_window_open(self):
        """检查窗口是否打开"""
        return self.window is not None and self.window.winfo_exists()

# 全局可视化窗口实例
_visualization_window = None

def get_visualization_window():
    """获取全局可视化窗口实例"""
    global _visualization_window
    if _visualization_window is None:
        _visualization_window = VisualizationWindow()
    return _visualization_window

def show_sample_visualization(sample_data, sample_info=None):
    """显示样本可视化（外部调用接口）"""
    try:
        # 每次都创建新的窗口实例，避免窗口销毁问题
        viz_window = VisualizationWindow()
        viz_window.show_sample(sample_data, sample_info)
        return True
    except Exception as e:
        print(f"⚠️ 可视化窗口创建失败: {e}")
        return False

def close_visualization_window():
    """关闭可视化窗口"""
    global _visualization_window
    if _visualization_window:
        _visualization_window.close_window()
        _visualization_window = None
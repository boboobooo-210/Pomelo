#!/usr/bin/env python3
"""
码本动作标注工具
用于人工标注Token对应的动作语义，构建码本-动作映射表
"""

import json
import os
import sys
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional

# LLM 友好格式导出器
try:
    from llm_annotation_exporter import LLMAnnotationExporter
    LLM_EXPORTER_AVAILABLE = True
except ImportError:
    LLM_EXPORTER_AVAILABLE = False
    print("⚠️ LLM 导出器不可用，将只支持标准格式导出")

try:
    import h5py
except ImportError:
    h5py = None
    print("⚠️ h5py 未安装，将使用模拟数据")

# 可视化库 (用于GIF生成)
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    GIF_AVAILABLE = True
except ImportError:
    GIF_AVAILABLE = False
    print("⚠️ matplotlib 不可用，GIF生成功能将被禁用")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas 不可用，MARS Token数据集加载将被禁用")

def safe_input(prompt, default="", timeout=None):
    """安全的输入函数，处理EOF等异常"""
    try:
        result = input(prompt).strip()
        return result if result else default
    except (EOFError, KeyboardInterrupt):
        print(f"\n⚠️ 输入中断，使用默认值: {default}")
        return default

# 全局标志：是否运行在批处理模式
BATCH_MODE = False

def set_batch_mode(enabled=True):
    """设置批处理模式"""
    global BATCH_MODE
    BATCH_MODE = enabled

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️ GUI组件不可用，将使用命令行界面")

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib不可用，将使用简化可视化")

# 导入可视化窗口模块
try:
    from visualization_window import show_sample_visualization, close_visualization_window
    VISUALIZATION_AVAILABLE = True
except ImportError:
    try:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from visualization_window import show_sample_visualization, close_visualization_window
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        print("⚠️ 可视化窗口模块不可用，将使用文本显示")

class SkeletonAnnotationTool:
    """骨架码本标注工具"""
    
    def __init__(self):
        self.annotation_data = {}
        self.samples_to_annotate = []
        self.current_sample_id = 0
        self.save_dir = "annotations"  # 标注结果保存目录
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # 会话ID
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "exports"), exist_ok=True)
        
        # 动作标签模板 - 支持NTU和MARS数据集
        # NTU数据集动作模板（原始完整版本）
        self.ntu_action_templates = {
            'head_spine': [
                "中性姿态", "抬头向上", "低头向下", "左转头部", "右转头部",
                "挺直脊柱", "前倾身体", "后仰身体", "左侧弯曲", "右侧弯曲",
                "点头动作", "摇头动作", "侧倾头部", "旋转躯干", "弓背姿态"
            ],
            'left_arm': [
                "自然下垂", "上举过头", "前伸指向", "侧平举", "弯曲撑腰",
                "交叉胸前", "挥手动作", "背后伸展", "握拳准备", "放松摆动",
                "推拉动作", "抱抱姿势", "敬礼动作", "遮挡面部", "支撑身体"
            ],
            'right_arm': [
                "自然下垂", "上举过头", "前伸指向", "侧平举", "弯曲撑腰",
                "交叉胸前", "挥手动作", "背后伸展", "握拳准备", "放松摆动",
                "推拉动作", "抱抱姿势", "敬礼动作", "遮挡面部", "支撑身体"
            ],
            'left_leg': [
                "直立支撑", "微弯准备", "抬起前踏", "侧向迈步", "蹲姿弯曲",
                "后退准备", "踢腿动作", "站立平衡", "交叉站立", "跳跃准备",
                "单腿支撑", "膝盖弯曲", "脚尖着地", "抬膝动作", "侧踢准备"
            ],
            'right_leg': [
                "直立支撑", "微弯准备", "抬起前踏", "侧向迈步", "蹲姿弯曲",
                "后退准备", "踢腿动作", "站立平衡", "交叉站立", "跳跃准备",
                "单腿支撑", "膝盖弯曲", "脚尖着地", "抬膝动作", "侧踢准备"
            ]
        }
        
        # MARS数据集简化动作模板（移除手指依赖动作）
        self.mars_action_templates = {
            'head_spine': [
                "正常姿态", "抬头", "低头看", "左侧转", "右侧转",
                "挺直站立", "前倾", "后仰", "左倾斜", "右倾斜"
            ],
            'left_arm': [
                "自然垂落", "上举", "前伸", "侧举", "叉腰",
                "向内弯曲", "自然弯曲", "后伸", "左侧抬起", "向上弯曲"
            ],
            'right_arm': [
                "自然垂落", "上举", "前伸", "侧举", "叉腰",
                "向内弯曲", "自然弯曲", "后伸", "右侧抬起", "向上弯曲"
            ],
            'left_leg': [
                "站立", "弯曲", "前抬", "侧抬", "蹲下",
                "后退", "踢出", "向前跨步", "向左跨步", "跳跃"
            ],
            'right_leg': [
                "站立", "弯曲", "前抬", "侧抬", "蹲下",
                "后退", "踢出", "向前跨步", "向右跨步", "跳跃"
            ]
        }
        
        # 根据数据集类型选择动作模板
        self.action_templates = self.ntu_action_templates  # 默认使用NTU模板
        
        self.part_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        self.part_display_names = ['头部脊柱', '左臂', '右臂', '左腿', '右腿']
        
        # 为兼容性添加body_parts别名
        self.body_parts = self.part_names
        
        # 根据可用性选择界面模式
        self.use_gui = GUI_AVAILABLE
        
    def generate_sample_data(self, num_samples: int = 50):
        """生成示例标注数据"""
        print(f"📊 生成 {num_samples} 个示例样本...")
        
        self.samples_to_annotate = []
        
        for i in range(num_samples):
            # 模拟骨架数据 (25关节点 x 3坐标)
            skeleton = np.random.randn(25, 3) * 0.5
            
            # 模拟token序列 - 使用一些预设的组合
            if i % 10 == 0:
                tokens = [28, 58, 65, 18, 23]  # 庆祝动作
            elif i % 10 == 1:
                tokens = [15, 76, 119, 72, 23]  # 问候动作
            elif i % 10 == 2:
                tokens = [45, 32, 41, 113, 126]  # 检查动作
            else:
                tokens = [
                    np.random.randint(0, 128),
                    np.random.randint(0, 128),
                    np.random.randint(0, 128),
                    np.random.randint(0, 128),
                    np.random.randint(0, 128)
                ]
            
            self.samples_to_annotate.append({
                'id': i,
                'skeleton': skeleton,
                'tokens': tokens,
                'annotated': False,
                'source': 'generated'
            })
            
        print(f"✅ 生成完成，共 {len(self.samples_to_annotate)} 个样本")
        
    def load_real_data(self, data_source: str = "ntu"):
        """加载真实数据
        
        Args:
            data_source: "ntu" 或 "radar_gt" 或 "both" 或 "mars_tokens"
        """
        print(f"📥 加载真实数据源: {data_source}")
        
        if data_source == "ntu":
            self._load_ntu_dataset()
        elif data_source == "radar_gt":
            self._load_radar_ground_truth()
        elif data_source == "mars_tokens":
            self._load_mars_token_dataset()
        elif data_source == "both":
            self._load_ntu_dataset()
            self._load_radar_ground_truth()
        else:
            print("❌ 无效的数据源，使用示例数据")
            self.generate_sample_data()
            
    def _load_ntu_dataset(self):
        """加载NTU RGB+D数据集"""
        print("📊 加载NTU RGB+D数据集...")
        
        try:
            # NTU数据集路径 - 优先使用骨架数据
            ntu_data_paths = [
                "/home/uo/myProject/HumanPoint-BERT/data/NTU-RGB+D",  # 优先：原始骨架数据路径
                "/home/uo/myProject/HumanPoint-BERT/data/NTU-Pred",  # 备用：预处理点云数据路径
                "/home/uo/myProject/CRSkeleton/data/ntu"  # 备用本地路径
            ]
            
            # 查找可用的NTU数据路径
            ntu_path = None
            for path in ntu_data_paths:
                if os.path.exists(path):
                    ntu_path = path
                    break
                    
            if ntu_path is None:
                print("⚠️ 未找到NTU数据集，生成模拟NTU数据...")
                self._generate_simulated_ntu_data()
                return
                
            print(f"✅ 找到NTU数据集: {ntu_path}")
            
            # 加载NTU动作标签映射
            ntu_action_labels = {
                1: "drink water", 2: "eat meal/snack", 3: "brushing teeth", 4: "brushing hair",
                5: "drop", 6: "pickup", 7: "throw", 8: "sitting down", 9: "standing up (from sitting position)",
                10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper", 14: "wear jacket",
                15: "take off jacket", 16: "wear a shoe", 17: "take off a shoe", 18: "wear on glasses",
                19: "take off glasses", 20: "put on a hat/cap", 21: "take off a hat/cap", 22: "cheer up",
                23: "hand waving", 24: "kicking something", 25: "reach into pocket", 26: "hopping (one foot jumping)",
                27: "jump up", 28: "make a phone call/answer phone", 29: "playing with phone/tablet", 30: "typing on a keyboard",
                31: "pointing to something with finger", 32: "taking a selfie", 33: "check time (from watch)",
                34: "rub two hands together", 35: "nod head/bow", 36: "shake head", 37: "wipe face",
                38: "salute", 39: "put the palms together", 40: "cross hands in front (say stop)",
                41: "sneeze/cough", 42: "staggering", 43: "falling", 44: "touch head (headache)",
                45: "touch chest (stomachache/heart pain)", 46: "touch back (backache)", 47: "touch neck (neckache)",
                48: "nausea or vomiting condition", 49: "use a fan (with hand or paper)/feeling warm",
                50: "punching/slapping other person", 51: "kicking other person", 52: "pushing other person",
                53: "pat on back of other person", 54: "point finger at the other person", 55: "hugging other person",
                56: "giving something to other person", 57: "touch other person's pocket", 58: "handshaking",
                59: "walking towards each other", 60: "walking apart from each other"
            }
            
            # 加载实际NTU数据样本
            self._load_ntu_samples(ntu_path, ntu_action_labels)
            
        except Exception as e:
            print(f"❌ 加载NTU数据失败: {e}")
            self._generate_simulated_ntu_data()
            
    def _load_radar_ground_truth(self):
        """加载雷达数据集的Ground Truth"""
        print("📡 加载雷达数据集Ground Truth...")
        
        try:
            # 雷达数据集路径 - 使用实际MARS项目路径
            radar_gt_paths = [
                "/home/uo/myProject/HumanPoint-BERT/data/MARS",  # MARS雷达数据路径
                "/home/uo/myProject/CRSkeleton/data/radar_gt",  # 备用本地路径
                "./data/radar_gt"  # 相对路径备用
            ]
            
            # 查找可用的雷达GT数据路径
            radar_path = None
            for path in radar_gt_paths:
                if os.path.exists(path):
                    radar_path = path
                    break
                    
            if radar_path is None:
                print("⚠️ 未找到雷达Ground Truth数据，生成模拟数据...")
                self._generate_simulated_radar_data()
                return
                
            print(f"✅ 找到雷达Ground Truth: {radar_path}")
            # 加载实际MARS雷达数据
            self._load_mars_samples(radar_path)
            
        except Exception as e:
            print(f"❌ 加载雷达Ground Truth失败: {e}")
            self._generate_simulated_radar_data()
    
    def _load_ntu_samples(self, ntu_path, action_labels, num_samples=10):
        """加载实际NTU数据样本"""
        try:
            print(f"🔄 从NTU数据集加载样本: {ntu_path}")
            
            # 首先尝试加载.skeleton文件（真正的骨架数据）
            if os.path.exists(ntu_path):
                skeleton_files = [f for f in os.listdir(ntu_path) if f.endswith('.skeleton')]
                if skeleton_files:
                    print(f"📋 找到{len(skeleton_files)}个.skeleton文件，加载骨架数据")
                    return self._load_ntu_skeleton_files(ntu_path, action_labels, num_samples)
                
                # 如果没有.skeleton文件，再尝试H5文件
                h5_files = [f for f in os.listdir(ntu_path) if f.endswith('.h5')]
                if h5_files and h5py is not None:
                    print(f"📋 找到{len(h5_files)}个.h5文件，加载点云数据")
                    return self._load_ntu_h5_files(ntu_path, action_labels, num_samples)
                elif h5_files:
                    print("❌ h5py 未安装，无法加载HDF5数据")
                    return False
            
            print("⚠️ 未找到NTU数据文件，使用模拟数据")
            return False
                
        except Exception as e:
            print(f"❌ 加载NTU真实数据失败: {e}")
            return False
    
    def _load_ntu_skeleton_files(self, ntu_path, action_labels, num_samples=10):
        """加载NTU .skeleton文件（真正的25关节骨架数据）"""
        try:
            skeleton_files = [f for f in os.listdir(ntu_path) if f.endswith('.skeleton')]
            selected_files = skeleton_files[:min(num_samples, len(skeleton_files))]
            
            sample_id_offset = len(self.samples_to_annotate)
            loaded_count = 0
            
            for filename in selected_files:
                try:
                    filepath = os.path.join(ntu_path, filename)
                    skeleton_data = self._read_ntu_skeleton_file(filepath)
                    
                    if skeleton_data is not None and len(skeleton_data) > 0:
                        # 取第一帧的骨架数据（25个关节点）
                        sample_frame = skeleton_data[0]  # (25, 3)
                        
                        # 从文件名提取动作ID
                        action_id = self._extract_action_id_from_filename(filename)
                        action_name = action_labels.get(action_id, f"action_{action_id}")
                        
                        # 创建样本 - 注意这里用skeleton_data而不是point_cloud_data
                        sample = {
                            'id': sample_id_offset + loaded_count,
                            'tokens': self._skeleton_to_mock_tokens(sample_frame),
                            'source': 'ntu_real',
                            'filename': filename,
                            'ground_truth_action': action_name,
                            'skeleton_data': sample_frame,  # 关键：25个关节点
                            'total_frames': len(skeleton_data)
                        }
                        
                        self.samples_to_annotate.append(sample)
                        loaded_count += 1
                        print(f"✅ 加载NTU骨架样本 {loaded_count}: {filename} -> {action_name} (关节数: {sample_frame.shape[0]}, 帧数: {len(skeleton_data)})")
                        
                except Exception as e:
                    print(f"⚠️ 跳过骨架文件 {filename}: {e}")
                    continue
                    
            print(f"📊 成功加载 {loaded_count} 个真实NTU骨架样本")
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ 加载NTU骨架文件失败: {e}")
            return False
    
    def _load_ntu_h5_files(self, ntu_path, action_labels, num_samples=10):
        """加载NTU H5文件（点云数据）"""
        try:
            h5_files = [f for f in os.listdir(ntu_path) if f.endswith('.h5')]
            selected_files = h5_files[:min(num_samples, len(h5_files))]
            
            sample_id_offset = len(self.samples_to_annotate)
            loaded_count = 0
            
            for i, filename in enumerate(selected_files):
                try:
                    filepath = os.path.join(ntu_path, filename)
                    with h5py.File(filepath, 'r') as f:
                        # 检查可用的键
                        available_keys = list(f.keys())
                        print(f"📋 文件 {filename} 可用键: {available_keys}")
                        
                        # 尝试常见的骨架数据键
                        skeleton_key = None
                        for key in ['enhanced_data', 'skeleton', 'data', 'joints', 'keypoints']:
                            if key in f:
                                skeleton_key = key
                                break
                        
                        if skeleton_key:
                            point_cloud_data = f[skeleton_key][:]
                            # NTU-Pred数据格式: [frames, points, coords] = (103, 720, 3)
                            if len(point_cloud_data.shape) == 3 and point_cloud_data.shape[0] > 0:
                                # 取第一帧点云数据
                                sample_frame = point_cloud_data[0]  # (720, 3)
                                
                                # 从文件名提取动作ID（NTU格式：...A[action_id]...）
                                action_id = self._extract_action_id_from_filename(filename)
                                action_name = action_labels.get(action_id, f"action_{action_id}")
                                
                                # 创建样本 - 这是点云数据
                                sample = {
                                    'id': sample_id_offset + loaded_count,
                                    'tokens': self._pointcloud_to_mock_tokens(sample_frame),
                                    'source': 'ntu_real',
                                    'filename': filename,
                                    'ground_truth_action': action_name,
                                    'point_cloud_data': sample_frame,  # 720个点的点云
                                    'total_frames': point_cloud_data.shape[0]
                                }
                                
                                self.samples_to_annotate.append(sample)
                                loaded_count += 1
                                print(f"✅ 加载NTU点云样本 {loaded_count}: {filename} -> {action_name} (点数: {sample_frame.shape[0]}, 帧数: {point_cloud_data.shape[0]})")
                except Exception as e:
                    print(f"⚠️ 跳过文件 {filename}: {e}")
                    continue
            
            print(f"📊 成功加载 {loaded_count} 个真实NTU点云样本")
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ 加载NTU H5文件失败: {e}")
            return False
    
    def _load_mars_samples(self, mars_path, num_samples=8):
        """加载实际MARS骨架ground truth数据样本"""
        try:
            print(f"🔄 从MARS数据集加载骨架ground truth: {mars_path}")
            
            # 直接使用骨架标签文件（ground truth）
            skeleton_files = [
                'labels_test.npy',
                'labels_train.npy',
                'labels_validate.npy'
            ]
            
            sample_id_offset = len(self.samples_to_annotate)
            loaded_count = 0
            
            for skeleton_file in skeleton_files:
                skeleton_path = os.path.join(mars_path, skeleton_file)
                
                if os.path.exists(skeleton_path):
                    try:
                        # 加载骨架ground truth数据
                        skeleton_labels = np.load(skeleton_path)
                        
                        print(f"📊 {skeleton_file} 骨架数据形状: {skeleton_labels.shape}")
                        
                        # 选择样本
                        total_samples = skeleton_labels.shape[0]
                        selected_indices = np.random.choice(total_samples, min(num_samples//3, total_samples), replace=False)
                        
                        for idx in selected_indices:
                            # 解析57维数据为19×3骨架格式
                            # 参考vis_gif_skeleton_extractor.py的parse_joints函数
                            # 格式: (x1...x19, y1...y19, z1...z19)
                            skeleton_57d = skeleton_labels[idx]
                            skeleton_19x3 = self._parse_mars_joints(skeleton_57d)
                            
                            # 创建样本 - 注意这里用skeleton_data而不是radar_data
                            sample = {
                                'id': sample_id_offset + loaded_count,
                                'tokens': self._skeleton_to_mock_tokens(skeleton_19x3),
                                'source': 'mars_real',
                                'filename': f"{skeleton_file}_{idx}",
                                'ground_truth_action': f"mars_skeleton_{idx}",
                                'skeleton_data': skeleton_19x3  # 19×3骨架数据
                            }
                            
                            self.samples_to_annotate.append(sample)
                            loaded_count += 1
                            print(f"✅ 加载MARS骨架样本 {loaded_count}: {skeleton_file}[{idx}] (关节数: {skeleton_19x3.shape[0]})")
                            
                            if loaded_count >= num_samples:
                                break
                                
                    except Exception as e:
                        print(f"⚠️ 加载 {skeleton_file} 失败: {e}")
                        continue
                
                if loaded_count >= num_samples:
                    break
            
            print(f"📊 成功加载 {loaded_count} 个真实MARS骨架样本")
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ 加载MARS真实数据失败: {e}")
            return False
    
    def _load_mars_token_dataset(self, num_samples=None):
        """加载MARS_recon_tokens数据集(带token序列)
        
        这是标注的主要数据源,包含:
        - 提取的骨架 (extracted)
        - 重建的骨架 (reconstructed) 
        - Token序列 (5个部位token)
        - VQ损失
        
        Args:
            num_samples: 加载样本数量,None表示全部加载
        """
        print("\n🎯 加载 MARS Token 数据集 (用于标注)")
        print("=" * 70)
        
        try:
            # MARS_recon_tokens 目录
            token_data_dir = 'data/MARS_recon_tokens'
            
            if not os.path.exists(token_data_dir):
                print(f"❌ 未找到目录: {token_data_dir}")
                print("💡 请先运行 skeleton_extraction_reconstruction_saver.py 生成数据")
                return False
            
            # 读取 CSV 索引文件
            csv_path = os.path.join(token_data_dir, 'index.csv')
            if not os.path.exists(csv_path):
                print(f"❌ 未找到索引文件: {csv_path}")
                return False
            
            import pandas as pd
            index_df = pd.read_csv(csv_path)
            
            print(f"✅ 找到 {len(index_df)} 个样本")
            print(f"   数据列: {list(index_df.columns)}")
            
            # 决定加载多少样本
            if num_samples is None:
                samples_to_load = len(index_df)
                print(f"📊 加载全部 {samples_to_load} 个样本")
            else:
                samples_to_load = min(num_samples, len(index_df))
                print(f"📊 加载前 {samples_to_load} 个样本")
            
            # 加载样本
            sample_id_offset = len(self.samples_to_annotate)
            loaded_count = 0
            
            for idx in range(samples_to_load):
                row = index_df.iloc[idx]
                
                try:
                    # 读取单个样本文件
                    sample_file = row['file_path']
                    if not os.path.exists(sample_file):
                        print(f"⚠️ 样本文件不存在: {sample_file}")
                        continue
                    
                    data = np.load(sample_file)
                    
                    # 解析 tokens 字符串 "[1, 2, 3, 4, 5]" -> [1, 2, 3, 4, 5]
                    tokens_str = row['tokens_str']
                    tokens = eval(tokens_str) if isinstance(tokens_str, str) else tokens_str
                    
                    # 创建样本
                    sample = {
                        'id': sample_id_offset + loaded_count,
                        'tokens': tokens,  # 5个部位token
                        'source': 'mars_tokens',
                        'filename': os.path.basename(sample_file),
                        'file_path': sample_file,
                        'split': row['split'],
                        'vq_loss': float(row['vq_loss']),
                        'token_first': int(row['token_first']),
                        # 骨架数据
                        'extracted': data['extracted'],  # (25, 3)
                        'reconstructed': data['reconstructed'],  # (25, 3)
                        'annotated': False
                    }
                    
                    self.samples_to_annotate.append(sample)
                    loaded_count += 1
                    
                    if loaded_count % 1000 == 0:
                        print(f"   已加载 {loaded_count}/{samples_to_load} 个样本...")
                    
                except Exception as e:
                    print(f"⚠️ 加载样本 {idx} 失败: {e}")
                    continue
            
            print(f"\n✅ 成功加载 {loaded_count} 个 MARS Token 样本")
            print(f"   - 每个样本包含: 5个token + 提取骨架 + 重建骨架")
            print(f"   - 数据来源: {token_data_dir}")
            
            # 设置为MARS动作模板（更适合骨架数据标注）
            self.action_templates = self.mars_action_templates
            print(f"   - 使用 MARS 动作模板进行标注")
            
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ 加载 MARS Token 数据集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_action_id_from_filename(self, filename):
        """从NTU文件名提取动作ID"""
        import re
        # NTU文件名格式：S001C001P001R001A001.h5 或 S001C001P001R001A001.skeleton
        match = re.search(r'A(\d+)', filename)
        if match:
            return int(match.group(1))
        return 1  # 默认动作ID
    
    def _read_ntu_skeleton_file(self, file_path):
        """读取NTU .skeleton文件，返回25关节点骨架数据（文本格式）"""
        try:
            with open(file_path, 'r') as f:
                # 读取帧数
                frame_count = int(f.readline().strip())
                
                frames_data = []
                for frame_idx in range(min(frame_count, 5)):  # 最多读取5帧
                    # 读取人体数量
                    body_count = int(f.readline().strip())
                    
                    frame_skeletons = []
                    for body_idx in range(body_count):
                        # 读取人体信息行（跳过）
                        body_info = f.readline().strip()
                        
                        # 读取关节数量
                        joint_count = int(f.readline().strip())
                        
                        # 读取关节数据
                        joints = []
                        for joint_idx in range(joint_count):
                            joint_line = f.readline().strip().split()
                            if len(joint_line) >= 3:
                                # NTU RGB+D坐标转换: (x,z,y) -> (x,y,z) 仅用于可视化
                                # 参考gcn_skeleton_gif_visualizer.py的处理方式
                                x, z, y = float(joint_line[0]), float(joint_line[1]), float(joint_line[2])
                                joints.append([x, y, z])
                        
                        if len(joints) == 25:  # NTU有25个关节
                            skeleton = np.array(joints)
                            # 数据预处理：标准化坐标系
                            skeleton = self._normalize_ntu_skeleton(skeleton)
                            frame_skeletons.append(skeleton)
                    
                    if frame_skeletons:
                        # 如果有多个人体，选择第一个
                        frames_data.append(frame_skeletons[0])
                
                return frames_data if frames_data else None
                
        except Exception as e:
            print(f"读取骨架文件失败: {e}")
            return None
    
    def _parse_mars_joints(self, joints_data):
        """解析MARS关节数据格式: (x1...x19, y1...y19, z1...z19)
        参考vis_gif_skeleton_extractor.py的parse_joints函数
        """
        if joints_data.shape == (57,):
            x_coords = joints_data[0:19]
            y_coords = joints_data[19:38]  
            z_coords = joints_data[38:57]
            return np.column_stack((x_coords, y_coords, z_coords))
        else:
            raise ValueError(f"无效的MARS关节数据形状: {joints_data.shape}")
    
    def _normalize_ntu_skeleton(self, skeleton):
        """标准化NTU骨架数据，使其适合可视化"""
        # 1. 以脊椎中心（关节1）为原点
        spine_center = skeleton[1].copy()  # 脊椎中心
        skeleton_centered = skeleton - spine_center
        
        # 2. 调整坐标系：原始NTU数据Z是深度，我们需要Z为竖直方向
        # NTU坐标系: X-左右, Y-上下, Z-前后(深度)
        # 目标坐标系: X-左右, Y-前后, Z-上下
        skeleton_reoriented = skeleton_centered.copy()
        skeleton_reoriented[:, 1] = skeleton_centered[:, 2]  # Z(深度) -> Y(前后)
        skeleton_reoriented[:, 2] = skeleton_centered[:, 1]  # Y(上下) -> Z(竖直)
        
        # 3. 缩放到合适的范围
        max_range = np.abs(skeleton_reoriented).max()
        if max_range > 0:
            skeleton_reoriented = skeleton_reoriented / max_range
        
        return skeleton_reoriented
    
    def _pointcloud_to_mock_tokens(self, point_cloud_data):
        """将点云数据转换为模拟token (NTU-Pred格式)"""
        # 点云数据格式: (720, 3) - 720个点，每个点xyz坐标
        if point_cloud_data.shape[0] >= 720 and point_cloud_data.shape[1] == 3:
            tokens = []
            # 将720个点分为5个区域 (对应5个身体部位)
            points_per_part = 144  # 720 / 5 = 144
            
            for i in range(5):
                start_idx = i * points_per_part
                end_idx = start_idx + points_per_part if i < 4 else point_cloud_data.shape[0]
                part_points = point_cloud_data[start_idx:end_idx]
                
                if len(part_points) > 0:
                    # 计算该部分点云的特征
                    centroid = np.mean(part_points, axis=0)
                    variance = np.var(part_points, axis=0)
                    feature_sum = np.sum(np.abs(centroid)) + np.sum(variance)
                    token = int(feature_sum * 50) % 128  # 映射到0-127
                    tokens.append(token)
                else:
                    tokens.append(np.random.randint(0, 128))
            
            return tokens
        else:
            return [np.random.randint(0, 128) for _ in range(5)]
    
    def _skeleton_to_mock_tokens(self, skeleton_data):
        """将骨架数据转换为模拟token"""
        # 简单的骨架->token映射
        if skeleton_data.shape[0] >= 25:  # NTU 25关节
            # 基于关节位置生成token
            tokens = []
            for part_name in self.body_parts:
                part_joints = self._get_part_joints(part_name, skeleton_data)
                if len(part_joints) > 0:
                    # 基于关节位置计算token
                    avg_pos = np.mean(part_joints, axis=0)
                    token = int(np.sum(np.abs(avg_pos)) * 100) % 128  # 简单映射到0-127
                    tokens.append(token)
                else:
                    tokens.append(np.random.randint(0, 128))
            return tokens[:5]  # 返回5个部位的token
        else:
            return [np.random.randint(0, 128) for _ in range(5)]
    
    def _radar_to_mock_tokens(self, radar_data):
        """将雷达数据转换为模拟token"""
        # 简单的雷达->token映射
        if len(radar_data.shape) >= 1:
            tokens = []
            # 将雷达数据分为5个区域
            data_flat = radar_data.flatten()
            chunk_size = len(data_flat) // 5
            for i in range(5):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < 4 else len(data_flat)
                chunk = data_flat[start_idx:end_idx]
                token = int(np.mean(np.abs(chunk)) * 1000) % 128 if len(chunk) > 0 else 0
                tokens.append(token)
            return tokens
        else:
            return [np.random.randint(0, 128) for _ in range(5)]
    
    def _get_part_joints(self, part_name, skeleton_data):
        """获取身体部位对应的关节点"""
        # 根据关节数量自动判断数据集类型
        num_joints = skeleton_data.shape[0]
        
        if num_joints == 25:  # NTU RGB+D 25关节点映射
            # 切换到NTU动作模板
            self.action_templates = self.ntu_action_templates
            joint_mapping = {
                'head_spine': [0, 1, 2, 3, 20],  # 头部和脊椎
                'left_arm': [4, 5, 6, 7, 21],    # 左臂
                'right_arm': [8, 9, 10, 11, 22], # 右臂
                'left_leg': [12, 13, 14, 15, 23], # 左腿
                'right_leg': [16, 17, 18, 19, 24] # 右腿
            }
        elif num_joints == 19:  # MARS Kinect 19关节点映射
            # 切换到MARS简化动作模板
            self.action_templates = self.mars_action_templates
            joint_mapping = {
                'head_spine': [0, 1, 2, 3, 18],  # 头部和脊椎 (spinebase, spinemid, head, neck, spineshoulder)
                'left_arm': [4, 5, 6],           # 左臂 (leftshoulder, leftelbow, leftwrist)
                'right_arm': [7, 8, 9],          # 右臂 (rightshoulder, rightelbow, rightwrist)
                'left_leg': [10, 11, 12, 13],    # 左腿 (hipleft, kneeleft, ankleleft, footleft)
                'right_leg': [14, 15, 16, 17]    # 右腿 (hipright, kneeright, ankleright, footright)
            }
        else:
            # 未知格式，使用NTU默认映射
            self.action_templates = self.ntu_action_templates
            joint_mapping = {
                'head_spine': list(range(min(5, num_joints))),
                'left_arm': [],
                'right_arm': [],
                'left_leg': [],
                'right_leg': []
            }
        
        joint_indices = joint_mapping.get(part_name, [])
        valid_joints = []
        for idx in joint_indices:
            if idx < skeleton_data.shape[0]:
                valid_joints.append(skeleton_data[idx])
        return np.array(valid_joints) if len(valid_joints) > 0 else np.array([])
            
    def _generate_simulated_ntu_data(self):
        """生成基于NTU动作类别的模拟数据"""
        print("🎭 生成NTU风格的模拟数据...")
        
        # NTU常见动作的Token模式
        ntu_token_patterns = {
            "drink water": [15, 76, 41, 18, 23],        # 中性头部 + 手臂动作 + 站立
            "clapping": [15, 58, 65, 18, 23],           # 中性头部 + 双手拍击 + 站立
            "hand waving": [15, 32, 119, 18, 23],       # 中性头部 + 挥手 + 站立
            "sitting down": [45, 32, 41, 113, 126],     # 低头 + 下垂手臂 + 蹲姿
            "standing up": [28, 58, 65, 72, 78],        # 抬头 + 上举手臂 + 迈步
            "jump up": [28, 103, 107, 95, 101],         # 抬头 + 侧举手臂 + 跳跃准备
            "reading": [45, 76, 82, 18, 23],            # 低头 + 前伸手臂 + 站立
            "phone call": [67, 76, 41, 18, 23],         # 转头 + 单手举起 + 站立
            "check time": [45, 76, 41, 18, 23],         # 低头 + 看手腕 + 站立
            "cross hands": [15, 124, 119, 18, 23],      # 中性头部 + 交叉手臂 + 站立
        }
        
        sample_id_offset = len(self.samples_to_annotate)
        
        for action_name, token_pattern in ntu_token_patterns.items():
            # 为每个动作生成2-3个变体
            for variant in range(2):
                # 添加轻微随机变化
                varied_tokens = []
                for token in token_pattern:
                    # 在原Token基础上添加±5的随机变化
                    varied_token = token + np.random.randint(-5, 6)
                    varied_token = max(0, min(127, varied_token))  # 确保在0-127范围内
                    varied_tokens.append(varied_token)
                
                # 生成对应的骨架数据（模拟）
                skeleton = self._generate_skeleton_for_action(action_name)
                
                self.samples_to_annotate.append({
                    'id': sample_id_offset + len(self.samples_to_annotate),
                    'skeleton': skeleton,
                    'tokens': varied_tokens,
                    'annotated': False,
                    'source': 'ntu_simulated',
                    'action_hint': action_name,
                    'original_ntu_action': action_name
                })
                
        print(f"✅ 生成了 {len(ntu_token_patterns) * 2} 个NTU风格样本")
        
    def _generate_simulated_radar_data(self):
        """生成雷达数据风格的模拟数据"""
        print("📡 生成雷达风格的模拟数据...")
        
        # 雷达数据可能更多是基础动作
        radar_actions = [
            "walking", "standing", "sitting", "raising_hand", 
            "bending", "turning", "reaching", "pointing"
        ]
        
        sample_id_offset = len(self.samples_to_annotate)
        
        for action in radar_actions:
            for variant in range(3):  # 每个动作3个变体
                tokens = [np.random.randint(0, 128) for _ in range(5)]
                skeleton = np.random.randn(25, 3) * 0.3  # 更小的变化范围
                
                self.samples_to_annotate.append({
                    'id': sample_id_offset + len(self.samples_to_annotate),
                    'skeleton': skeleton,
                    'tokens': tokens,
                    'annotated': False,
                    'source': 'radar_simulated',
                    'action_hint': action,
                    'radar_action': action
                })
                
        print(f"✅ 生成了 {len(radar_actions) * 3} 个雷达风格样本")
        
    def _generate_ntu_based_samples(self, ntu_labels):
        """基于真实NTU标签生成样本"""
        print("📊 基于NTU标签生成样本...")
        
        # 选择一些代表性的动作进行标注
        priority_actions = [1, 2, 8, 9, 10, 23, 27, 28, 31, 35, 36, 40]  # 优先标注的动作
        
        sample_id_offset = len(self.samples_to_annotate)
        
        for action_id in priority_actions:
            if action_id in ntu_labels:
                action_name = ntu_labels[action_id]
                
                # 为每个动作生成样本
                tokens = self._generate_tokens_for_ntu_action(action_id, action_name)
                skeleton = self._generate_skeleton_for_action(action_name)
                
                self.samples_to_annotate.append({
                    'id': sample_id_offset + len(self.samples_to_annotate),
                    'skeleton': skeleton,
                    'tokens': tokens,
                    'annotated': False,
                    'source': 'ntu_real',
                    'ntu_action_id': action_id,
                    'ntu_action_name': action_name,
                    'priority': True
                })
                
        print(f"✅ 基于NTU标签生成了 {len(priority_actions)} 个优先样本")
        
    def _generate_tokens_for_ntu_action(self, action_id: int, action_name: str) -> List[int]:
        """根据NTU动作生成合理的Token序列"""
        
        # 基于动作语义生成Token模式
        token_patterns = {
            1: [15, 76, 41, 18, 23],    # drink water: 中性头部 + 举手到嘴边
            2: [45, 76, 82, 18, 23],    # eat meal: 低头 + 双手进食动作
            8: [45, 32, 41, 113, 126],  # sitting down: 低头 + 下垂手臂 + 蹲姿
            9: [28, 58, 65, 72, 78],    # standing up: 抬头 + 上举手臂 + 起立
            10: [15, 58, 65, 18, 23],   # clapping: 中性头部 + 双手拍击
            23: [15, 32, 119, 18, 23],  # hand waving: 中性头部 + 挥手
            27: [28, 103, 107, 95, 101], # jump up: 抬头 + 侧举 + 跳跃
            28: [67, 76, 41, 18, 23],   # phone call: 转头 + 单手举起
            31: [15, 76, 82, 18, 23],   # pointing: 中性头部 + 指向
            35: [89, 32, 41, 18, 23],   # nod head: 点头 + 自然手臂
            36: [67, 32, 41, 18, 23],   # shake head: 摇头 + 自然手臂
            40: [15, 124, 119, 18, 23], # cross hands: 中性头部 + 交叉手臂
        }
        
        if action_id in token_patterns:
            base_tokens = token_patterns[action_id]
            # 添加少量随机变化
            varied_tokens = []
            for token in base_tokens:
                variation = np.random.randint(-3, 4)
                varied_token = max(0, min(127, token + variation))
                varied_tokens.append(varied_token)
            return varied_tokens
        else:
            # 对于未预设的动作，生成随机但合理的Token
            return [np.random.randint(0, 128) for _ in range(5)]
            
    def _generate_skeleton_for_action(self, action_name: str) -> np.ndarray:
        """根据动作名称生成对应的骨架数据"""
        
        # 生成基础骨架 (25关节点)
        base_skeleton = np.random.randn(25, 3) * 0.2
        
        # 根据动作调整骨架姿态
        if "sitting" in action_name or "down" in action_name:
            # 坐下动作：降低高度，腿部弯曲
            base_skeleton[:, 1] -= 0.3  # 降低Y坐标
            base_skeleton[12:20, :] *= 0.7  # 腿部收缩
            
        elif "jump" in action_name or "up" in action_name:
            # 跳跃动作：抬高，手臂上举
            base_skeleton[:, 1] += 0.2  # 抬高Y坐标
            base_skeleton[4:12, 1] += 0.3  # 手臂上举
            
        elif "clapping" in action_name:
            # 拍手：双手接近
            base_skeleton[7, 0] = -0.1   # 左手向中心
            base_skeleton[11, 0] = 0.1   # 右手向中心
            
        elif "waving" in action_name:
            # 挥手：一只手臂抬起
            base_skeleton[7:9, 1] += 0.4  # 抬起一只手臂
            
        return base_skeleton
            
    def run_cli_annotation(self):
        """运行命令行标注界面"""
        print("\n🏷️ 码本动作标注工具 (命令行模式)")
        print("=" * 60)
        
        while True:
            self.show_cli_menu()
            choice = safe_input("请选择操作: ", "0")
            
            if choice == '1':
                self.generate_sample_data()
            elif choice == '2':
                self.load_real_data("ntu")
            elif choice == '3':
                self.load_real_data("radar_gt")
            elif choice == '4':
                self.load_real_data("both")
            elif choice == '5':
                self.load_real_data("mars_tokens")
            elif choice == '6':
                self.annotate_samples_cli()
            elif choice == '7':
                self.batch_annotate_cli()
            elif choice == '8':
                self.show_progress_cli()
            elif choice == '9':
                self.export_annotations()
            elif choice == '10':
                self.load_previous_session()
            elif choice == 'a' or choice == 'A':
                self.token_analysis_cli()
            elif choice == 's' or choice == 'S':
                self.sequence_frame_annotation()
            elif choice == 'v' or choice == 'V':
                self.open_visualization_window()
            elif choice == '0':
                break
            else:
                print("❌ 无效选择，请重试")
                
        print("👋 标注工具退出")
        
    def show_cli_menu(self):
        """显示命令行菜单"""
        print("\n📋 操作菜单:")
        print("1. 生成示例数据")
        print("2. 加载NTU数据集") 
        print("3. 加载雷达Ground Truth数据")
        print("4. 加载混合数据 (NTU + 雷达GT)")
        print("5. 加载MARS Token数据集 (推荐，包含Token序列)")
        print("6. 开始标注样本")
        print("7. 智能批量标注 (🔥推荐)")
        print("8. 查看标注进度")
        print("9. 导出标注结果")
        print("10. 加载之前的会话")
        print("a. Token分析与采样策略")
        print("s. 序列帧批注 (MARS推荐)")
        print("v. 打开3D可视化窗口")
        print("0. 退出")
        
    def annotate_samples_cli(self):
        """命令行模式标注样本 - 包含可视化"""
        if not self.samples_to_annotate:
            print("❌ 暂无样本数据，请先生成或加载数据")
            return
            
        print(f"\n🏷️ 开始标注 ({len(self.samples_to_annotate)} 个样本)")
        
        for i, sample in enumerate(self.samples_to_annotate):
            if sample.get('annotated', False):
                continue
                
            print(f"\n" + "="*80)
            print(f"📋 样本 {i+1}/{len(self.samples_to_annotate)}")
            print(f"📁 文件: {sample.get('filename', '未知')}")
            print(f"🎯 Token序列: {sample['tokens']}")
            
            # 显示Ground Truth信息
            if 'ground_truth_action' in sample:
                print(f"🎭 Ground Truth: {sample['ground_truth_action']}")
            
            # 可视化样本数据
            print(f"\n📊 数据可视化:")
            self._visualize_sample_data_cli(sample)
            
            # 选择标注模式或操作
            while True:
                print(f"\n🏷️ 操作选择:")
                print("1. 详细分部位标注 (推荐，准确性高)")
                print("2. 快速整体标注 (速度快)")
                if GIF_AVAILABLE:
                    print("g. 查看相邻帧GIF动画 (时序上下文)")
                print("3. 跳过此样本")
                print("4. 退出标注")
                
                mode_choice = safe_input("选择操作 (1-4/g): ", "4")
                
                if mode_choice.lower() == 'g' and GIF_AVAILABLE:
                    # 生成并显示GIF
                    print(f"\n🎬 生成相邻帧GIF动画...")
                    num_frames = safe_input("包含多少帧? (默认5): ", "5")
                    try:
                        num_frames = int(num_frames)
                        num_frames = max(3, min(num_frames, 11))  # 限制在3-11帧
                    except:
                        num_frames = 5
                    
                    gif_path = self._generate_adjacent_frames_gif(i, num_frames=num_frames)
                    if gif_path:
                        print(f"✅ GIF已保存到: {gif_path}")
                        print(f"💡 提示: 可以用图片查看器打开查看动画")
                        input("按回车继续...")
                    continue  # 返回菜单
                    
                elif mode_choice == '1':
                    success = self._detailed_part_annotation_cli(sample)
                    break
                elif mode_choice == '2':
                    success = self._quick_overall_annotation_cli(sample)
                    break
                elif mode_choice == '3':
                    print("⏭️ 跳过样本")
                    success = False
                    break
                elif mode_choice == '4' or mode_choice == "":
                    print("🚪 退出标注")
                    return
                else:
                    print("❌ 无效选择，请重新输入")
                    # 防止无限循环
                    if mode_choice == "":
                        print("🚪 输入为空，退出标注")
                        return
                    continue
            
            if success:
                sample['annotated'] = True
                print(f"✅ 样本 {i+1} 标注完成")
            
            # 询问是否继续
            if i < len(self.samples_to_annotate) - 1:
                continue_choice = safe_input(f"\n继续标注下一个样本? (y/n, 默认y): ", "y").lower()
                if continue_choice == 'n':
                    break
        
        print(f"\n🎉 标注会话结束")
    
    def _visualize_sample_data_cli(self, sample):
        """CLI模式下可视化样本数据"""
        try:
            # 首先尝试打开独立可视化窗口
            if VISUALIZATION_AVAILABLE:
                print("🖼️ 打开3D可视化窗口...")
                success = show_sample_visualization(sample, sample)
                if success:
                    print("✅ 可视化窗口已打开，请查看独立窗口")
                    # 等待用户确认看到了可视化
                    input("👀 请查看可视化窗口，确认后按回车继续...")
                else:
                    print("⚠️ 可视化窗口打开失败，使用文本描述")
                    self._show_text_visualization(sample)
            else:
                print("📊 使用文本模式显示数据信息:")
                self._show_text_visualization(sample)
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")
            self._show_text_visualization(sample)
    
    def _show_text_visualization(self, sample):
        """显示文本模式的数据信息"""
        if 'point_cloud_data' in sample:
            self._show_point_cloud_info(sample['point_cloud_data'], sample)
        elif 'radar_data' in sample:
            self._show_radar_info(sample['radar_data'], sample)
        elif 'skeleton_data' in sample:
            self._show_skeleton_info(sample['skeleton_data'], sample)
        elif 'extracted' in sample or 'reconstructed' in sample:
            # MARS Token 数据集格式: extracted/reconstructed 骨架
            skeleton = sample.get('reconstructed', sample.get('extracted'))
            self._show_skeleton_info(skeleton, sample)
        else:
            self._show_basic_info(sample)
    
    def _show_point_cloud_info(self, point_cloud_data, sample):
        """显示点云数据信息"""
        print(f"☁️ 点云数据 (形状: {point_cloud_data.shape})")
        
        # 计算整体统计
        min_coords = np.min(point_cloud_data, axis=0)
        max_coords = np.max(point_cloud_data, axis=0)
        center = np.mean(point_cloud_data, axis=0)
        
        print(f"   📏 边界: X[{min_coords[0]:.2f}~{max_coords[0]:.2f}] Y[{min_coords[1]:.2f}~{max_coords[1]:.2f}] Z[{min_coords[2]:.2f}~{max_coords[2]:.2f}]")
        print(f"   📍 中心: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        # 显示5个身体部位的特征
        points_per_part = len(point_cloud_data) // 5
        print(f"   🦴 各部位分析:")
        
        for i, part_name in enumerate(['头颈', '左臂', '右臂', '左腿', '右腿']):
            start_idx = i * points_per_part
            end_idx = start_idx + points_per_part if i < 4 else len(point_cloud_data)
            part_points = point_cloud_data[start_idx:end_idx]
            
            if len(part_points) > 0:
                part_center = np.mean(part_points, axis=0)
                part_spread = np.std(part_points, axis=0)
                print(f"     {part_name}: 中心({part_center[0]:.1f},{part_center[1]:.1f},{part_center[2]:.1f}) "
                      f"分布({part_spread[0]:.1f},{part_spread[1]:.1f},{part_spread[2]:.1f})")
    
    def _show_radar_info(self, radar_data, sample):
        """显示雷达数据信息"""
        print(f"📡 雷达数据 (形状: {radar_data.shape})")
        print(f"   📊 值域: [{np.min(radar_data):.3f} ~ {np.max(radar_data):.3f}]")
        print(f"   📈 均值: {np.mean(radar_data):.3f}, 标准差: {np.std(radar_data):.3f}")
        
        # 如果是特征图，分析各通道
        if len(radar_data.shape) == 3 and radar_data.shape[2] == 5:
            print(f"   🔬 通道分析:")
            for ch in range(5):
                ch_data = radar_data[:, :, ch]
                print(f"     通道{ch}: [{np.min(ch_data):.2f}~{np.max(ch_data):.2f}] 均值{np.mean(ch_data):.2f}")
    
    def _show_skeleton_info(self, skeleton_data, sample):
        """显示骨架数据信息"""
        print(f"🦴 骨架数据 (形状: {skeleton_data.shape})")
        if len(skeleton_data.shape) == 2 and skeleton_data.shape[1] == 3:
            print(f"   🔗 关节点数: {skeleton_data.shape[0]}")
            
            # 分析各身体部位 (基于NTU 25关节标准)
            joint_groups = {
                '头颈': [0, 1, 2, 3, 20],
                '左臂': [4, 5, 6, 7, 21], 
                '右臂': [8, 9, 10, 11, 22],
                '左腿': [12, 13, 14, 15, 23],
                '右腿': [16, 17, 18, 19, 24]
            }
            
            for part_name, joint_indices in joint_groups.items():
                valid_joints = [idx for idx in joint_indices if idx < skeleton_data.shape[0]]
                if valid_joints:
                    part_joints = skeleton_data[valid_joints]
                    center = np.mean(part_joints, axis=0)
                    print(f"     {part_name}: 中心({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})")
    
    def _show_basic_info(self, sample):
        """显示基础样本信息"""
        print(f"📋 基础信息:")
        for key, value in sample.items():
            if key not in ['point_cloud_data', 'radar_data', 'skeleton_data', 'tokens', 'extracted', 'reconstructed']:
                if isinstance(value, (int, float, str, bool)):
                    print(f"   {key}: {value}")
    
    # ==================== GIF 可视化功能 ====================
    
    def _generate_adjacent_frames_gif(self, sample_idx, num_frames=5, output_dir='temp_gifs'):
        """生成相邻几帧的GIF动画
        
        Args:
            sample_idx: 当前样本索引
            num_frames: GIF中包含的总帧数 (建议奇数,以当前样本为中心)
            output_dir: GIF保存目录
            
        Returns:
            GIF文件路径，如果失败返回None
        """
        if not GIF_AVAILABLE:
            print("❌ matplotlib不可用，无法生成GIF")
            return None
        
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 计算相邻样本索引范围
            half = num_frames // 2
            start_idx = max(0, sample_idx - half)
            end_idx = min(len(self.samples_to_annotate), sample_idx + half + 1)
            
            # 收集骨架帧数据
            skeleton_frames = []
            frame_labels = []
            
            for idx in range(start_idx, end_idx):
                sample = self.samples_to_annotate[idx]
                
                # 优先使用 reconstructed，其次 extracted，最后 skeleton_data
                skeleton = None
                data_source = ""
                
                if 'reconstructed' in sample:
                    skeleton = sample['reconstructed']
                    data_source = "reconstructed"
                elif 'extracted' in sample:
                    skeleton = sample['extracted']
                    data_source = "extracted"
                elif 'skeleton_data' in sample:
                    skeleton = sample['skeleton_data']
                    data_source = "skeleton_data"
                
                if skeleton is not None:
                    # 转换为 (num_joints, 3) 格式
                    if len(skeleton.shape) == 2 and skeleton.shape[1] == 3:
                        skeleton_frames.append(skeleton)
                        is_current = "★" if idx == sample_idx else ""
                        frame_labels.append(f"Sample {idx} {is_current} ({data_source})")
                    else:
                        print(f"⚠️ 样本 {idx} 骨架格式异常: {skeleton.shape}")
            
            if len(skeleton_frames) == 0:
                print("❌ 没有找到可用的骨架数据")
                return None
            
            # 生成GIF文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = os.path.join(output_dir, f"sample_{sample_idx:05d}_{timestamp}.gif")
            
            # 调用GIF生成函数
            success = self._create_skeleton_sequence_gif(
                skeleton_frames, 
                frame_labels, 
                gif_path,
                title=f"Adjacent Frames Around Sample {sample_idx}"
            )
            
            if success:
                print(f"✅ GIF已生成: {gif_path}")
                return gif_path
            else:
                print(f"❌ GIF生成失败")
                return None
                
        except Exception as e:
            print(f"❌ 生成GIF时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_skeleton_sequence_gif(self, skeleton_frames, frame_labels, output_path, 
                                     title="Skeleton Sequence", fps=2):
        """创建骨架序列GIF动画
        
        Args:
            skeleton_frames: List of (num_joints, 3) numpy arrays
            frame_labels: List of frame label strings
            output_path: GIF输出路径
            title: GIF标题
            fps: 帧率
            
        Returns:
            成功返回True，失败返回False
        """
        if not GIF_AVAILABLE:
            return False
        
        try:
            # 检测骨架类型并使用对应的连接
            num_joints = skeleton_frames[0].shape[0]
            
            if num_joints == 19:
                # MARS 19关节骨架连接定义 (0-based索引)
                skeleton_connections = [
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
                skeleton_type = "MARS (19 joints)"
            elif num_joints == 25:
                # NTU RGB+D 25关节骨架连接定义
                # 参考 tools/analyze_ntu_skeleton.py 的标准定义
                skeleton_connections = [
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
                skeleton_type = "NTU RGB+D (25 joints)"
            else:
                print(f"⚠️ 未知骨架类型: {num_joints} 关节，使用简单连接")
                # 创建简单的顺序连接
                skeleton_connections = [(i, i+1) for i in range(num_joints-1)]
                skeleton_type = f"Unknown ({num_joints} joints)"
            
            # 计算所有帧的数据边界
            all_joints = np.vstack(skeleton_frames)
            x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
            y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()
            z_min, z_max = all_joints[:, 2].min(), all_joints[:, 2].max()
            
            # 计算统一范围
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            margin = max_range * 0.2
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2
            half_range = max_range / 2 + margin
            
            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            fig.suptitle(f'{title} - {skeleton_type}', fontsize=14, fontweight='bold')
            ax = fig.add_subplot(111, projection='3d')
            
            # 设置轴属性
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)
            ax.set_xlim([x_center - half_range, x_center + half_range])
            ax.set_ylim([y_center - half_range, y_center + half_range])
            ax.set_zlim([z_center - half_range, z_center + half_range])
            ax.view_init(elev=20, azim=45)
            
            # 添加帧信息文本
            frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10, fontweight='bold')
            
            def animate(frame_idx):
                """动画更新函数"""
                ax.clear()
                
                joints = skeleton_frames[frame_idx]
                
                # 翻转Z轴让骨架正立显示 (数据中头部Z值 < 脚部Z值，是倒立的)
                # 参考 vis_gif_skeleton_extractor.py 的处理方式
                joints_display = joints.copy()
                joints_display[:, 2] = -joints_display[:, 2]  # 翻转Z轴
                
                # 绘制关节点
                ax.scatter(joints_display[:, 0], joints_display[:, 1], joints_display[:, 2],
                          c='blue', s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
                
                # 绘制骨架连接线
                for connection in skeleton_connections:
                    if connection[0] < len(joints_display) and connection[1] < len(joints_display):
                        joint1 = joints_display[connection[0]]
                        joint2 = joints_display[connection[1]]
                        ax.plot([joint1[0], joint2[0]],
                               [joint1[1], joint2[1]],
                               [joint1[2], joint2[2]],
                               color='blue', alpha=0.7, linewidth=2)
                
                # 重新设置轴属性
                ax.set_xlabel('X', fontsize=10)
                ax.set_ylabel('Y', fontsize=10)
                ax.set_zlabel('Z (Up)', fontsize=10)
                ax.set_xlim([x_center - half_range, x_center + half_range])
                ax.set_ylim([y_center - half_range, y_center + half_range])
                # Z轴范围也需要翻转
                ax.set_zlim([-(z_center + half_range), -(z_center - half_range)])
                ax.view_init(elev=20, azim=45)
                
                # 更新帧信息
                frame_text.set_text(f'{frame_labels[frame_idx]} | Frame {frame_idx+1}/{len(skeleton_frames)}')
                
                return []
            
            # 创建动画
            anim = animation.FuncAnimation(
                fig, animate, frames=len(skeleton_frames),
                interval=int(1000/fps), blit=False, repeat=True
            )
            
            # 保存GIF
            try:
                anim.save(output_path, writer='pillow', fps=fps, dpi=80)
                plt.close(fig)
                return True
            except Exception as e:
                print(f"❌ 保存GIF失败: {e}")
                plt.close(fig)
                return False
                
        except Exception as e:
            print(f"❌ 创建GIF动画失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== 标注功能 ====================

    
    def _detailed_part_annotation_cli(self, sample):
        """CLI详细分部位标注"""
        print(f"\n🔍 详细分部位标注模式")
        print("=" * 50)
        
        annotations = {}
        tokens = sample.get('tokens', [])
        
        if len(tokens) != 5:
            print(f"❌ Token数量异常: {len(tokens)}, 期望5个")
            return False
        
        part_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        part_display = ['头颈', '左臂', '右臂', '左腿', '右腿']
        
        for i, (part, display, token) in enumerate(zip(part_names, part_display, tokens)):
            print(f"\n🦴 标注 {display} (Token: {token}) - [{i+1}/5]")
            print("-" * 30)
            
            # 使用 mars_action_templates 的动作选项
            common_actions = self.action_templates[part]
            
            print("常见动作:")
            for j, action in enumerate(common_actions, 1):
                print(f"  {j:2d}. {action}")
            
            while True:
                print(f"\n输入选项:")
                print(f"1-{len(common_actions)}. 选择预设动作")
                print("c. 自定义描述")
                print("s. 跳过此部位")
                
                choice = safe_input(f"请选择: ", "s").lower()
                
                if choice == 's' or choice == "":
                    print(f"⏭️ 跳过 {display}")
                    break
                elif choice == 'c':
                    custom_desc = safe_input(f"请描述{display}的动作: ", "正常状态")
                    if custom_desc and custom_desc != "正常状态":
                        annotations[part] = {
                            'token': token,
                            'description': custom_desc,
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"✅ {display}: {custom_desc}")
                        break
                    else:
                        # 使用默认描述
                        annotations[part] = {
                            'token': token,
                            'description': "正常状态",
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"✅ {display}: 正常状态 (默认)")
                        break
                else:
                    try:
                        action_idx = int(choice) - 1
                        if 0 <= action_idx < len(common_actions):
                            selected_action = common_actions[action_idx]
                            annotations[part] = {
                                'token': token,
                                'description': selected_action,
                                'timestamp': datetime.now().isoformat()
                            }
                            print(f"✅ {display}: {selected_action}")
                            break
                        else:
                            print(f"❌ 请输入1-{len(common_actions)}之间的数字")
                    except ValueError:
                        print("❌ 无效输入，请重试")
        
        # 整体动作描述
        print(f"\n🎭 整体动作描述:")
        if 'ground_truth_action' in sample:
            print(f"💡 参考GT: {sample['ground_truth_action']}")
        
        overall_action = safe_input("请描述整体动作 (可参考GT): ", sample.get('ground_truth_action', '未描述'))
        if not overall_action:
            overall_action = sample.get('ground_truth_action', '未描述')
        
        # 保存标注
        sample['annotations'] = annotations
        sample['overall_action'] = overall_action
        sample['annotation_time'] = datetime.now().isoformat()
        
        # 自动保存到会话文件
        self._auto_save_sample(sample)
        
        print(f"\n✅ 详细标注完成，共标注 {len(annotations)} 个部位")
        print(f"💾 标注结果已自动保存")
        return True
    
    def _quick_overall_annotation_cli(self, sample):
        """CLI快速整体标注（仅标注整体动作，部位使用默认描述）"""
        print(f"\n⚡ 快速整体标注模式")
        print("=" * 30)
        
        # 显示参考信息
        if 'ground_truth_action' in sample:
            print(f"🎯 Ground Truth: {sample['ground_truth_action']}")
        
        default_action = sample.get('ground_truth_action', '未描述')
        overall_action = safe_input("请描述整体动作: ", default_action)
        if not overall_action:
            overall_action = default_action
        
        # 自动为各部位生成简单默认描述
        tokens = sample.get('tokens', [])
        if len(tokens) != 5:
            print(f"❌ Token数量异常")
            return False
        
        annotations = {}
        part_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        part_defaults = ['正常姿态', '自然状态', '自然状态', '站立支撑', '站立支撑']
        
        for part, token, default_desc in zip(part_names, tokens, part_defaults):
            annotations[part] = {
                'token': token,
                'description': default_desc,
                'timestamp': datetime.now().isoformat(),
                'auto_generated': True
            }
        
        # 保存标注
        sample['annotations'] = annotations
        sample['overall_action'] = overall_action
        sample['annotation_time'] = datetime.now().isoformat()
        
        # 自动保存到会话文件
        self._auto_save_sample(sample)
        
        print(f"✅ 快速标注完成")
        print(f"💾 标注结果已自动保存")
        return True
    
    def _auto_save_sample(self, sample):
        """自动保存单个样本标注"""
        try:
            # 会话文件路径
            session_file = os.path.join(self.save_dir, "sessions", f"session_{self.session_id}.json")
            
            # 加载现有会话数据
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
            else:
                session_data = {
                    'session_id': self.session_id,
                    'created_time': datetime.now().isoformat(),
                    'samples': {},
                    'statistics': {
                        'total_samples': 0,
                        'annotated_samples': 0,
                        'annotation_modes': {}
                    }
                }
            
            # 添加/更新样本数据
            sample_key = f"sample_{sample.get('id', len(session_data['samples']))}"
            
            # 创建清理后的样本数据（移除大数据对象）
            clean_sample = {}
            for key, value in sample.items():
                # 排除numpy数组数据字段
                if key not in ['point_cloud_data', 'radar_data', 'skeleton_data', 'extracted', 'reconstructed']:
                    # 转换numpy类型为Python原生类型
                    if isinstance(value, np.ndarray):
                        clean_sample[f"{key}_summary"] = {
                            'shape': list(value.shape),
                            'dtype': str(value.dtype),
                            'size': int(value.size)
                        }
                    elif isinstance(value, (np.integer, np.floating)):
                        clean_sample[key] = int(value) if isinstance(value, np.integer) else float(value)
                    elif isinstance(value, list):
                        # 转换列表中的numpy类型
                        clean_sample[key] = [int(x) if isinstance(x, np.integer) else 
                                            float(x) if isinstance(x, np.floating) else x 
                                            for x in value]
                    else:
                        clean_sample[key] = value
                else:
                    # 只保存数据概要信息
                    if isinstance(value, np.ndarray):
                        clean_sample[f"{key}_summary"] = {
                            'shape': list(value.shape),
                            'dtype': str(value.dtype),
                            'size': int(value.size)
                        }
            
            session_data['samples'][sample_key] = clean_sample
            session_data['last_updated'] = datetime.now().isoformat()
            
            # 更新统计信息
            session_data['statistics']['total_samples'] = len(session_data['samples'])
            session_data['statistics']['annotated_samples'] = len([s for s in session_data['samples'].values() if s.get('annotated', False)])
            
            # 保存会话文件
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"⚠️ 自动保存失败: {e}")
            return False
    
    def export_annotations(self):
        """导出标注结果"""
        if not any(s.get('annotated', False) for s in self.samples_to_annotate):
            print("❌ 暂无已标注的样本数据")
            return
        
        print("\n📤 导出标注结果")
        print("=" * 40)
        
        # 选择导出格式
        print("选择导出格式:")
        print("1. JSON格式 (完整数据)")
        print("2. CSV格式 (表格数据)")
        print("3. 码本映射表 (Token->Action)")
        print("4. 全部格式")
        
        choice = safe_input("选择导出格式 (1-4): ", "1")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_base_name = f"annotations_export_{timestamp}"
        
        try:
            if choice in ['1', '4']:
                self._export_json(export_base_name)
            if choice in ['2', '4']:
                self._export_csv(export_base_name)
            if choice in ['3', '4']:
                self._export_mapping_table(export_base_name)
            
            print(f"✅ 导出完成，文件保存在: {self.save_dir}/exports/")
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def _export_json(self, base_name):
        """导出JSON格式"""
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'total_samples': len(self.samples_to_annotate),
                'annotated_samples': len([s for s in self.samples_to_annotate if s.get('annotated', False)])
            },
            'samples': []
        }
        
        for sample in self.samples_to_annotate:
            if sample.get('annotated', False):
                # 清理样本数据
                clean_sample = {}
                for key, value in sample.items():
                    if key not in ['point_cloud_data', 'radar_data', 'skeleton_data']:
                        clean_sample[key] = value
                export_data['samples'].append(clean_sample)
        
        json_file = os.path.join(self.save_dir, "exports", f"{base_name}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 JSON文件已导出: {json_file}")
    
    def _export_csv(self, base_name):
        """导出CSV格式"""
        import csv
        
        csv_file = os.path.join(self.save_dir, "exports", f"{base_name}.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = [
                'sample_id', 'filename', 'source', 'ground_truth_action', 'overall_action',
                'head_spine_token', 'head_spine_desc',
                'left_arm_token', 'left_arm_desc',
                'right_arm_token', 'right_arm_desc', 
                'left_leg_token', 'left_leg_desc',
                'right_leg_token', 'right_leg_desc',
                'annotation_time'
            ]
            writer.writerow(headers)
            
            # 写入数据行
            for sample in self.samples_to_annotate:
                if sample.get('annotated', False):
                    annotations = sample.get('annotations', {})
                    tokens = sample.get('tokens', [0, 0, 0, 0, 0])
                    
                    row = [
                        sample.get('id', ''),
                        sample.get('filename', ''),
                        sample.get('source', ''),
                        sample.get('ground_truth_action', ''),
                        sample.get('overall_action', ''),
                    ]
                    
                    # 添加各部位的token和描述
                    part_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
                    for i, part in enumerate(part_names):
                        token = tokens[i] if i < len(tokens) else 0
                        desc = annotations.get(part, {}).get('description', '')
                        row.extend([token, desc])
                    
                    row.append(sample.get('annotation_time', ''))
                    writer.writerow(row)
        
        print(f"📊 CSV文件已导出: {csv_file}")
    
    def _export_mapping_table(self, base_name):
        """导出码本映射表"""
        # 构建Token到动作的映射
        token_mapping = {}
        
        for sample in self.samples_to_annotate:
            if sample.get('annotated', False):
                annotations = sample.get('annotations', {})
                tokens = sample.get('tokens', [])
                
                part_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
                for i, part in enumerate(part_names):
                    if i < len(tokens):
                        token = tokens[i]
                        desc = annotations.get(part, {}).get('description', '')
                        
                        if token not in token_mapping:
                            token_mapping[token] = {}
                        if part not in token_mapping[token]:
                            token_mapping[token][part] = []
                        
                        if desc and desc not in token_mapping[token][part]:
                            token_mapping[token][part].append(desc)
        
        # 导出映射表
        mapping_file = os.path.join(self.save_dir, "exports", f"{base_name}_mapping.json")
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(token_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"🗂️ 码本映射表已导出: {mapping_file}")
        
        # 同时生成可读的映射表
        readable_file = os.path.join(self.save_dir, "exports", f"{base_name}_mapping.txt")
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write("码本-动作映射表\n")
            f.write("=" * 50 + "\n\n")
            
            for token in sorted(token_mapping.keys()):
                f.write(f"Token {token}:\n")
                for part, descriptions in token_mapping[token].items():
                    f.write(f"  {part}: {', '.join(descriptions)}\n")
                f.write("\n")
        
        print(f"📖 可读映射表已导出: {readable_file}")
    
    def show_progress_cli(self):
        """显示标注进度"""
        total_samples = len(self.samples_to_annotate)
        if total_samples == 0:
            print("❌ 暂无样本数据")
            return
        
        annotated_samples = len([s for s in self.samples_to_annotate if s.get('annotated', False)])
        progress_percent = (annotated_samples / total_samples) * 100 if total_samples > 0 else 0
        
        print("\n📊 标注进度统计")
        print("=" * 40)
        print(f"总样本数: {total_samples}")
        print(f"已标注数: {annotated_samples}")
        print(f"完成进度: {progress_percent:.1f}%")
        
        # 进度条
        bar_length = 30
        filled_length = int(bar_length * annotated_samples // total_samples)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"进度条: |{bar}| {progress_percent:.1f}%")
        
        # 数据源统计
        source_stats = {}
        for sample in self.samples_to_annotate:
            source = sample.get('source', 'unknown')
            source_stats[source] = source_stats.get(source, 0) + 1
        
        print(f"\n📋 数据源分布:")
        for source, count in source_stats.items():
            print(f"  {source}: {count} 个样本")
        
        # 最近标注活动
        recent_annotations = [s for s in self.samples_to_annotate 
                            if s.get('annotated', False) and 'annotation_time' in s]
        recent_annotations.sort(key=lambda x: x['annotation_time'], reverse=True)
        
        if recent_annotations:
            print(f"\n🕐 最近标注活动:")
            for i, sample in enumerate(recent_annotations[:5]):
                time_str = sample['annotation_time'][:19].replace('T', ' ')
                filename = sample.get('filename', 'unknown')[:20]
                print(f"  {i+1}. {time_str} - {filename}")
    
    def load_previous_session(self):
        """加载之前的标注会话"""
        sessions_dir = os.path.join(self.save_dir, "sessions")
        if not os.path.exists(sessions_dir):
            print("❌ 暂无保存的会话")
            return
        
        session_files = [f for f in os.listdir(sessions_dir) if f.endswith('.json')]
        if not session_files:
            print("❌ 暂无保存的会话文件")
            return
        
        print(f"\n📂 发现 {len(session_files)} 个会话文件:")
        for i, session_file in enumerate(session_files, 1):
            session_path = os.path.join(sessions_dir, session_file)
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                created_time = session_data.get('created_time', '未知')[:19].replace('T', ' ')
                sample_count = len(session_data.get('samples', {}))
                print(f"  {i}. {session_file} - {created_time} ({sample_count} 样本)")
            except:
                print(f"  {i}. {session_file} - 损坏的文件")
        
        choice = safe_input(f"\n选择要加载的会话 (1-{len(session_files)}, 0=取消): ", "0")
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(session_files):
                session_file = session_files[choice_idx]
                session_path = os.path.join(sessions_dir, session_file)
                
                with open(session_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # 恢复会话数据
                self.session_id = session_data['session_id']
                print(f"✅ 已加载会话: {session_file}")
                print(f"📊 包含 {len(session_data['samples'])} 个样本")
                
        except ValueError:
            print("❌ 无效选择")
        except Exception as e:
            print(f"❌ 加载会话失败: {e}")
    
    def open_visualization_window(self):
        """手动打开可视化窗口"""
        if not VISUALIZATION_AVAILABLE:
            print("❌ 可视化窗口不可用")
            return
        
        if not self.samples_to_annotate:
            print("❌ 暂无样本数据，请先加载数据")
            return
        
        print("\n🖼️ 选择要可视化的样本:")
        
        # 显示样本并标记数据集类型
        for i, sample in enumerate(self.samples_to_annotate[:10]):  # 只显示前10个
            filename = sample.get('filename', f'sample_{i}')
            source = sample.get('source', 'unknown')
            status = "✅" if sample.get('annotated', False) else "⭕"
            
            # 数据集类型标识
            if 'ntu' in source.lower():
                dataset_tag = "[NTU]"
            elif 'mars' in source.lower():
                dataset_tag = "[MARS]"
            else:
                dataset_tag = "[UNKNOWN]"
                
            print(f"  {i+1}. {status} {dataset_tag} {filename}")
        
        if len(self.samples_to_annotate) > 10:
            print(f"  ... 和其他 {len(self.samples_to_annotate)-10} 个样本")
        
        choice = safe_input(f"选择样本 (1-{min(10, len(self.samples_to_annotate))}): ", "1")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.samples_to_annotate):
                sample = self.samples_to_annotate[idx]
                
                # 显示将要可视化的样本信息
                source = sample.get('source', 'unknown')
                print(f"\n🔍 即将可视化: {sample.get('filename', 'unknown')}")
                print(f"📊 数据集类型: {source}")
                
                if 'skeleton_data' in sample:
                    joints_count = sample['skeleton_data'].shape[0]
                    print(f"🦴 骨架关节数: {joints_count}")
                    if joints_count == 25:
                        print("💡 这是NTU RGB+D 25关节骨架数据")
                    elif joints_count == 19:
                        print("💡 这是MARS 19关节骨架数据") 
                elif 'point_cloud_data' in sample:
                    points_count = sample['point_cloud_data'].shape[0]
                    print(f"☁️ 点云数据点数: {points_count}")
                elif 'radar_data' in sample:
                    print("📡 这是雷达特征数据")
                
                success = show_sample_visualization(sample, sample)
                if success:
                    print("✅ 可视化窗口已打开")
                    print("💡 您可以在可视化窗口中查看3D骨架数据")
                    print("💡 关闭可视化窗口后会自动返回主菜单")
                    
                    # 等待用户确认或可视化窗口关闭
                    input("\n👀 查看完毕后请按回车键返回主菜单...")
                else:
                    print("❌ 可视化窗口打开失败")
            else:
                print("❌ 无效的样本编号")
        except ValueError:
            print("❌ 请输入有效数字")
        
    def sequence_frame_annotation(self):
        """序列帧批注功能 - 可视化相邻几帧动作，批量标注整个序列"""
        if not self.samples_to_annotate:
            print("❌ 暂无样本数据")
            return
            
        print("\n🎬 序列帧批注模式")
        print("=" * 60)
        print("💡 此模式适合MARS数据集：可视化相邻帧，批量标注整个动作序列")
        
        # 配置序列参数
        try:
            sequence_length = int(safe_input("请输入序列长度 (建议3-8帧): ", "5"))
            sequence_length = max(2, min(10, sequence_length))  # 限制在2-10帧
            
            step_size = int(safe_input("请输入步长 (跳过多少帧): ", "1"))
            step_size = max(1, step_size)
            
        except ValueError:
            sequence_length = 5
            step_size = 1
            print(f"使用默认参数: 序列长度={sequence_length}, 步长={step_size}")
        
        print(f"\n📋 序列配置: {sequence_length}帧/序列, 步长={step_size}")
        
        # 按序列处理样本
        sequence_count = 0
        total_sequences = (len(self.samples_to_annotate) - sequence_length + 1) // step_size
        
        i = 0
        while i <= len(self.samples_to_annotate) - sequence_length:
            sequence_count += 1
            
            # 提取当前序列
            current_sequence = self.samples_to_annotate[i:i+sequence_length]
            
            print(f"\n" + "="*80)
            print(f"🎬 序列 {sequence_count}/{total_sequences}")
            print(f"📁 帧范围: {i+1} - {i+sequence_length}")
            
            # 显示序列中每一帧的信息
            print("序列帧信息:")
            sequence_tokens = []
            for j, frame in enumerate(current_sequence):
                status = "✓已标注" if frame.get('annotated', False) else "○未标注"
                print(f"  帧{i+j+1}: {frame['tokens']} {status}")
                sequence_tokens.append(tuple(frame['tokens']))
            
            # 检查序列是否已全部标注
            already_annotated = all(frame.get('annotated', False) for frame in current_sequence)
            if already_annotated:
                print("✓ 此序列已完全标注，跳过")
                i += step_size
                continue
            
            # 显示可视化选项
            print(f"\n🎯 序列批注选项:")
            print("1. 可视化此序列")
            print("2. 直接标注此序列")
            print("3. 跳过此序列")
            print("4. 退出序列标注")
            
            choice = safe_input("请选择操作: ", "1").strip()
            
            if choice == '1':
                # 可视化序列
                self._visualize_sequence(current_sequence, i)
                
                # 可视化后询问是否标注
                annotate_choice = safe_input("是否标注此序列? (y/n): ", "n").strip().lower()
                if annotate_choice in ['y', 'yes']:
                    self._annotate_sequence(current_sequence, i)
                    
            elif choice == '2':
                # 直接标注序列
                self._annotate_sequence(current_sequence, i)
                
            elif choice == '3':
                # 跳过序列
                print("⏭️ 跳过序列")
                
            elif choice == '4':
                # 退出
                print("🚪 退出序列标注模式")
                break
            else:
                print("❌ 无效选择，跳过此序列")
            
            i += step_size
            
        print(f"\n✅ 序列标注模式结束，共处理 {sequence_count} 个序列")
        
    def _visualize_sequence(self, sequence, start_index):
        """可视化动作序列"""
        print(f"\n🖼️ 可视化序列 (帧 {start_index+1}-{start_index+len(sequence)})")
        
        # 尝试使用可视化窗口
        if VISUALIZATION_AVAILABLE:
            try:
                # 尝试初始化可视化窗口（如果尚未初始化）
                if not hasattr(self, 'visualization_window') or not self.visualization_window:
                    print("🔧 初始化可视化窗口...")
                    self._init_visualization_window()
                
                # 检查是否有可用的可视化窗口
                if hasattr(self, 'visualization_window') and self.visualization_window:
                    # 依次显示序列中的每一帧
                    for j, frame in enumerate(sequence):
                        if 'skeleton' in frame:
                            print(f"显示第 {j+1} 帧...")
                            self.visualization_window.update_skeleton(frame['skeleton'])
                            self.visualization_window.update_display()
                            
                            if j < len(sequence) - 1:  # 不是最后一帧
                                input("按Enter键查看下一帧...")
                            
                    print("序列可视化完成")
                    return
                else:
                    print("⚠️ 可视化窗口初始化失败，使用文本显示")
                
            except Exception as e:
                print(f"⚠️ 可视化失败: {e}，使用文本显示")
        else:
            print("⚠️ 可视化模块不可用，使用文本显示")
        
        # 备选方案：显示详细文本信息
        print("📊 序列帧详细信息:")
        print("=" * 60)
        
        for j, frame in enumerate(sequence):
            print(f"📋 帧 {start_index+j+1}:")
            
            # Token信息解析
            tokens = frame['tokens']
            print(f"  🎯 Token序列: {tokens}")
            print(f"     头部脊椎: {tokens[0]}")
            print(f"     左臂: {tokens[1]}")  
            print(f"     右臂: {tokens[2]}")
            print(f"     左腿: {tokens[3]}")
            print(f"     右腿: {tokens[4]}")
            
            # 骨架信息
            if 'skeleton' in frame:
                skeleton = frame['skeleton']
                if isinstance(skeleton, np.ndarray):
                    joint_count = skeleton.shape[0]
                    dataset_type = "MARS" if joint_count == 19 else ("NTU" if joint_count == 25 else "未知")
                    
                    # 计算骨架统计
                    if len(skeleton) > 0:
                        avg_pos = np.mean(skeleton, axis=0)
                        min_pos = np.min(skeleton, axis=0)
                        max_pos = np.max(skeleton, axis=0)
                        range_pos = max_pos - min_pos
                        
                        print(f"  🦴 骨架信息: {joint_count}关节点 ({dataset_type}格式)")
                        print(f"     中心位置: [{avg_pos[0]:.2f}, {avg_pos[1]:.2f}, {avg_pos[2]:.2f}]")
                        print(f"     范围: X={range_pos[0]:.2f}, Y={range_pos[1]:.2f}, Z={range_pos[2]:.2f}")
                        
                        # 显示主要关节位置（头部和骨盆）
                        if joint_count == 19:  # MARS格式
                            head_pos = skeleton[2]  # head
                            spine_pos = skeleton[0]  # spinebase
                            print(f"     头部位置: [{head_pos[0]:.2f}, {head_pos[1]:.2f}, {head_pos[2]:.2f}]")
                            print(f"     骨盆位置: [{spine_pos[0]:.2f}, {spine_pos[1]:.2f}, {spine_pos[2]:.2f}]")
                        elif joint_count == 25:  # NTU格式
                            head_pos = skeleton[3]  # head
                            spine_pos = skeleton[0]  # spinebase
                            print(f"     头部位置: [{head_pos[0]:.2f}, {head_pos[1]:.2f}, {head_pos[2]:.2f}]")
                            print(f"     骨盆位置: [{spine_pos[0]:.2f}, {spine_pos[1]:.2f}, {spine_pos[2]:.2f}]")
            
            print(f"  📁 文件: {frame.get('filename', '未知')}")
            
            if j < len(sequence) - 1:
                print("  " + "-" * 50)
        
        print("=" * 60)
        print(f"💡 提示: 这是 {len(sequence)} 帧的动作序列")
        print(f"   Token变化可以反映动作的连续性")
        print(f"   建议基于整体动作模式进行标注")
    
    def _annotate_sequence(self, sequence, start_index):
        """标注整个动作序列"""
        print(f"\n🏷️ 标注序列 (帧 {start_index+1}-{start_index+len(sequence)})")
        
        # 获取序列的统一标注
        print("为整个序列选择动作描述:")
        
        # 分部位标注
        sequence_annotation = {
            'part_annotations': {},
            'global_action': '',
            'sequence_info': {
                'start_frame': start_index + 1,
                'end_frame': start_index + len(sequence),
                'length': len(sequence)
            }
        }
        
        for part_name, display_name in zip(self.part_names, self.part_display_names):
            print(f"\n{display_name} 动作:")
            
            # 显示动作选项
            templates = self.action_templates.get(part_name, ['正常姿态'])
            for i, template in enumerate(templates, 1):
                print(f"  {i}. {template}")
            
            # 获取用户选择
            try:
                choice_input = safe_input(f"选择 {display_name} 动作 (1-{len(templates)}) 或输入自定义: ", "1")
                
                if choice_input.isdigit():
                    choice_idx = int(choice_input) - 1
                    if 0 <= choice_idx < len(templates):
                        selected_action = templates[choice_idx]
                    else:
                        selected_action = templates[0]  # 默认选择第一个
                else:
                    # 自定义输入
                    selected_action = choice_input.strip() or templates[0]
                    
                sequence_annotation['part_annotations'][part_name] = selected_action
                print(f"✓ {display_name}: {selected_action}")
                
            except (ValueError, IndexError):
                # 使用默认
                default_action = templates[0]
                sequence_annotation['part_annotations'][part_name] = default_action
                print(f"✓ {display_name}: {default_action} (默认)")
        
        # 整体动作描述
        global_action = safe_input("\n整体动作描述 (可选): ", "").strip()
        sequence_annotation['global_action'] = global_action or self._generate_action_description(sequence_annotation['part_annotations'])
        
        # 应用标注到序列中的所有帧
        annotated_count = 0
        for j, frame in enumerate(sequence):
            if not frame.get('annotated', False):  # 只标注未标注的帧
                frame_annotation = sequence_annotation.copy()
                frame_annotation['sample_id'] = frame['id']
                frame_annotation['tokens'] = frame['tokens']
                frame_annotation['timestamp'] = datetime.now().isoformat()
                frame_annotation['annotation_method'] = 'sequence_batch'
                frame_annotation['frame_in_sequence'] = j + 1
                
                self.annotation_data[frame['id']] = frame_annotation
                frame['annotated'] = True
                annotated_count += 1
                
                # 自动保存
                self._auto_save_sample(frame)
        
        print(f"\n✅ 序列标注完成！")
        print(f"   标注动作: {sequence_annotation['global_action']}")
        print(f"   应用到 {annotated_count} 帧")
        print(f"   序列范围: 帧 {start_index+1}-{start_index+len(sequence)}")

    def _init_visualization_window(self):
        """初始化可视化窗口"""
        try:
            if VISUALIZATION_AVAILABLE:
                # 尝试导入可视化窗口类
                from tools.visualization_window import VisualizationWindow
                self.visualization_window = VisualizationWindow()
                print("✅ 可视化窗口初始化成功")
                return True
        except ImportError:
            try:
                # 备选导入路径
                from visualization_window import VisualizationWindow
                self.visualization_window = VisualizationWindow()
                print("✅ 可视化窗口初始化成功")
                return True
            except ImportError:
                print("❌ 无法导入可视化窗口类")
        except Exception as e:
            print(f"❌ 可视化窗口初始化失败: {e}")
        
        self.visualization_window = None
        return False

    def batch_annotate_cli(self):
        """智能批量标注 - 基于Token聚类"""
        if not self.samples_to_annotate:
            print("❌ 暂无样本数据")
            return
            
        print("\n🔥 智能批量标注系统")
        print("=" * 70)
        print("💡 适用场景: 大规模数据集(如MARS 40k样本)")
        print("💡 标注策略: 代表性采样 + 自动推广")
        print("=" * 70)
        
        # 1. Token统计分析
        print("\n📊 步骤1: Token组合分析")
        token_groups = self._analyze_token_patterns()
        
        if not token_groups:
            print("❌ Token分析失败")
            return
        
        print(f"\n✅ 发现 {len(token_groups)} 个不同的Token组合")
        print(f"   覆盖 {sum(len(g['samples']) for g in token_groups.values())} 个样本")
        
        # 2. 显示Top Token组合
        print("\n📈 Top 20 最常见Token组合:")
        sorted_groups = sorted(token_groups.items(), 
                              key=lambda x: len(x[1]['samples']), 
                              reverse=True)
        
        for i, (token_key, group_info) in enumerate(sorted_groups[:20], 1):
            count = len(group_info['samples'])
            percentage = count / len(self.samples_to_annotate) * 100
            annotated = group_info['annotated_count']
            status = f"✅ 已标注{annotated}" if annotated > 0 else "⭕ 未标注"
            print(f"  {i:2d}. {token_key} → {count:5d}样本 ({percentage:5.2f}%) {status}")
        
        # 3. 标注策略选择
        print("\n🎯 标注策略:")
        print("1. 按频率标注 - 优先标注最常见的组合(覆盖率高)")
        print("2. 采样标注 - 每个组合标注1个代表样本")
        print("3. 自定义 - 选择特定Token组合批量标注")
        print("4. 返回主菜单")
        
        choice = safe_input("请选择策略 (1-4): ", "1")
        
        if choice == '1':
            self._annotate_by_frequency(sorted_groups)
        elif choice == '2':
            self._annotate_by_sampling(sorted_groups)
        elif choice == '3':
            self._annotate_by_custom_selection(token_groups)
        else:
            return
    
    def _analyze_token_patterns(self):
        """分析Token组合模式"""
        token_groups = {}
        
        for sample in self.samples_to_annotate:
            tokens = tuple(sample.get('tokens', []))
            token_key = str(list(tokens))
            
            if token_key not in token_groups:
                token_groups[token_key] = {
                    'tokens': tokens,
                    'samples': [],
                    'annotated_count': 0
                }
            
            token_groups[token_key]['samples'].append(sample)
            if sample.get('annotated', False):
                token_groups[token_key]['annotated_count'] += 1
        
        return token_groups
    
    def _annotate_by_frequency(self, sorted_groups):
        """按频率优先标注"""
        print("\n📌 按频率标注模式")
        print("=" * 70)
        
        try:
            target_coverage = float(safe_input("目标覆盖率 (0-100%, 推荐80): ", "80"))
            target_coverage = min(100, max(0, target_coverage))
        except:
            target_coverage = 80
        
        total_samples = len(self.samples_to_annotate)
        target_count = int(total_samples * target_coverage / 100)
        
        print(f"\n🎯 目标: 标注 {target_count}/{total_samples} 个样本 ({target_coverage}%覆盖)")
        
        covered_samples = 0
        groups_to_annotate = []
        
        for token_key, group_info in sorted_groups:
            if covered_samples >= target_count:
                break
            
            group_size = len(group_info['samples'])
            if group_info['annotated_count'] == 0:  # 未标注的组
                groups_to_annotate.append((token_key, group_info))
                covered_samples += group_size
        
        print(f"\n📋 需要标注 {len(groups_to_annotate)} 个Token组合")
        print(f"   将覆盖 {covered_samples} 个样本")
        
        confirm = safe_input("\n开始标注? (y/n): ", "n")
        if confirm.lower() != 'y':
            return
        
        # 依次标注每个组的代表样本
        for i, (token_key, group_info) in enumerate(groups_to_annotate, 1):
            print(f"\n{'='*70}")
            print(f"📦 Token组 [{i}/{len(groups_to_annotate)}]: {token_key}")
            print(f"   包含 {len(group_info['samples'])} 个样本")
            print(f"{'='*70}")
            
            # 选择代表样本（第一个）
            representative = group_info['samples'][0]
            
            # 标注代表样本
            success = self._annotate_single_sample_with_visual(representative)
            
            if success and representative.get('annotated', False):
                # 询问是否应用到整个组
                apply_choice = safe_input(
                    f"\n应用到该组的其他 {len(group_info['samples'])-1} 个样本? (y/n/s=跳过整组): ",
                    "y"
                )
                
                if apply_choice.lower() == 's':
                    print("⏭️ 跳过此Token组")
                    continue
                elif apply_choice.lower() == 'y':
                    # 复制标注到整个组
                    self._apply_annotation_to_group(representative, group_info['samples'][1:])
                    print(f"✅ 已将标注应用到 {len(group_info['samples'])} 个样本")
            
            # 询问是否继续
            if i < len(groups_to_annotate):
                continue_choice = safe_input("继续下一组? (y/n): ", "y")
                if continue_choice.lower() != 'y':
                    break
        
        print(f"\n🎉 批量标注完成!")
        self._show_annotation_summary()
    
    def _annotate_by_sampling(self, sorted_groups):
        """采样标注 - 每组标注1个代表"""
        print("\n🎲 采样标注模式")
        print("=" * 70)
        print("💡 每个Token组合只标注1个代表样本，快速建立基础标注库")
        
        try:
            max_groups = int(safe_input("最多标注多少个组? (推荐50-100): ", "50"))
        except:
            max_groups = 50
        
        # 过滤未标注的组
        unannotated_groups = [(k, v) for k, v in sorted_groups 
                              if v['annotated_count'] == 0]
        
        groups_to_annotate = unannotated_groups[:max_groups]
        
        print(f"\n📋 将标注 {len(groups_to_annotate)} 个Token组合")
        total_coverage = sum(len(g[1]['samples']) for g in groups_to_annotate)
        print(f"   潜在覆盖: {total_coverage} 个样本 "
              f"({total_coverage/len(self.samples_to_annotate)*100:.1f}%)")
        
        confirm = safe_input("\n开始采样标注? (y/n): ", "n")
        if confirm.lower() != 'y':
            return
        
        for i, (token_key, group_info) in enumerate(groups_to_annotate, 1):
            print(f"\n{'='*70}")
            print(f"📦 [{i}/{len(groups_to_annotate)}] {token_key}")
            print(f"   代表 {len(group_info['samples'])} 个样本")
            
            representative = group_info['samples'][0]
            success = self._annotate_single_sample_with_visual(representative)
            
            if i < len(groups_to_annotate):
                continue_choice = safe_input("继续? (y/n): ", "y")
                if continue_choice.lower() != 'y':
                    break
        
        print(f"\n🎉 采样标注完成!")
        self._show_annotation_summary()
    
    def _annotate_by_custom_selection(self, token_groups):
        """自定义选择Token组合标注"""
        print("\n✏️ 自定义标注模式")
        print("=" * 70)
        
        sorted_groups = sorted(token_groups.items(), 
                              key=lambda x: len(x[1]['samples']), 
                              reverse=True)
        
        print("\n可用Token组合:")
        for i, (token_key, group_info) in enumerate(sorted_groups[:30], 1):
            count = len(group_info['samples'])
            status = "✅" if group_info['annotated_count'] > 0 else "⭕"
            print(f"  {i:2d}. {status} {token_key} ({count}个样本)")
        
        try:
            indices = safe_input("\n输入要标注的组编号(逗号分隔，如1,3,5): ", "")
            selected_indices = [int(x.strip()) - 1 for x in indices.split(',') if x.strip()]
            
            selected_groups = [sorted_groups[i] for i in selected_indices 
                             if 0 <= i < len(sorted_groups)]
            
            if not selected_groups:
                print("❌ 无效选择")
                return
            
            print(f"\n已选择 {len(selected_groups)} 个组")
            
            for i, (token_key, group_info) in enumerate(selected_groups, 1):
                print(f"\n{'='*70}")
                print(f"📦 [{i}/{len(selected_groups)}] {token_key}")
                print(f"   包含 {len(group_info['samples'])} 个样本")
                
                representative = group_info['samples'][0]
                success = self._annotate_single_sample_with_visual(representative)
                
                if success and representative.get('annotated', False):
                    apply = safe_input(f"应用到该组全部{len(group_info['samples'])}个样本? (y/n): ", "y")
                    if apply.lower() == 'y':
                        self._apply_annotation_to_group(representative, group_info['samples'][1:])
                
        except Exception as e:
            print(f"❌ 选择失败: {e}")
    
    def _apply_annotation_to_group(self, source_sample, target_samples):
        """将标注应用到一组样本"""
        if not source_sample.get('annotated', False):
            return 0
        
        applied_count = 0
        for sample in target_samples:
            # 复制标注信息
            sample['annotations'] = source_sample.get('annotations', {}).copy()
            sample['overall_action'] = source_sample.get('overall_action', '')
            sample['annotation_time'] = datetime.now().isoformat()
            sample['annotated'] = True
            sample['batch_applied'] = True
            sample['batch_source_id'] = source_sample.get('id')
            
            # 保存
            self._auto_save_sample(sample)
            applied_count += 1
        
        return applied_count
    
    def _annotate_single_sample_with_visual(self, sample):
        """标注单个样本（带可视化）"""
        print(f"\n📁 文件: {sample.get('filename', '未知')}")
        print(f"🎯 Token: {sample.get('tokens', [])}")
        
        # 显示可视化
        if VISUALIZATION_AVAILABLE:
            try:
                success = show_sample_visualization(sample, sample)
                if success:
                    print("✅ 可视化窗口已打开")
                    input("👀 查看完毕后按回车继续...")
            except Exception as e:
                print(f"⚠️ 可视化失败: {e}")
        
        # 选择标注模式
        print("\n标注选项:")
        print("1. 详细分部位标注")
        print("2. 快速整体标注")
        print("3. 跳过")
        
        choice = safe_input("选择 (1-3): ", "1")
        
        if choice == '1':
            return self._detailed_annotation_cli(sample)
        elif choice == '2':
            return self._quick_overall_annotation_cli(sample)
        else:
            return False
    
    def _show_annotation_summary(self):
        """显示标注摘要"""
        annotated = sum(1 for s in self.samples_to_annotate if s.get('annotated', False))
        total = len(self.samples_to_annotate)
        
        print(f"\n📊 标注摘要:")
        print(f"   已标注: {annotated}/{total} ({annotated/total*100:.1f}%)")
        
        # Token覆盖率
        token_groups = self._analyze_token_patterns()
        annotated_groups = sum(1 for g in token_groups.values() if g['annotated_count'] > 0)
        print(f"   Token组覆盖: {annotated_groups}/{len(token_groups)} "
              f"({annotated_groups/len(token_groups)*100:.1f}%)")
    
    def token_analysis_cli(self):
        """Token分析与采样策略"""
        if not self.samples_to_annotate:
            print("❌ 暂无样本数据")
            return
        
        print("\n📊 Token统计分析")
        print("=" * 70)
        
        # Token组合分析
        token_groups = self._analyze_token_patterns()
        
        print(f"📈 数据概况:")
        print(f"   总样本数: {len(self.samples_to_annotate)}")
        print(f"   Token组合数: {len(token_groups)}")
        print(f"   平均每组: {len(self.samples_to_annotate)/len(token_groups):.1f} 个样本")
        
        # 分布统计
        group_sizes = [len(g['samples']) for g in token_groups.values()]
        import statistics
        
        print(f"\n📊 分布统计:")
        print(f"   最大组: {max(group_sizes)} 个样本")
        print(f"   最小组: {min(group_sizes)} 个样本")
        print(f"   中位数: {statistics.median(group_sizes):.0f} 个样本")
        print(f"   平均值: {statistics.mean(group_sizes):.1f} 个样本")
        
        # Top组合
        sorted_groups = sorted(token_groups.items(), 
                              key=lambda x: len(x[1]['samples']), 
                              reverse=True)
        
        print(f"\n🔝 Top 10 Token组合:")
        for i, (token_key, group_info) in enumerate(sorted_groups[:10], 1):
            count = len(group_info['samples'])
            percentage = count / len(self.samples_to_annotate) * 100
            status = f"✅{group_info['annotated_count']}" if group_info['annotated_count'] > 0 else "⭕"
            print(f"  {i:2d}. {status} {token_key}")
            print(f"      → {count} 样本 ({percentage:.2f}%)")
        
        # 采样建议
        print(f"\n💡 标注策略建议:")
        
        # 计算不同覆盖率需要标注的组数
        cumulative = 0
        for coverage_target in [50, 80, 90, 95]:
            target_count = len(self.samples_to_annotate) * coverage_target / 100
            groups_needed = 0
            cumulative_temp = 0
            
            for _, group_info in sorted_groups:
                if cumulative_temp >= target_count:
                    break
                cumulative_temp += len(group_info['samples'])
                groups_needed += 1
            
            print(f"   {coverage_target}% 覆盖率 → 需标注 {groups_needed} 个Token组")
        
        print(f"\n推荐策略:")
        print(f"   🔸 快速建立基础: 采样标注模式 (标注50-100个代表)")
        print(f"   🔸 高覆盖率: 按频率标注 (目标80%覆盖)")
        print(f"   🔸 针对性标注: 自定义选择特定Token组合")
    
    def old_batch_annotate_cli(self):
        """命令行批量标注"""
        if not self.annotation_data:
            print("❌ 暂无已标注样本，无法进行批量标注")
            return
            
        print("\n🔄 批量标注相似样本")
        
        # 检查是否有标注数据
        if len(self.annotation_data) == 0:
            print("❌ 当前没有已标注样本")
            return
        
        # 显示已标注的样本
        print(f"已标注样本 (共{len(self.annotation_data)}个):")
        displayed_count = 0
        for sample_id, annotation in self.annotation_data.items():
            if displayed_count >= 15:  # 最多显示15个
                print(f"  ... 还有{len(self.annotation_data) - displayed_count}个样本")
                break
                
            tokens_str = str(annotation.get('tokens', 'N/A'))
            global_action = annotation.get('global_action', '未描述')
            method = annotation.get('annotation_method', 'manual')
            
            print(f"  样本{sample_id}: {tokens_str} -> {global_action} ({method})")
            displayed_count += 1
            
        try:
            reference_id = int(input("选择参考样本ID: ").strip())
            if reference_id not in self.annotation_data:
                print("❌ 样本ID不存在")
                return
                
            reference_annotation = self.annotation_data[reference_id]
            reference_tokens = tuple(reference_annotation['tokens'])
            
            # 找到相似样本
            similar_samples = []
            for sample in self.samples_to_annotate:
                if tuple(sample['tokens']) == reference_tokens and not sample['annotated']:
                    similar_samples.append(sample)
                    
            if not similar_samples:
                print("❌ 未找到相似的未标注样本")
                return
                
            print(f"找到 {len(similar_samples)} 个相似样本")
            confirm = input("是否批量应用标注? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                for sample in similar_samples:
                    new_annotation = reference_annotation.copy()
                    new_annotation['sample_id'] = sample['id']
                    new_annotation['timestamp'] = datetime.now().isoformat()
                    new_annotation['batch_source'] = reference_id
                    
                    self.annotation_data[sample['id']] = new_annotation
                    sample['annotated'] = True
                    
                print(f"✅ 批量标注完成，共标注 {len(similar_samples)} 个样本")
            
        except ValueError:
            print("❌ 请输入有效的样本ID")
            
    def show_progress_cli(self):
        """显示标注进度"""
        if not self.samples_to_annotate:
            print("❌ 暂无样本数据")
            return
            
        print("\n📊 标注进度统计")
        print("=" * 60)
        
        # 基础统计
        total_samples = len(self.samples_to_annotate)
        annotated_count = len(self.annotation_data)
        progress_percentage = (annotated_count / total_samples * 100) if total_samples > 0 else 0
        
        print(f"总样本数: {total_samples}")
        print(f"已标注: {annotated_count}")
        print(f"未标注: {total_samples - annotated_count}")
        print(f"完成度: {progress_percentage:.1f}%")
        
        if annotated_count == 0:
            print("暂无已标注样本")
            return
        
        # 统计各部位标注情况
        part_stats = {}
        action_stats = {}
        
        for annotation in self.annotation_data.values():
            for part, action in annotation['part_annotations'].items():
                if part not in part_stats:
                    part_stats[part] = {}
                part_stats[part][action] = part_stats[part].get(action, 0) + 1
                
                if action not in action_stats:
                    action_stats[action] = 0
                action_stats[action] += 1
                
        print(f"\n各部位标注分布:")
        for part, display_name in zip(self.part_names, self.part_display_names):
            if part in part_stats:
                print(f"  {display_name}: {len(part_stats[part])} 种动作")
                # 显示前3个最常见动作
                top_actions = sorted(part_stats[part].items(), key=lambda x: x[1], reverse=True)[:3]
                for action, count in top_actions:
                    print(f"    {action}: {count}次")
            else:
                print(f"  {display_name}: 暂无标注")
                
    def export_annotations(self):
        """导出标注结果"""
        if not self.annotation_data:
            print("❌ 暂无标注数据，无法导出")
            return
            
        print(f"\n📤 导出标注结果")
        print(f"准备导出 {len(self.annotation_data)} 个已标注样本")
        
        # 构建映射数据
        mapping_data = {
            'part_mappings': {},
            'global_mappings': {},
            'statistics': {
                'total_samples': len(self.samples_to_annotate),
                'annotated_samples': len(self.annotation_data),
                'annotation_date': datetime.now().isoformat()
            },
            'raw_annotations': self.annotation_data
        }
        
        # 构建部位映射
        for part in self.part_names:
            mapping_data['part_mappings'][part] = {}
            
        for annotation in self.annotation_data.values():
            tokens = annotation['tokens']
            part_annotations = annotation['part_annotations']
            
            for i, part in enumerate(self.part_names):
                if i < len(tokens) and part in part_annotations:
                    token_id = str(tokens[i])
                    action = part_annotations[part]
                    
                    if token_id not in mapping_data['part_mappings'][part]:
                        mapping_data['part_mappings'][part][token_id] = {
                            'semantic': action,
                            'frequency': 0,
                            'confidence': 0.8,  # 默认置信度
                            'samples': []
                        }
                    
                    mapping_data['part_mappings'][part][token_id]['frequency'] += 1
                    mapping_data['part_mappings'][part][token_id]['samples'].append(annotation['sample_id'])
                    
        # 构建全局映射
        for annotation in self.annotation_data.values():
            if annotation.get('global_action'):
                token_combo = tuple(annotation['tokens'])
                global_action = annotation['global_action']
                
                combo_key = str(token_combo)
                if combo_key not in mapping_data['global_mappings']:
                    mapping_data['global_mappings'][combo_key] = {
                        'action': global_action,
                        'frequency': 0,
                        'confidence': 0.85,
                        'category': 'annotated',
                        'samples': []
                    }
                    
                mapping_data['global_mappings'][combo_key]['frequency'] += 1
                mapping_data['global_mappings'][combo_key]['samples'].append(annotation['sample_id'])
                
        # 保存文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"codebook_annotations_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 标注结果已导出到: {filename}")
        
        # 同时保存为标准映射格式
        mapping_filename = "codebook_action_mappings.json"
        with open(mapping_filename, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 映射表已保存到: {mapping_filename}")
        
        # 显示导出统计
        print(f"\n📊 导出统计:")
        print(f"  已标注样本: {len(self.annotation_data)}")
        print(f"  部位映射数: {sum(len(mappings) for mappings in mapping_data['part_mappings'].values())}")
        print(f"  全局映射数: {len(mapping_data['global_mappings'])}")
        
        # 显示各部位的token数量
        for part, mappings in mapping_data['part_mappings'].items():
            if mappings:
                print(f"    {part}: {len(mappings)} 个token")
        
        print("✅ 标准格式导出完成！")
        
        # ============= LLM 友好格式导出 =============
        if LLM_EXPORTER_AVAILABLE:
            print(f"\n🤖 检测到 LLM 导出器")
            export_llm = safe_input("是否同时导出 LLM 友好格式? (y/n, 默认n): ", "n").lower()
            
            if export_llm == 'y':
                print(f"\n🚀 开始导出 LLM 友好格式...")
                print("   包括: Few-shot学习, 指令微调, 对话格式, RAG知识库")
                
                try:
                    # 准备样本数据
                    samples_data = []
                    for sample in self.samples_to_annotate:
                        if sample['id'] in self.annotation_data:
                            sample_info = {
                                'id': sample['id'],
                                'split': sample.get('split', 'unknown'),
                                'file_path': sample.get('file_path', sample.get('filename', '')),
                                'vq_loss': sample.get('vq_loss', 0.0),
                                'ground_truth': sample.get('ground_truth', sample.get('ground_truth_action', ''))
                            }
                            samples_data.append(sample_info)
                    
                    # 获取输出目录
                    output_dir = safe_input("输出目录 (默认: llm_annotations): ", "llm_annotations")
                    dataset_name = safe_input("数据集名称 (默认: MARS): ", "MARS")
                    
                    # 执行导出
                    exporter = LLMAnnotationExporter()
                    exporter.export_enhanced_annotations(
                        annotation_data=self.annotation_data,
                        samples_data=samples_data,
                        output_dir=output_dir,
                        dataset_name=dataset_name
                    )
                    
                    print(f"\n✅ LLM 友好格式导出完成！")
                    print(f"📂 文件位置: {output_dir}/")
                    print(f"💡 可用于: Few-shot学习, 指令微调, 对话训练, RAG检索")
                    
                except Exception as e:
                    print(f"\n❌ LLM 格式导出失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⏭️  跳过 LLM 格式导出")
        
    def import_annotations(self):
        """导入现有标注"""
        filename = input("请输入标注文件路径 (默认: codebook_action_mappings.json): ").strip()
        if not filename:
            filename = "codebook_action_mappings.json"
            
        if not os.path.exists(filename):
            print(f"❌ 文件不存在: {filename}")
            return
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'raw_annotations' in data:
                imported_count = len(data['raw_annotations'])
                self.annotation_data.update(data['raw_annotations'])
                
                # 更新样本标注状态
                for sample in self.samples_to_annotate:
                    if sample['id'] in self.annotation_data:
                        sample['annotated'] = True
                        
                print(f"✅ 导入 {imported_count} 个标注")
            else:
                print("⚠️ 文件格式不包含原始标注数据")
                
        except Exception as e:
            print(f"❌ 导入失败: {e}")

def main():
    """主函数"""
    import sys
    
    # 检查命令行参数
    batch_mode = '--batch' in sys.argv or '--auto' in sys.argv
    demo_mode = '--demo' in sys.argv
    
    print("🏷️ 码本动作标注工具")
    print("=" * 50)
    
    if batch_mode:
        print("🤖 批处理模式启动")
        set_batch_mode(True)
        
    tool = SkeletonAnnotationTool()
    
    if demo_mode:
        print("🎬 演示模式：自动加载数据并进行快速标注")
        # 自动演示流程
        tool.load_real_data("ntu")
        if tool.samples_to_annotate:
            # 对前3个样本进行快速标注
            for i, sample in enumerate(tool.samples_to_annotate[:3]):
                print(f"\n🎭 自动标注样本 {i+1}: {sample.get('filename', 'unknown')}")
                tool._quick_overall_annotation_cli(sample)
                sample['annotated'] = True
            print("🎉 演示完成")
        return
    
    if tool.use_gui and GUI_AVAILABLE and not batch_mode:
        print("🖥️ 启动图形界面...")
        # 这里可以启动GUI版本
        print("⚠️ GUI界面开发中，使用命令行界面")
        tool.run_cli_annotation()
    else:
        print("💻 使用命令行界面...")
        tool.run_cli_annotation()

if __name__ == "__main__":
    main()
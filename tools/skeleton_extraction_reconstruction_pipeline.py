#!/usr/bin/env python3
"""
骨架提取 + GCN重构 + 文本映射完整流程
结合skeleton_extractor.py的雷达骨架提取和GCNSkeletonTokenizer.py的骨架重构
实现雷达信号 → 骨架提取 → 码本编码 → 重构可视化 → 文本描述的完整管线
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import cv2
from PIL import Image
import io
import gc
import json
from matplotlib.animation import FuncAnimation, PillowWriter

# 导入文本映射模块
try:
    from tools.codebook_text_mapper import CodebookTextMapper
    TEXT_MAPPING_AVAILABLE = True
except ImportError:
    TEXT_MAPPING_AVAILABLE = False
    print("⚠️ 文本映射模块不可用")

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')

# 添加models路径以导入关节点映射器
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.skeleton_joint_mapper import SkeletonJointMapper, EnhancedSkeletonMapper

# 配置中文字体支持 - 解决中文显示问题
import matplotlib.font_manager as fm

# 尝试设置支持中文的字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
font_found = False
for font_name in chinese_fonts:
    try:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font_name]
            font_found = True
            break
    except:
        continue

if not font_found:
    # 如果没有找到中文字体，使用默认字体但避免中文显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("⚠️ 未找到中文字体，将使用英文显示")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入自定义模块
try:
    from models.skeleton_extractor import MARSTransformerModel
    from models.GCNSkeletonTokenizer import GCNSkeletonTokenizer
    from utils.config import cfg_from_yaml_file
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class SkeletonExtractionReconstructionPipeline:
    """骨架提取和重构完整流程 - 支持多尺度特征融合"""
    
    def __init__(self, extractor_model_path, gcn_model_path, gcn_config_path, use_enhanced_mapping=True):
        print("=" * 80)
        print("初始化骨架提取和重构流程")
        print("=" * 80)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  设备: {self.device}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("\n" + "-" * 80)
        print("步骤 1/3: 初始化关节点映射器")
        print("-" * 80)
        
        # 初始化关节点映射器 (MARS 19关节 -> NTU 25关节)
        if use_enhanced_mapping:
            self.joint_mapper = EnhancedSkeletonMapper().to(self.device)
            print("🎯 使用增强关节点映射器 (MARS 19关节 -> NTU 25关节)")
        else:
            self.joint_mapper = SkeletonJointMapper().to(self.device)
            print("🎯 使用基础关节点映射器 (MARS 19关节 -> NTU 25关节)")
        
        print("\n" + "-" * 80)
        print("步骤 2/3: 加载骨架提取器")
        print("-" * 80)
        
        # 加载骨架提取器
        self.skeleton_extractor = self._load_skeleton_extractor(extractor_model_path)
        
        print("\n" + "-" * 80)
        print("步骤 3/3: 加载GCN重构器")
        print("-" * 80)
        
        # 加载GCN重构器
        self.gcn_reconstructor = self._load_gcn_reconstructor(gcn_model_path, gcn_config_path)
        
        print("\n" + "-" * 80)
        print("初始化文本映射器")
        print("-" * 80)
        
        # 初始化文本映射器
        if TEXT_MAPPING_AVAILABLE:
            self.text_mapper = CodebookTextMapper()
            print("🔤 文本映射器初始化完成")
        else:
            self.text_mapper = None
            print("⚠️ 文本映射器不可用")
        
        # NTU RGB+D 25关节点连接关系
        self.skeleton_edges = [
            (3, 2), (2, 20), (20, 1), (1, 0),  # 头部和脊柱
            (20, 4), (4, 5), (5, 6), (6, 22), (6, 7), (7, 21),  # 左臂
            (20, 8), (8, 9), (9, 10), (10, 24), (10, 11), (11, 23),  # 右臂
            (0, 12), (12, 13), (13, 14), (14, 15),  # 左腿
            (0, 16), (16, 17), (17, 18), (18, 19)   # 右腿
        ]
        
        print("\n" + "=" * 80)
        print("✅ 流水线初始化完成!")
        print("=" * 80)
    
    def _load_skeleton_extractor(self, model_path):
        """加载MARS骨架提取器 - 支持多尺度特征融合"""
        print(f"Loading skeleton extractor: {model_path}")
        
        # 尝试加载权重以检测模型类型
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 检测是否为多尺度模型（通过第一层Linear的输入维度判断）
        # 查找feature_projection的第一层Linear权重
        first_linear_key = 'regression_head.feature_projection.0.weight'
        
        if first_linear_key in state_dict:
            input_dim = state_dict[first_linear_key].shape[1]  # (out_features, in_features)
            is_multi_scale = (input_dim == 448)
            
            if is_multi_scale:
                print("🔍 检测到多尺度模型 (448维输入)")
                model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=True)
            else:
                print("🔍 检测到单尺度模型 (256维输入)")
                model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=False)
        else:
            # 如果找不到关键层，默认尝试多尺度（新版本）
            print("⚠️ 无法检测模型类型，默认使用多尺度模型")
            model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=True)
        
        # 加载权重
        try:
            model.load_state_dict(state_dict)
            print("✅ 权重加载成功！")
        except RuntimeError as e:
            print(f"❌ 权重加载失败: {e}")
            print("⚠️ 可能是模型版本不匹配，请检查权重文件")
            raise
        
        model.to(self.device)
        model.eval()
        
        # 输出模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数: {total_params:,}")
        print(f"📊 Backbone输出维度: {model.backbone.output_dim}")
        print("✅ Skeleton extractor loaded successfully!")
        
        return model
    
    def _load_gcn_reconstructor(self, model_path, config_path):
        """加载GCN骨架重构器"""
        print(f"Loading GCN reconstructor: {model_path}")
        
        try:
            # 加载配置
            config = cfg_from_yaml_file(config_path)
            print(f"📋 配置文件: {config_path}")
            
            # 创建模型
            model = GCNSkeletonTokenizer(config.model)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'base_model' in checkpoint:
                state_dict = checkpoint['base_model']
                print("🔍 使用checkpoint中的base_model")
            else:
                state_dict = checkpoint
                print("🔍 直接加载state_dict")
                
            # 处理分布式训练权重
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            model.to(self.device)
            model.eval()
            
            # 输出模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 模型参数: {total_params:,}")
            
            # 输出码本信息（如果有）
            if hasattr(model, 'codebooks') and hasattr(model.codebooks, 'num_groups'):
                print(f"📊 码本配置: {model.codebooks.num_groups}组 × {model.codebooks.codebook_size}个码字")
            
            print("✅ GCN reconstructor loaded successfully!")
            return model
            
        except Exception as e:
            print(f"❌ 加载GCN重构器失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回None而不是崩溃
            return None
    
    def extract_skeleton_from_radar(self, radar_data):
        """从雷达特征图提取骨架"""
        with torch.no_grad():
            # 转换数据格式：(B, H, W, C) -> (B, C, H, W)
            if len(radar_data.shape) == 4:
                radar_tensor = torch.from_numpy(radar_data.transpose(0, 3, 1, 2)).float().to(self.device)
            elif len(radar_data.shape) == 3:
                radar_tensor = torch.from_numpy(radar_data.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
            else:
                raise ValueError(f"Unexpected radar data shape: {radar_data.shape}")
            
            # MARS骨架提取器推理
            mars_output = self.skeleton_extractor(radar_tensor)
            
            # 检查输出格式
            if len(mars_output.shape) == 2 and mars_output.shape[1] == 57:
                # MARS格式：(B, 57) = (x1...x19, y1...y19, z1...z19)
                # 参照vis_skeleton_extractor.py的处理方式
                batch_size = mars_output.shape[0]
                
                # 重组为(B, 19, 3)格式: 从扁平化的57维重组为19个关节点
                x_coords = mars_output[:, 0:19]    # x坐标: 0-18
                y_coords = mars_output[:, 19:38]   # y坐标: 19-37  
                z_coords = mars_output[:, 38:57]   # z坐标: 38-56
                mars_skeleton = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # (B, 19, 3)
            else:
                # 假设已经是(B, 19, 3)格式
                mars_skeleton = mars_output
            
            # 使用映射器转换为NTU 25关节点
            # 注意：映射器需要原始57维输出，会内部处理坐标重组
            ntu_skeleton = self.joint_mapper(mars_output)  # 传入原始57维输出
            
            return ntu_skeleton
    
    def reconstruct_skeleton_with_gcn(self, skeleton_data):
        """使用GCN码本重构骨架"""
        if self.gcn_reconstructor is None:
            print("❌ GCN重构器未加载")
            return None
            
        with torch.no_grad():
            # 标准化骨架数据（与GCN训练时一致）
            normalized_skeleton = self._normalize_skeleton(skeleton_data)
            
            # 转换为张量
            if isinstance(normalized_skeleton, np.ndarray):
                skeleton_tensor = torch.from_numpy(normalized_skeleton.astype(np.float32))
            else:
                skeleton_tensor = normalized_skeleton
            
            skeleton_tensor = skeleton_tensor.to(self.device)
            
            # GCN前向传播
            try:
                output = self.gcn_reconstructor(skeleton_tensor, return_recon=True)
                
                # 提取结果
                reconstructed_xzy = output['reconstructed'].cpu().numpy()
                token_sequence = output['token_sequence'].cpu().numpy()
                vq_loss = output['vq_loss'].item()
                
                # 参照gcn_skeleton_gif_visualizer.py的处理方式：
                # 对重建的骨架进行坐标转换: (x,z,y) -> (x,y,z) 以匹配可视化
                reconstructed = reconstructed_xzy[:, :, [0, 2, 1]]  # [x,z,y] -> [x,y,z]
                
                # 将归一化结果也转换回(x,y,z)格式以保持一致性
                normalized_xyz = normalized_skeleton[:, :, [0, 2, 1]]  # [x,z,y] -> [x,y,z]
                
                return {
                    'original': skeleton_data,
                    'normalized': normalized_xyz,
                    'reconstructed': reconstructed,
                    'token_sequence': token_sequence,
                    'vq_loss': vq_loss,
                    'group_results': output.get('group_results', {})
                }
                
            except Exception as e:
                print(f"❌ GCN重构过程中出错: {e}")
                return None
    
    def _normalize_skeleton(self, skeleton):
        """标准化骨架数据（与GCN训练时保持一致）
        
        关键修正：pipeline中的skeleton数据来自MARS→NTU映射器，
        已经是NTU标准的(x,y,z)格式，但GCN模型期望(x,z,y)格式输入
        """
        if isinstance(skeleton, torch.Tensor):
            skeleton = skeleton.cpu().numpy()
        
        normalized_skeletons = []
        for i in range(skeleton.shape[0]):
            single_skeleton = skeleton[i]
            
            # MARS映射器输出的是标准NTU格式(x,y,z)
            # 需要转换为GCN训练时使用的(x,z,y)格式
            single_skeleton_xzy = single_skeleton[:, [0, 2, 1]]  # [x,y,z] -> [x,z,y]
            
            # 禁用骨架对齐，因为它可能导致倒立问题
            aligned = single_skeleton_xzy  # 直接使用原始方向
            # aligned = self._align_skeleton_orientation(single_skeleton_xzy)
            
            # 使用与训练时一致的标准化方法
            normalized = self._normalize_single_skeleton(aligned)
                
            normalized_skeletons.append(normalized)
        
        return np.array(normalized_skeletons)
    
    def _align_skeleton_orientation(self, skeleton):
        """对齐骨架方向，减少旋转导致的重建错误
        
        完全复制gcn_skeleton_visualizer.py的逻辑以确保一致性
        """
        # 计算主要身体轴向（从骨盆到头部）
        # NTU RGB+D关节点索引：0=骨盆中心, 3=头顶
        if len(skeleton) >= 4:
            pelvis = skeleton[0]  # 骨盆中心
            head = skeleton[3]   # 头顶

            # 计算身体主轴
            body_axis = head - pelvis
            body_axis_norm = np.linalg.norm(body_axis)

            if body_axis_norm > 1e-6:
                # 在(x,z,y)格式中，Z轴(索引1)是垂直方向
                # 将身体主轴对齐到Z轴正方向(向上)
                target_axis = np.array([0, 1, 0])  # 在(x,z,y)格式中对应Z轴(垂直向上)
                body_axis_normalized = body_axis / body_axis_norm

                # 计算旋转角度
                cos_angle = np.dot(body_axis_normalized, target_axis)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)

                # 如果身体轴向与Z轴相反（倒立），进行180度旋转
                if cos_angle < -0.5:  # 角度大于120度，认为是倒立
                    # 绕X轴旋转180度翻正骨架
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                    skeleton = np.dot(skeleton, rotation_matrix.T)

        return skeleton
    
    def _normalize_single_skeleton(self, skeleton):
        """标准化单个骨架"""
        # 计算质心
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
    
    def _denormalize_skeleton(self, normalized_skeleton, reference_skeleton):
        """将标准化的骨架反变换到原始坐标空间
        
        Args:
            normalized_skeleton: 标准化的骨架数据
            reference_skeleton: 参考的原始骨架（用于获取标准化参数）
        
        Returns:
            反标准化后的骨架数据
        """
        # 确保数据是numpy数组
        if isinstance(normalized_skeleton, torch.Tensor):
            normalized_skeleton = normalized_skeleton.cpu().numpy()
        if isinstance(reference_skeleton, torch.Tensor):
            reference_skeleton = reference_skeleton.cpu().numpy()
        
        # 计算参考骨架的标准化参数
        reference_centroid = np.mean(reference_skeleton, axis=0)
        reference_centered = reference_skeleton - reference_centroid
        reference_distances = np.sqrt(np.sum(reference_centered**2, axis=1))
        reference_max_distance = np.max(reference_distances)
        
        # 反变换：先乘以最大距离，再加上质心
        if reference_max_distance > 0:
            denormalized = normalized_skeleton * reference_max_distance + reference_centroid
        else:
            denormalized = normalized_skeleton + reference_centroid
            
        return denormalized
    
    def process_complete_pipeline(self, radar_feature_map):
        """完整的处理流程：雷达 → 骨架提取 → GCN重构"""
        print("🔄 执行完整处理流程...")
        
        # 步骤1：从雷达数据提取骨架
        print("  1️⃣ 从雷达特征图提取骨架...")
        extracted_skeleton = self.extract_skeleton_from_radar(radar_feature_map)
        print(f"     ✅ 提取骨架形状: {extracted_skeleton.shape}")
        
        # 步骤2：使用GCN重构骨架
        print("  2️⃣ 使用GCN码本重构骨架...")
        reconstruction_result = self.reconstruct_skeleton_with_gcn(extracted_skeleton)
        print(f"     ✅ 重构完成，VQ损失: {reconstruction_result['vq_loss']:.6f}")
        
        # 步骤3：生成文本描述
        text_description = None
        if self.text_mapper is not None and reconstruction_result is not None:
            print("  3️⃣ 生成动作文本描述...")
            try:
                token_sequence = reconstruction_result['token_sequence']
                if token_sequence.ndim > 1:
                    token_sequence = token_sequence.flatten()
                
                # 确保token序列长度为5
                if len(token_sequence) >= 5:
                    token_list = token_sequence[:5].astype(int).tolist()
                    text_description = self.text_mapper.map_tokens_to_text(token_list)
                    print(f"     ✅ 文本描述: {text_description['natural_language']}")
                else:
                    print(f"     ⚠️ Token序列长度不足: {len(token_sequence)}")
                    
            except Exception as e:
                print(f"     ⚠️ 文本描述生成失败: {e}")
        
        return {
            'radar_input': radar_feature_map,
            'extracted_skeleton': extracted_skeleton,
            'reconstruction_result': reconstruction_result,
            'text_description': text_description
        }
    
    def visualize_results(self, pipeline_results_list, save_path):
        """可视化多个样本的流程结果"""
        print(f"🎨 生成多样本可视化结果...")
        
        num_samples = len(pipeline_results_list)
        # 创建2x2网格，每个样本显示4个子图：3个骨架+1个误差
        fig = plt.figure(figsize=(20, 16))
        
        for sample_idx, pipeline_result in enumerate(pipeline_results_list):
            extracted_skeleton = pipeline_result['extracted_skeleton'][0]  # 取第一个样本
            reconstruction_result = pipeline_result['reconstruction_result']
            normalized_skeleton = reconstruction_result['normalized'][0]  # MARS标签骨架（归一化后）
            reconstructed_skeleton = reconstruction_result['reconstructed'][0]
            
            # 为每个样本创建4个子图
            base_idx = sample_idx * 4
            
            # 1. MARS标签骨架（归一化）
            ax1 = fig.add_subplot(num_samples, 4, base_idx + 1, projection='3d')
            self._plot_skeleton_3d(ax1, normalized_skeleton, f'Sample {sample_idx+1}: MARS Label Skeleton', 'blue')
            
            # 2. 提取骨架（原始）
            ax2 = fig.add_subplot(num_samples, 4, base_idx + 2, projection='3d')
            self._plot_skeleton_3d(ax2, extracted_skeleton, f'Sample {sample_idx+1}: Extracted Skeleton', 'green')
            
            # 3. 重构骨架
            ax3 = fig.add_subplot(num_samples, 4, base_idx + 3, projection='3d')
            self._plot_skeleton_3d(ax3, reconstructed_skeleton, f'Sample {sample_idx+1}: Reconstructed Skeleton', 'red')
            
            # 4. 关节重建损失
            ax4 = fig.add_subplot(num_samples, 4, base_idx + 4)
            errors = np.sqrt(np.sum((normalized_skeleton - reconstructed_skeleton)**2, axis=1))
            bars = ax4.bar(range(len(errors)), errors)
            ax4.set_title(f'Sample {sample_idx+1}: Joint Reconstruction Errors', fontsize=10, fontweight='bold')
            ax4.set_xlabel('Joint Index')
            ax4.set_ylabel('Error (L2)')
            
            # 为误差高的关节点标记颜色
            max_error = np.max(errors)
            for i, bar in enumerate(bars):
                if errors[i] > max_error * 0.7:
                    bar.set_color('red')
                elif errors[i] > max_error * 0.4:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
        
        # 计算总体统计信息
        all_recon_mse_errors = []
        all_real_world_mse_errors = []  # 添加实际坐标空间的MSE
        all_vq_losses = []
        all_max_errors = []
        all_mean_errors = []
        
        for pipeline_result in pipeline_results_list:
            reconstruction_result = pipeline_result['reconstruction_result']
            normalized_skeleton = reconstruction_result['normalized'][0]
            reconstructed_skeleton = reconstruction_result['reconstructed'][0]
            original_skeleton = reconstruction_result['original'][0]  # 原始提取的骨架
            
            # 确保所有数据都是numpy数组
            if isinstance(normalized_skeleton, torch.Tensor):
                normalized_skeleton = normalized_skeleton.cpu().numpy()
            if isinstance(reconstructed_skeleton, torch.Tensor):
                reconstructed_skeleton = reconstructed_skeleton.cpu().numpy()
            if isinstance(original_skeleton, torch.Tensor):
                original_skeleton = original_skeleton.cpu().numpy()
            
            # 标准化空间中的MSE（训练时的评估指标）
            recon_mse_error = np.mean((normalized_skeleton - reconstructed_skeleton)**2)
            
            # 实际坐标空间的MSE（更直观的误差表示）
            # 将重构骨架反变换到原始坐标空间进行比较
            reconstructed_original = self._denormalize_skeleton(reconstructed_skeleton, original_skeleton)
            real_world_mse_error = np.mean((original_skeleton - reconstructed_original)**2)
            
            vq_loss = reconstruction_result['vq_loss']
            errors = np.sqrt(np.sum((normalized_skeleton - reconstructed_skeleton)**2, axis=1))
            
            all_recon_mse_errors.append(recon_mse_error)
            all_real_world_mse_errors.append(real_world_mse_error)
            all_vq_losses.append(vq_loss)
            all_max_errors.append(np.max(errors))
            all_mean_errors.append(np.mean(errors))
        
        # 添加整体标题 - 显示两种MSE
        avg_recon_mse = np.mean(all_recon_mse_errors)
        avg_real_world_mse = np.mean(all_real_world_mse_errors)
        avg_vq = np.mean(all_vq_losses)
        plt.suptitle(f'Multi-Sample Skeleton Analysis ({num_samples} samples)\\n'
                    f'Normalized MSE: {avg_recon_mse:.6f} | Real-World MSE: {avg_real_world_mse:.4f} | VQ Loss: {avg_vq:.6f}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Multi-sample visualization saved: {save_path}")
        
        return {
            'recon_mse_error': float(np.mean(all_recon_mse_errors)),
            'real_world_mse_error': float(np.mean(all_real_world_mse_errors)),
            'vq_loss': float(np.mean(all_vq_losses)),
            'max_joint_error': float(np.mean(all_max_errors)),
            'mean_joint_error': float(np.mean(all_mean_errors)),
            'sample_details': [
                {
                    'recon_mse_error': float(all_recon_mse_errors[i]),
                    'real_world_mse_error': float(all_real_world_mse_errors[i]),
                    'vq_loss': float(all_vq_losses[i]),
                    'max_joint_error': float(all_max_errors[i]),
                    'mean_joint_error': float(all_mean_errors[i])
                } for i in range(num_samples)
            ]
        }
    
    def _plot_skeleton_3d(self, ax, skeleton, title, color):
        """绘制3D骨架，特别处理MARS映射的6个额外关节点"""
        # 确保skeleton是numpy数组
        if isinstance(skeleton, torch.Tensor):
            skeleton = skeleton.cpu().numpy()
        
        # 修正可视化方向：反转Z轴以改善视角
        skeleton = skeleton.copy()  # 避免修改原始数据
        skeleton[:, 2] = -skeleton[:, 2]  # 反转Z轴改善视角
        
        # 绘制骨骼连接
        for edge in self.skeleton_edges:
            if edge[0] < len(skeleton) and edge[1] < len(skeleton):
                start = skeleton[edge[0]]
                end = skeleton[edge[1]]
                if not (np.all(start == 0) or np.all(end == 0)):
                    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                             color=color, alpha=0.8, linewidth=2.0)
        
        # 绘制所有25个关节点（统一样式）
        for i in range(len(skeleton)):
            joint = skeleton[i]
            if not np.all(joint == 0):
                ax.scatter(joint[0], joint[1], joint[2],
                          c=color, s=25, alpha=0.9, edgecolors='white', linewidth=0.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 设置等比例坐标范围
        valid_joints = skeleton[~np.all(skeleton == 0, axis=1)]
        if len(valid_joints) > 0:
            # 计算骨架的实际范围
            min_coords = np.min(valid_joints, axis=0)
            max_coords = np.max(valid_joints, axis=0)
            center = np.mean(valid_joints, axis=0)
            
            # 计算最大范围以确保等比例
            ranges = max_coords - min_coords
            max_range = max(np.max(ranges) / 2, 0.3)  # 最小范围0.3
            
            # 设置等比例坐标轴
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([center[1] - max_range, center[1] + max_range])
            ax.set_zlim([center[2] - max_range, center[2] + max_range])
            
            # 强制等比例 - 参照两个参考文件的设置方式
            ax.set_box_aspect([1,1,1])
        
        # 参考gcn_skeleton_gif_visualizer.py的视角设置
        ax.view_init(elev=15, azim=45)
    
    def generate_individual_sample_visualizations(self, pipeline_results, output_dir):
        """为每个样本生成单独的可视化图片"""
        print(f"🖼️ 生成单独样本可视化...")
        
        individual_metrics = []
        
        for i, result in enumerate(pipeline_results):
            sample_idx = i + 1
            
            # 提取数据 - 确保所有数据都转换为numpy
            radar_input = result['radar_input']
            extracted_skeleton = result['extracted_skeleton'][0].cpu().numpy() if isinstance(result['extracted_skeleton'], torch.Tensor) else result['extracted_skeleton'][0]
            reconstructed_skeleton = result['reconstruction_result']['reconstructed'][0]
            normalized_skeleton = result['reconstruction_result']['normalized'][0]
            original_skeleton = result['reconstruction_result']['original'][0]
            
            # 确保所有骨架数据都是numpy数组
            if isinstance(reconstructed_skeleton, torch.Tensor):
                reconstructed_skeleton = reconstructed_skeleton.cpu().numpy()
            if isinstance(normalized_skeleton, torch.Tensor):
                normalized_skeleton = normalized_skeleton.cpu().numpy()
            if isinstance(original_skeleton, torch.Tensor):
                original_skeleton = original_skeleton.cpu().numpy()
            
            # 计算标准化空间的真正重构误差（与训练时一致）
            normalized_recon_mse = np.mean((normalized_skeleton - reconstructed_skeleton) ** 2)
            
            # 计算实际坐标空间的重构误差（更直观）
            reconstructed_original = self._denormalize_skeleton(reconstructed_skeleton, original_skeleton)
            real_world_recon_mse = np.mean((original_skeleton - reconstructed_original) ** 2)
            
            # 计算提取-重构差异（仅用于调试，标准化extracted_skeleton后比较）
            extracted_normalized = self._normalize_skeleton(extracted_skeleton.reshape(1, 25, 3))[0]
            extraction_recon_mse = np.mean((extracted_normalized - reconstructed_skeleton) ** 2)
            
            joint_errors = np.sqrt(np.sum((normalized_skeleton - reconstructed_skeleton) ** 2, axis=1))
            max_joint_error = np.max(joint_errors)
            mean_joint_error = np.mean(joint_errors)
            
            # 获取VQ损失
            vq_loss = result['reconstruction_result'].get('vq_loss', 0.0)
            
            # 创建单独的图形 - 1行3列布局
            fig = plt.figure(figsize=(18, 6))
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 雷达输入可视化 (显示雷达特征图信息)
            ax1 = fig.add_subplot(131, projection='3d')
            # 由于原始骨架数据不直接可用，显示雷达输入的维度信息
            ax1.text(0.5, 0.5, 0.5, f'雷达特征图\n{radar_input.shape}', ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            ax1.set_title(f'样本 {sample_idx}: 雷达输入', fontsize=12, fontweight='bold')
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            ax1.set_zlim([0, 1])
            
            # Extracted skeleton
            ax2 = fig.add_subplot(132, projection='3d')
            self._plot_skeleton_3d(ax2, extracted_skeleton, f'Sample {sample_idx}: Extracted', 'green')
            
            # Reconstructed skeleton 
            ax3 = fig.add_subplot(133, projection='3d')
            self._plot_skeleton_3d(ax3, reconstructed_skeleton, f'Sample {sample_idx}: GCN Reconstructed', 'red')
            
            # 添加误差信息 - 明确显示两种MSE，主要关注标准化MSE（与训练时一致）
            fig.suptitle(f'样本 {sample_idx} - 骨架提取与重构对比\n'
                        f'标准化MSE: {normalized_recon_mse:.6f} | 实际坐标MSE: {real_world_recon_mse:.4f} | 提取-重构MSE: {extraction_recon_mse:.6f}',
                        fontsize=16, fontweight='bold', y=0.95)
            
            # 添加处理信息
            info_text = f"Radar Input: {radar_input.shape} -> Extracted: {extracted_skeleton.shape} -> Reconstructed: {reconstructed_skeleton.shape}"
            
            fig.text(0.02, 0.02, info_text, fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9),
                    verticalalignment='bottom')
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(output_dir, f'extraction_reconstruction_sample_{sample_idx:02d}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"✅ 已保存样本 {sample_idx:02d}: {os.path.basename(save_path)}")
            
            individual_metrics.append({
                'sample_id': sample_idx,
                'normalized_recon_mse': normalized_recon_mse,  # 主要指标：与训练时一致的标准化MSE
                'real_world_recon_mse': real_world_recon_mse,  # 辅助指标：实际坐标空间MSE
                'extraction_recon_mse': extraction_recon_mse,  # 调试指标：提取-重构MSE
                'vq_loss': vq_loss,
                'max_joint_error': max_joint_error,
                'mean_joint_error': mean_joint_error,
                'file_path': save_path
            })
        
        return individual_metrics
    
    def generate_sequence_gif_animations(self, radar_data_path, output_dir, num_sequences=5, frames_per_sequence=8, fps=3):
        """生成相邻几帧的GIF动画展示重构过程
        
        Args:
            radar_data_path: 雷达数据路径
            output_dir: GIF保存目录
            num_sequences: 生成序列数量
            frames_per_sequence: 每个序列的帧数
            fps: GIF帧率
        """
        print(f"🎬 生成骨架重构GIF动画...")
        
        # 创建GIF输出目录
        gif_output_dir = os.path.join(output_dir, "../skeleton_extraction_gif_reconstruction_new")
        os.makedirs(gif_output_dir, exist_ok=True)
        
        # 加载完整的雷达数据
        if not os.path.exists(radar_data_path):
            print(f"❌ 雷达数据文件不存在: {radar_data_path}")
            return []
        
        # 加载标签数据
        labels_path = '/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy'
        if not os.path.exists(labels_path):
            print(f"❌ 标签数据文件不存在: {labels_path}")
            return []
        
        full_data = np.load(radar_data_path)
        label_data = np.load(labels_path)
        print(f"✅ 加载完整雷达数据: {full_data.shape}")
        print(f"✅ 加载标签数据: {label_data.shape}")
        
        gif_info_list = []
        
        # 生成多个序列的GIF
        for seq_idx in range(num_sequences):
            # 为每个序列选择不同的起始位置
            start_idx = seq_idx * (len(full_data) // (num_sequences + 1))
            end_idx = min(start_idx + frames_per_sequence, len(full_data))
            
            if end_idx - start_idx < frames_per_sequence:
                # 如果数据不够，从末尾向前取
                end_idx = len(full_data) - 1
                start_idx = max(0, end_idx - frames_per_sequence + 1)
            
            print(f"📹 生成序列 {seq_idx+1}/{num_sequences}: 帧 {start_idx}-{end_idx-1}")
            
            # 提取序列数据
            sequence_data = full_data[start_idx:end_idx]
            sequence_labels = label_data[start_idx:end_idx]
            
            # 处理序列中的每一帧
            frame_results = []
            for frame_idx, (radar_frame, label_frame) in enumerate(zip(sequence_data, sequence_labels)):
                # 处理单帧
                frame_result = self.process_complete_pipeline(radar_frame.reshape(1, 8, 8, 5))
                
                # 提取重构结果
                try:
                    vq_loss = frame_result['reconstruction_result']['vq_loss']
                except:
                    vq_loss = 0.0
                
                frame_results.append({
                    'frame_idx': frame_idx,
                    'label': label_frame,  # 添加标签数据
                    'extracted': frame_result['extracted_skeleton'][0].cpu().numpy(),
                    'reconstructed': frame_result['reconstruction_result']['reconstructed'][0],
                    'vq_loss': vq_loss
                })
            
            # 生成GIF
            gif_path = os.path.join(gif_output_dir, f'skeleton_reconstruction_sequence_{seq_idx+1:02d}.gif')
            gif_info = self._create_skeleton_sequence_gif(frame_results, gif_path, fps)
            gif_info['sequence_id'] = seq_idx + 1
            gif_info['start_frame'] = start_idx
            gif_info['end_frame'] = end_idx - 1
            gif_info_list.append(gif_info)
            
        return gif_info_list
    
    def _create_skeleton_sequence_gif(self, frame_results, gif_path, fps=3):
        """创建单个序列的骨架重构GIF动画"""
        
        num_frames = len(frame_results)
        if num_frames == 0:
            return {'success': False, 'path': gif_path}
        
        # 创建图形布局: 1行3列
        fig = plt.figure(figsize=(24, 8))
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'Droid Sans Fallback']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        def animate(frame_idx):
            """动画更新函数"""
            fig.clear()
            
            # 获取当前帧数据
            current_frame = frame_results[frame_idx]
            label = current_frame['label']
            extracted = current_frame['extracted']
            reconstructed = current_frame['reconstructed']
            vq_loss = current_frame['vq_loss']
            
            # 将标签数据从MARS 57维格式转换为NTU 25关节格式，与提取骨架保持一致
            def convert_mars_to_ntu_format(mars_57_data):
                """将MARS 57维数据转换为NTU 25关节格式，复用现有的映射器"""
                # 使用关节映射器将MARS数据转换为NTU格式
                mars_tensor = torch.tensor(mars_57_data).unsqueeze(0).float().to(self.device)
                ntu_skeleton = self.joint_mapper(mars_tensor)
                return ntu_skeleton[0].detach().cpu().numpy()  # 返回(25, 3)格式
            
            # 转换标签数据为NTU格式
            label_ntu = convert_mars_to_ntu_format(label)
            
            # 正确的MSE计算方式：
            # 1. 将提取的骨架标准化后与重构骨架比较（这是训练时的评估方式）
            extracted_normalized = self._normalize_skeleton(extracted.reshape(1, 25, 3))[0]
            
            # 确保reconstructed数据格式正确
            if isinstance(reconstructed, torch.Tensor):
                reconstructed_array = reconstructed.cpu().numpy()
            else:
                reconstructed_array = reconstructed
            
            # 确保数据形状一致，并且重构数据已经是标准化的
            if reconstructed_array.shape != extracted_normalized.shape:
                if len(reconstructed_array.shape) == 1 and reconstructed_array.shape[0] == 75:
                    # 如果是75维扁平化数据，重塑为(25, 3)
                    reconstructed_array = reconstructed_array.reshape(25, 3)
                elif len(reconstructed_array.shape) == 2 and reconstructed_array.shape[0] == 1:
                    # 如果是(1, 75)，先展平再重塑
                    reconstructed_array = reconstructed_array.flatten().reshape(25, 3)
                else:
                    print(f"警告：重构数据形状不匹配: {reconstructed_array.shape} vs {extracted_normalized.shape}")
            
            # 检查重构数据是否在合理范围内（标准化数据应该在[0,1]范围内）
            if reconstructed_array.min() < -0.1 or reconstructed_array.max() > 1.1:
                print(f"警告：重构数据可能未标准化，范围: [{reconstructed_array.min():.3f}, {reconstructed_array.max():.3f}]")
                # 如果重构数据不在标准化范围内，可能需要重新标准化
                # 但这里我们假设模型输出已经是标准化的
            
            # 标准化空间MSE（与训练时一致的评估指标）
            normalized_mse_error = np.mean((extracted_normalized - reconstructed_array) ** 2)
            
            # 实际坐标空间的重构误差（将重构数据反标准化后比较）
            try:
                # 将重构骨架反标准化到原始坐标空间
                reconstructed_denormalized = self._denormalize_skeleton(reconstructed_array.reshape(1, 25, 3), extracted.reshape(1, 25, 3))[0]
                real_world_recon_error = np.mean((extracted - reconstructed_denormalized) ** 2)
                
                # 计算Ground Truth与提取骨架的差异（参考）
                real_world_extraction_error = np.mean((label_ntu - extracted) ** 2)
            except Exception as e:
                print(f"警告：反标准化失败: {e}")
                # 备用计算：使用标准化空间的MSE作为参考
                real_world_recon_error = normalized_mse_error
                real_world_extraction_error = normalized_mse_error
            
            # 创建子图: 3列布局
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d') 
            ax3 = fig.add_subplot(133, projection='3d')
            
            # 绘制标签骨架 (Ground Truth) - 现在使用相同的NTU格式和绘制函数
            self._plot_skeleton_3d(ax1, label_ntu, 
                                 f'Frame {frame_idx+1}/{num_frames}: Ground Truth', 'blue')
            
            # 绘制提取的骨架
            self._plot_skeleton_3d(ax2, extracted, 
                                 f'Frame {frame_idx+1}/{num_frames}: Extracted Skeleton', 'green')
            
            # 绘制重构的骨架 - 使用正确的数据格式
            if isinstance(reconstructed_array, np.ndarray) and reconstructed_array.shape == (25, 3):
                self._plot_skeleton_3d(ax3, reconstructed_array,
                                     f'Frame {frame_idx+1}/{num_frames}: Reconstructed Skeleton\nVQ Loss: {vq_loss:.4f}', 'red')
            else:
                # 如果数据格式不正确，尝试反标准化到原始空间进行显示
                try:
                    reconstructed_denormalized = self._denormalize_skeleton(reconstructed_array.reshape(1, 25, 3), extracted.reshape(1, 25, 3))[0]
                    self._plot_skeleton_3d(ax3, reconstructed_denormalized,
                                         f'Frame {frame_idx+1}/{num_frames}: Reconstructed Skeleton\nVQ Loss: {vq_loss:.4f}', 'red')
                except:
                    # 最后的备用方案
                    self._plot_skeleton_3d(ax3, reconstructed_array.reshape(25, 3),
                                         f'Frame {frame_idx+1}/{num_frames}: Reconstructed Skeleton\nVQ Loss: {vq_loss:.4f}', 'red')
            
            # 构建标题文本
            title_text = f'Complete Skeleton Reconstruction Pipeline\nFrame {frame_idx+1}/{num_frames} | Normalized MSE: {normalized_mse_error:.6f} | Real-world MSE: {real_world_recon_error:.4f}'
            
            fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.95)
            
            # 调整布局
            plt.tight_layout()
        
        # 创建动画
        try:
            anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, blit=False, repeat=True)
            
            # 保存GIF
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer, dpi=150)
            plt.close(fig)
            
            print(f"✅ GIF保存成功: {os.path.basename(gif_path)}")
            
            # 计算序列统计 - 修正：使用正确的标准化MSE计算
            normalized_mse_errors = []
            real_world_mse_errors = []  # 实际坐标空间MSE
            vq_losses = [fr['vq_loss'] for fr in frame_results]
            
            for fr in frame_results:
                # 标准化MSE：将提取的骨架标准化后与重构骨架比较（正确的MSE）
                extracted_normalized = self._normalize_skeleton(fr['extracted'].reshape(1, 25, 3))[0]
                
                # 确保重构数据格式正确并且转换为numpy
                reconstructed_data = fr['reconstructed']
                if isinstance(reconstructed_data, torch.Tensor):
                    reconstructed_data = reconstructed_data.cpu().numpy()
                    
                # 数据形状检查和修正
                if reconstructed_data.shape != (25, 3):
                    if len(reconstructed_data.shape) == 1 and reconstructed_data.shape[0] == 75:
                        reconstructed_data = reconstructed_data.reshape(25, 3)
                    elif len(reconstructed_data.shape) == 2 and reconstructed_data.shape[0] == 1:
                        reconstructed_data = reconstructed_data.flatten().reshape(25, 3)
                    else:
                        print(f"⚠️ 警告：重构数据形状异常: {reconstructed_data.shape}，跳过此帧")
                        continue
                        
                # 数据范围验证：重构数据应该在标准化空间(-2, 2)范围内
                # 注意：标准化后的数据不一定在[0,1]，而是在质心归零、最大距离归一化后的范围
                data_min, data_max = reconstructed_data.min(), reconstructed_data.max()
                if data_min < -2.0 or data_max > 2.0:
                    print(f"⚠️ 警告：重构数据范围 [{data_min:.3f}, {data_max:.3f}] 超出预期范围 [-2, 2]")
                    print(f"   这可能表明模型输出异常，但仍继续计算MSE")
                
                # 计算标准化MSE（主要评估指标，与训练时一致）
                normalized_mse = np.mean((extracted_normalized - reconstructed_data) ** 2)
                normalized_mse_errors.append(normalized_mse)
                
                # 实际坐标空间MSE：将重构骨架反标准化后与提取骨架比较
                try:
                    reconstructed_denormalized = self._denormalize_skeleton(
                        reconstructed_data.reshape(1, 25, 3), 
                        fr['extracted'].reshape(1, 25, 3)
                    )[0]
                    real_world_mse = np.mean((fr['extracted'] - reconstructed_denormalized) ** 2)
                    real_world_mse_errors.append(real_world_mse)
                except Exception as e:
                    print(f"警告：反标准化失败: {e}")
                    # 备用：使用标准化MSE作为近似
                    real_world_mse_errors.append(normalized_mse * 1000)  # 粗略转换
            
            return {
                'success': True,
                'path': gif_path,
                'num_frames': num_frames,
                'avg_normalized_mse': np.mean(normalized_mse_errors),  # 主要指标：标准化MSE
                'avg_real_world_mse': np.mean(real_world_mse_errors) if real_world_mse_errors else 0.0,  # 实际坐标MSE
                'avg_vq_loss': np.mean(vq_losses),
                'frame_range': (0, num_frames-1)
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ GIF生成失败: {e}")
            print(f"详细错误: {error_details}")
            plt.close(fig)
            return {
                'success': False,
                'path': gif_path,
                'error': str(e),
                'error_details': error_details
            }

def load_test_radar_data(data_path, num_samples=12):
    """加载测试雷达数据 - 增加样本数量以展示更多动作"""
    print(f"📁 加载测试数据: {data_path}")
    
    try:
        # 尝试加载测试数据
        if os.path.exists(data_path):
            test_data = np.load(data_path)
            print(f"✅ 成功加载数据，形状: {test_data.shape}")
            
            # 为了获得更多样化的动作，从不同位置采样
            if len(test_data) > num_samples:
                # 均匀采样以获得更多样化的动作
                indices = np.linspace(0, len(test_data) - 1, num_samples, dtype=int)
                test_data = test_data[indices]
                print(f"✅ 从 {len(test_data)} 个样本中均匀采样 {num_samples} 个，索引: {indices}")
            
            return test_data
        else:
            print(f"❌ 数据文件不存在: {data_path}")
            print("🔄 生成模拟雷达数据...")
            
            # 生成模拟数据
            mock_data = np.random.rand(num_samples, 8, 8, 5).astype(np.float32)
            return mock_data
            
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        print("🔄 生成模拟雷达数据...")
        
        # 生成模拟数据
        mock_data = np.random.rand(num_samples, 8, 8, 5).astype(np.float32)
        return mock_data

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 骨架提取 + GCN重构完整流程")
    print("=" * 80)
    
    # 配置文件路径
    extractor_model_path = "mars_transformer_best.pth"
    gcn_model_path = "experiments/gcn_skeleton_memory_optimized/NTU_models/default/ckpt-best.pth"
    gcn_config_path = "cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml"
    
    # 检查文件存在性
    if not os.path.exists(extractor_model_path):
        print(f"❌ 骨架提取器权重不存在: {extractor_model_path}")
        return
        
    if not os.path.exists(gcn_model_path):
        print(f"❌ GCN重构器权重不存在: {gcn_model_path}")
        return
        
    if not os.path.exists(gcn_config_path):
        print(f"❌ GCN配置文件不存在: {gcn_config_path}")
        return
    
    print("✅ 所有必需文件存在")
    
    try:
        # 创建流水线
        print("\\n🏗️ 初始化处理流水线...")
        pipeline = SkeletonExtractionReconstructionPipeline(
            extractor_model_path, gcn_model_path, gcn_config_path
        )
        
        # 加载测试数据
        print("\\n📊 加载测试数据...")
        radar_data_path = "/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_test.npy"
        
        # 创建输出目录
        output_dir = "visualizations/skeleton_extraction_reconstruction_new"
        os.makedirs(output_dir, exist_ok=True)
        
        # 增加样本数量以展示更多动作
        num_samples = 12  # 增加到12个样本
        test_radar_data = load_test_radar_data(radar_data_path, num_samples=num_samples)
        
        # 处理每个测试样本
        print(f"\\n🎯 处理 {len(test_radar_data)} 个测试样本...")
        
        all_pipeline_results = []
        all_results = []
        
        for i, radar_sample in enumerate(test_radar_data):
            print(f"\\n--- 处理样本 {i+1}/{len(test_radar_data)} ---")
            
            # 执行完整流程
            pipeline_result = pipeline.process_complete_pipeline(radar_sample)
            all_pipeline_results.append(pipeline_result)
        
        # 生成个别样本可视化
        print(f"\\n🖼️ 生成每个样本的单独可视化...")
        individual_metrics = pipeline.generate_individual_sample_visualizations(all_pipeline_results, output_dir)
        
        # 生成多样本综合可视化
        print(f"\\n🎨 生成多样本综合可视化...")
        multi_sample_save_path = os.path.join(output_dir, f'multi_sample_skeleton_analysis.png')
        multi_metrics = pipeline.visualize_results(all_pipeline_results, multi_sample_save_path)
        
        # 生成GIF动画序列
        print(f"\\n🎬 生成骨架重构GIF动画序列...")
        gif_info_list = pipeline.generate_sequence_gif_animations(
            radar_data_path=radar_data_path,
            output_dir=output_dir,
            num_sequences=6,  # 生成6个GIF序列
            frames_per_sequence=6,  # 每个序列6帧
            fps=2  # 2帧每秒，较慢以便观察细节
        )
            
        # 从多样本指标中提取每个样本的结果
        for i, sample_detail in enumerate(multi_metrics['sample_details']):
            result_summary = {
                'sample_id': int(i + 1),
                'recon_mse_error': float(sample_detail['recon_mse_error']),
                'vq_loss': float(sample_detail['vq_loss']),
                'max_joint_error': float(sample_detail['max_joint_error']),
                'mean_joint_error': float(sample_detail['mean_joint_error']),

            }
            all_results.append(result_summary)
            
            print(f"  📊 样本 {i+1} 指标:")
            print(f"     Real Reconstruction MSE: {sample_detail['recon_mse_error']:.6f}")
            print(f"     VQ损失: {sample_detail['vq_loss']:.6f}")
            print(f"     最大关节误差: {sample_detail['max_joint_error']:.6f}")
            print(f"     平均关节误差: {sample_detail['mean_joint_error']:.6f}")
        
        # 保存所有结果
        import json
        results_path = os.path.join(output_dir, 'pipeline_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # 计算总体统计
        print("\\n" + "=" * 80)
        print("📈 总体统计结果:")
        print("=" * 80)
        
        mse_errors = [r['recon_mse_error'] for r in all_results]
        vq_losses = [r['vq_loss'] for r in all_results]
        
        print(f"Average Real Reconstruction MSE: {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
        print(f"Average VQ Loss: {np.mean(vq_losses):.6f} ± {np.std(vq_losses):.6f}")
        print(f"Best Real Reconstruction MSE: {np.min(mse_errors):.6f}")
        print(f"Worst Real Reconstruction MSE: {np.max(mse_errors):.6f}")
        
        print(f"\\n📁 输出文件:")
        print(f"  个别样本可视化: {len(individual_metrics)} 张PNG图片")
        print(f"  多样本综合图: {os.path.basename(multi_sample_save_path)}")  
        print(f"  GIF动画序列: {len(gif_info_list)} 个GIF文件")
        print(f"  统计结果: {os.path.basename(results_path)}")
        print(f"  PNG输出目录: {output_dir}/")
        print(f"  GIF输出目录: visualizations/skeleton_extraction_gif_reconstruction/")
        
        # 显示生成的图片列表
        print(f"\\n🖼️ 生成的可视化图片:")
        for i, metric in enumerate(individual_metrics):
            print(f"  样本 {i+1:02d}: extraction_reconstruction_sample_{i+1:02d}.png (标准化MSE: {metric['normalized_recon_mse']:.6f})")
        
        # 显示生成的GIF列表
        print(f"\\n🎬 生成的GIF动画:")
        for gif_info in gif_info_list:
            if gif_info['success']:
                print(f"  序列 {gif_info['sequence_id']:02d}: {os.path.basename(gif_info['path'])} "
                      f"(帧 {gif_info['start_frame']}-{gif_info['end_frame']}, "
                      f"平均标准化MSE: {gif_info.get('avg_normalized_mse', 'N/A'):.6f})")
            else:
                print(f"  序列 {gif_info['sequence_id']:02d}: 生成失败 - {gif_info.get('error', '未知错误')}")
        
        print("\\n🎉 骨架提取+GCN重构流程完成！")
        
    except Exception as e:
        print(f"\\n❌ 流程执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
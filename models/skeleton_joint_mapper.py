#!/usr/bin/env python3
"""
骨架关节点映射转换器
解决MARS 19关节点 与 NTU RGB+D 25关节点 之间的兼容性问题
"""

import torch
import torch.nn as nn
import numpy as np


class SkeletonJointMapper(nn.Module):
    """骨架关节点映射转换器
    
    将MARS的19关节点映射到NTU RGB+D的25关节点标准
    
    MARS 19关节点 (标号1-19，基于用户提供的真实连接关系):
    1: spinebase     2: spinemid     3: head         4: neck
    5: leftshoulder  6: leftelbow    7: leftwrist    8: rightshoulder  
    9: rightelbow    10: rightwrist  11: hipleft     12: kneeleft
    13: ankleleft    14: footleft    15: hipright    16: kneeright
    17: ankleright   18: footright   19: spineshoulder
    
    NTU RGB+D 25关节点 (标号0-24):
    0: spinebase      1: spinemid     2: neck         3: head
    4: leftshoulder   5: leftelbow    6: leftwrist    7: lefthand
    8: rightshoulder  9: rightelbow   10: rightwrist  11: righthand
    12: lefthip       13: leftknee    14: leftankle   15: leftfoot
    16: righthip      17: rightknee   18: rightankle  19: rightfoot
    20: spineshoulder 21: lefthandtip 22: leftthumb   23: righthandtip 
    24: rightthumb
    """
    
    def __init__(self):
        super().__init__()
        
        # 定义关节点映射关系
        self._build_mapping_matrix()
        
    def _build_mapping_matrix(self):
        """构建映射矩阵和插值规则"""
        
        # MARS到NTU的直接映射关系 
        # 注意: MARS使用1-19标号，但在57维向量中以0-18索引访问
        # 即MARS关节点i对应57维向量中的索引(i-1)
        self.direct_mapping = {
            # 头部和脊柱 (MARS索引 -> NTU索引)
            2: 3,   # MARS[3] head -> NTU[3] head  (向量索引2)
            3: 2,   # MARS[4] neck -> NTU[2] neck  (向量索引3)
            18: 20, # MARS[19] spineshoulder -> NTU[20] spineshoulder (向量索引18)
            1: 1,   # MARS[2] spinemid -> NTU[1] spinemid (向量索引1)
            0: 0,   # MARS[1] spinebase -> NTU[0] spinebase (向量索引0)
            
            # 左臂
            4: 4,   # MARS[5] leftshoulder -> NTU[4] leftshoulder (向量索引4)
            5: 5,   # MARS[6] leftelbow -> NTU[5] leftelbow (向量索引5)
            6: 6,   # MARS[7] leftwrist -> NTU[6] leftwrist (向量索引6)
            
            # 右臂
            7: 8,   # MARS[8] rightshoulder -> NTU[8] rightshoulder (向量索引7)
            8: 9,   # MARS[9] rightelbow -> NTU[9] rightelbow (向量索引8)
            9: 10,  # MARS[10] rightwrist -> NTU[10] rightwrist (向量索引9)
            
            # 左腿
            10: 12, # MARS[11] hipleft -> NTU[12] lefthip (向量索引10)
            11: 13, # MARS[12] kneeleft -> NTU[13] leftknee (向量索引11)
            12: 14, # MARS[13] ankleleft -> NTU[14] leftankle (向量索引12)
            13: 15, # MARS[14] footleft -> NTU[15] leftfoot (向量索引13)
            
            # 右腿
            14: 16, # MARS[15] hipright -> NTU[16] righthip (向量索引14)
            15: 17, # MARS[16] kneeright -> NTU[17] rightknee (向量索引15)
            16: 18, # MARS[17] ankleright -> NTU[18] rightankle (向量索引16)
            17: 19, # MARS[18] footright -> NTU[19] rightfoot (向量索引17)
        }
        
        # 需要插值生成的关节点 (NTU缺失的手部关节点)
        self.interpolated_joints = {
            7: [(6, 0.8), (20, 0.2)],   # NTU[7] lefthand = 0.8*leftwrist + 0.2*spineshoulder
            11: [(10, 0.8), (20, 0.2)], # NTU[11] righthand = 0.8*rightwrist + 0.2*spineshoulder
            21: [(6, 1.2), (7, -0.2)],  # NTU[21] lefthandtip = 1.2*leftwrist - 0.2*lefthand(外推)
            22: [(7, 0.9), (6, 0.1)],   # NTU[22] leftthumb = 0.9*lefthand + 0.1*leftwrist
            23: [(10, 1.2), (11, -0.2)], # NTU[23] righthandtip = 1.2*rightwrist - 0.2*righthand
            24: [(11, 0.9), (10, 0.1)],  # NTU[24] rightthumb = 0.9*righthand + 0.1*rightwrist
        }
        
        # 创建映射矩阵 (25, 19) - MARS有19个关节点
        self.register_buffer('mapping_matrix', torch.zeros(25, 19))
        self.register_buffer('interpolation_matrix', torch.zeros(25, 25))
        
        # 填充直接映射 (mars_idx是在57维向量中的索引0-18)
        for mars_vector_idx, ntu_idx in self.direct_mapping.items():
            self.mapping_matrix[ntu_idx, mars_vector_idx] = 1.0
            
        # 创建插值矩阵
        self.interpolation_matrix = torch.eye(25)
        for ntu_idx, interpolation_rule in self.interpolated_joints.items():
            self.interpolation_matrix[ntu_idx, ntu_idx] = 0  # 清零自身
            for source_ntu_idx, weight in interpolation_rule:
                self.interpolation_matrix[ntu_idx, source_ntu_idx] = weight
    
    def forward(self, mars_skeleton):
        """将MARS 19关节点转换为NTU 25关节点
        
        Args:
            mars_skeleton: (B, 19, 3) 或 (B, 57) MARS格式骨架
            
        Returns:
            ntu_skeleton: (B, 25, 3) NTU RGB+D格式骨架
        """
        batch_size = mars_skeleton.shape[0]
        
        # 处理输入格式
        if len(mars_skeleton.shape) == 2 and mars_skeleton.shape[1] == 57:
            # 从(B, 57)转换为(B, 19, 3)
            # MARS数据排列：(x1...x19, y1...y19, z1...z19)
            x_coords = mars_skeleton[:, 0:19]    # x坐标: 0-18
            y_coords = mars_skeleton[:, 19:38]   # y坐标: 19-37  
            z_coords = mars_skeleton[:, 38:57]   # z坐标: 38-56
            mars_joints = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # (B, 19, 3)
        elif len(mars_skeleton.shape) == 3 and mars_skeleton.shape[1] == 19:
            mars_joints = mars_skeleton
        else:
            raise ValueError(f"Unexpected input shape: {mars_skeleton.shape}, expected (B, 57) or (B, 19, 3)")
        
        # 步骤1: 直接映射
        # 将(B, 19, 3)转换为(B, 19*3)用于矩阵运算
        mars_flat = mars_joints.view(batch_size, 19, 3)
        ntu_joints = torch.zeros(batch_size, 25, 3, device=mars_skeleton.device, dtype=mars_skeleton.dtype)
        
        # 应用直接映射 (mars_vector_idx是在19关节数组中的索引0-18)
        for mars_vector_idx, ntu_idx in self.direct_mapping.items():
            ntu_joints[:, ntu_idx, :] = mars_flat[:, mars_vector_idx, :]
        
        # 步骤2: 插值生成缺失关节点
        for ntu_idx, interpolation_rule in self.interpolated_joints.items():
            interpolated_joint = torch.zeros_like(ntu_joints[:, 0, :])
            for source_ntu_idx, weight in interpolation_rule:
                interpolated_joint += weight * ntu_joints[:, source_ntu_idx, :]
            ntu_joints[:, ntu_idx, :] = interpolated_joint
        
        # 修复MARS→NTU坐标系差异
        # 测试: 只反转Z轴(前后)，保持Y轴(上下)不变
        rotation_matrix = torch.tensor([
            [1.0, 0.0, 0.0],    # X轴保持不变（左右方向正确）
            [0.0, 1.0, 0.0],    # Y轴保持不变（上下方向保持）
            [0.0, 0.0, -1.0]    # Z轴反向（前后方向反转，胸部朝前）
        ], device=ntu_joints.device, dtype=ntu_joints.dtype)
        
        # 应用旋转矩阵
        ntu_joints = torch.matmul(ntu_joints, rotation_matrix.T)
        
        return ntu_joints
    
    def get_mapping_info(self):
        """获取映射信息用于调试"""
        info = {
            'mars_joints': 19,
            'ntu_joints': 25,
            'direct_mappings': len(self.direct_mapping),
            'interpolated_joints': len(self.interpolated_joints),
            'mapping_details': {
                'direct': self.direct_mapping,
                'interpolated': self.interpolated_joints
            }
        }
        return info


class EnhancedSkeletonMapper(nn.Module):
    """增强版骨架映射器 - 使用学习的方式优化映射"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # 基础映射器
        self.base_mapper = SkeletonJointMapper()
        
        # 学习网络来细化映射 (MARS输入是57维)
        self.refinement_net = nn.Sequential(
            nn.Linear(57, hidden_dim),  # 直接处理57维MARS输出
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6 * 3),  # 6个插值关节点的偏移
            nn.Tanh()  # 限制偏移范围
        )
        
        # 偏移缩放因子
        self.offset_scale = 0.1
        
    def forward(self, mars_skeleton):
        """增强映射"""
        batch_size = mars_skeleton.shape[0]
        
        # 获取基础映射结果
        base_ntu = self.base_mapper(mars_skeleton)
        
        # 学习偏移量 (确保输入是57维)
        if len(mars_skeleton.shape) == 3 and mars_skeleton.shape[1] == 19:
            # 从(B, 19, 3)转换回MARS的57维排列格式 (x1...x19, y1...y19, z1...z19)
            x_coords = mars_skeleton[:, :, 0]  # (B, 19)
            y_coords = mars_skeleton[:, :, 1]  # (B, 19)
            z_coords = mars_skeleton[:, :, 2]  # (B, 19)
            mars_flat = torch.cat([x_coords, y_coords, z_coords], dim=1)  # (B, 57)
        elif len(mars_skeleton.shape) == 2 and mars_skeleton.shape[1] == 57:
            mars_flat = mars_skeleton
        else:
            raise ValueError(f"Expected MARS input (B,57) or (B,19,3), got {mars_skeleton.shape}")
            
        offsets = self.refinement_net(mars_flat) * self.offset_scale  # (B, 18)
        offsets = offsets.view(batch_size, 6, 3)  # (B, 6, 3)
        
        # 应用偏移到插值的关节点
        refined_ntu = base_ntu.clone()
        interpolated_indices = [7, 11, 21, 22, 23, 24]  # 6个插值关节点
        
        for i, ntu_idx in enumerate(interpolated_indices):
            refined_ntu[:, ntu_idx, :] += offsets[:, i, :]
            
        return refined_ntu


def demo_joint_mapping():
    """演示关节点映射过程"""
    print("=" * 80)
    print("🦴 骨架关节点映射演示")
    print("=" * 80)
    
    # 创建映射器
    mapper = SkeletonJointMapper()
    enhanced_mapper = EnhancedSkeletonMapper()
    
    # 获取映射信息
    info = mapper.get_mapping_info()
    print(f"📊 映射信息:")
    print(f"   MARS关节点数: {info['mars_joints']}")
    print(f"   NTU关节点数: {info['ntu_joints']}")
    print(f"   直接映射: {info['direct_mappings']}个")
    print(f"   插值生成: {info['interpolated_joints']}个")
    
    # 模拟MARS骨架数据
    batch_size = 4
    mars_skeleton_57d = torch.randn(batch_size, 57) * 0.5  # (B, 57)
    mars_skeleton_19j = torch.randn(batch_size, 19, 3) * 0.5  # (B, 19, 3)
    
    print(f"\n🔄 映射测试:")
    
    # 测试57维输入
    ntu_from_57d = mapper(mars_skeleton_57d)
    print(f"   输入: MARS {mars_skeleton_57d.shape} -> 输出: NTU {ntu_from_57d.shape}")
    
    # 测试19×3输入  
    ntu_from_19j = mapper(mars_skeleton_19j)
    print(f"   输入: MARS {mars_skeleton_19j.shape} -> 输出: NTU {ntu_from_19j.shape}")
    
    # 测试增强映射器
    enhanced_ntu = enhanced_mapper(mars_skeleton_57d)
    print(f"   增强映射: MARS {mars_skeleton_57d.shape} -> 输出: NTU {enhanced_ntu.shape}")
    
    # 计算映射前后的差异
    basic_diff = torch.norm(ntu_from_57d - ntu_from_19j.view_as(ntu_from_57d), dim=-1).mean()
    enhanced_diff = torch.norm(enhanced_ntu - ntu_from_57d, dim=-1).mean()
    
    print(f"\n📈 质量分析:")
    print(f"   基础映射差异: {basic_diff:.6f}")
    print(f"   增强映射偏移: {enhanced_diff:.6f}")
    
    # 显示关节点映射详情
    print(f"\n🔗 关节点映射详情:")
    
    mars_joint_names = [
        "头顶", "颈部", "右肩", "右肘", "右腕", "左肩", "左肘", "左腕",
        "右髋", "右膝", "右踝", "左髋", "左膝", "左踝", "胸部", "脊椎中段", 
        "骨盆", "左脚尖", "右脚尖"
    ]
    
    ntu_joint_names = [
        "骨盆中心", "脊椎基础", "脊椎中段", "脊椎顶部", "左肩", "左肘", "左腕", "左手",
        "右肩", "右肘", "右腕", "右手", "左髋", "左膝", "左踝", "左脚",
        "右髋", "右膝", "右踝", "右脚", "脊椎肩部", "左手尖", "左拇指", "右手尖", "右拇指"
    ]
    
    print("   直接映射:")
    for mars_idx, ntu_idx in info['mapping_details']['direct'].items():
        mars_name = mars_joint_names[mars_idx] if mars_idx < len(mars_joint_names) else f"#{mars_idx}"
        ntu_name = ntu_joint_names[ntu_idx] if ntu_idx < len(ntu_joint_names) else f"#{ntu_idx}"
        print(f"     MARS[{mars_idx:2d}] {mars_name:>8} -> NTU[{ntu_idx:2d}] {ntu_name}")
    
    print("   插值生成:")
    for ntu_idx, rule in info['mapping_details']['interpolated'].items():
        ntu_name = ntu_joint_names[ntu_idx] if ntu_idx < len(ntu_joint_names) else f"#{ntu_idx}"
        rule_str = " + ".join([f"{w:.1f}*NTU[{src}]" for src, w in rule])
        print(f"     NTU[{ntu_idx:2d}] {ntu_name:>8} = {rule_str}")
    
    return mapper, enhanced_mapper


if __name__ == "__main__":
    demo_joint_mapping()
#!/usr/bin/env python3
"""
骨架数据专用评估指标
"""

import torch
import numpy as np

class SkeletonMetrics:
    """骨架数据重建质量评估指标"""
    
    @staticmethod
    def names():
        return ['MAE', 'MSE', 'RMSE', 'Joint_Acc@0.1', 'Joint_Acc@0.05']
    
    @staticmethod
    def get(pred, gt, threshold_1=0.1, threshold_2=0.05):
        """
        计算骨架重建指标
        Args:
            pred: 预测的骨架点 (B, N, 3)
            gt: 真实的骨架点 (B, N, 3)
            threshold_1: 关节精度阈值1 (默认0.1)
            threshold_2: 关节精度阈值2 (默认0.05)
        Returns:
            list: [MAE, MSE, RMSE, Joint_Acc@0.1, Joint_Acc@0.05]
        """
        pred = pred.detach()
        gt = gt.detach()
        
        # 计算每个关节点的欧氏距离
        joint_distances = torch.sqrt(torch.sum((pred - gt) ** 2, dim=2))  # (B, N)
        
        # MAE (平均绝对误差)
        mae = torch.mean(torch.abs(pred - gt)).item()
        
        # MSE (均方误差)
        mse = torch.mean((pred - gt) ** 2).item()
        
        # RMSE (均方根误差)
        rmse = torch.sqrt(torch.mean(joint_distances ** 2)).item()
        
        # 关节精度：距离小于阈值的关节点比例
        joint_acc_1 = torch.mean((joint_distances < threshold_1).float()).item()
        joint_acc_2 = torch.mean((joint_distances < threshold_2).float()).item()
        
        return [mae, mse, rmse, joint_acc_1, joint_acc_2]
    
    @staticmethod
    def get_detailed_metrics(pred, gt):
        """
        计算详细的骨架重建指标
        Args:
            pred: 预测的骨架点 (B, N, 3)
            gt: 真实的骨架点 (B, N, 3)
        Returns:
            dict: 详细指标字典
        """
        pred = pred.detach()
        gt = gt.detach()
        
        # 计算每个关节点的欧氏距离
        joint_distances = torch.sqrt(torch.sum((pred - gt) ** 2, dim=2))  # (B, N)
        
        # 每个维度的误差
        x_error = torch.mean(torch.abs(pred[:, :, 0] - gt[:, :, 0])).item()
        y_error = torch.mean(torch.abs(pred[:, :, 1] - gt[:, :, 1])).item()
        z_error = torch.mean(torch.abs(pred[:, :, 2] - gt[:, :, 2])).item()
        
        # 整体指标
        mae = torch.mean(torch.abs(pred - gt)).item()
        mse = torch.mean((pred - gt) ** 2).item()
        rmse = torch.sqrt(torch.mean(joint_distances ** 2)).item()
        
        # 不同阈值下的关节精度
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
        joint_accuracies = {}
        for thresh in thresholds:
            acc = torch.mean((joint_distances < thresh).float()).item()
            joint_accuracies[f'Joint_Acc@{thresh}'] = acc
        
        # 平均关节距离误差
        mean_joint_error = torch.mean(joint_distances).item()
        max_joint_error = torch.max(joint_distances).item()
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'X_Error': x_error,
            'Y_Error': y_error,
            'Z_Error': z_error,
            'Mean_Joint_Error': mean_joint_error,
            'Max_Joint_Error': max_joint_error,
            **joint_accuracies
        }
    
    @staticmethod
    def print_metrics(metrics_dict, logger=None):
        """打印指标"""
        def print_fn(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)
        
        print_fn("=" * 60)
        print_fn("骨架重建指标详情:")
        print_fn("=" * 60)
        print_fn(f"平均绝对误差 (MAE):     {metrics_dict['MAE']:.6f}")
        print_fn(f"均方误差 (MSE):         {metrics_dict['MSE']:.6f}")
        print_fn(f"均方根误差 (RMSE):      {metrics_dict['RMSE']:.6f}")
        print_fn("-" * 40)
        print_fn(f"X轴误差:               {metrics_dict['X_Error']:.6f}")
        print_fn(f"Y轴误差:               {metrics_dict['Y_Error']:.6f}")
        print_fn(f"Z轴误差:               {metrics_dict['Z_Error']:.6f}")
        print_fn("-" * 40)
        print_fn(f"平均关节距离误差:       {metrics_dict['Mean_Joint_Error']:.6f}")
        print_fn(f"最大关节距离误差:       {metrics_dict['Max_Joint_Error']:.6f}")
        print_fn("-" * 40)
        print_fn("关节精度 (不同阈值):")
        for key, value in metrics_dict.items():
            if key.startswith('Joint_Acc@'):
                threshold = key.split('@')[1]
                print_fn(f"  阈值 {threshold}:        {value:.4f} ({value*100:.1f}%)")
        print_fn("=" * 60)

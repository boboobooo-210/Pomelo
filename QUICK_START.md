# MCSkeleton 快速启动指南

## 项目概览

MCSkeleton是从CRSkeleton项目复制的完整副本，包含所有必要的代码、配置和预训练模型。

**项目路径**: `/home/uo/myProject/MCSkeleton`  
**虚拟环境**: `pb_final`

---

## 目录结构

```
MCSkeleton/
├── annotations/          # 标注数据
├── cfgs/                # 配置文件
│   ├── MARS_models/     # MARS数据集模型配置
│   ├── MMFI_models/     # MMFI数据集模型配置
│   ├── NTU_models/      # NTU数据集模型配置
│   └── dataset_configs/ # 数据集配置
├── data/                # 数据目录
├── datasets/            # 数据集加载器
├── experiments/         # 实验输出
├── extensions/          # CUDA扩展 (chamfer_dist, emd)
├── models/              # 模型定义
├── Pointnet2_PyTorch/   # PointNet++实现
├── tools/               # 工具脚本
├── utils/               # 工具函数
├── visualizations/      # 可视化相关
├── main.py              # 主入口
├── mars_transformer_best.pth  # 预训练模型
├── requirements.txt     # 依赖列表
└── README.md            # 项目说明
```

---

## 环境激活

```bash
# 激活虚拟环境
conda activate pb_final

# 切换到项目目录
cd /home/uo/myProject/MCSkeleton
```

---

## 快速测试

### 1. 项目完整性测试
```bash
python test_project_integrity.py
```
预期输出：所有测试✅通过

### 2. 查看主程序帮助
```bash
python main.py --help
```

---

## 运行示例

### 训练NTU模型（GCN Skeleton Tokenizer）
```bash
python main.py \
    --config cfgs/NTU_models/gcn_skeleton_tokenizer_25joints.yaml \
    --exp_name ntu_gcn_experiment
```

### 训练MARS模型（Skeleton DVAE）
```bash
python main.py \
    --config cfgs/MARS_models/skeleton_dvae_pretrain.yaml \
    --exp_name mars_dvae_experiment
```

### 使用预训练模型推理
```bash
python main.py \
    --config cfgs/MARS_models/skeleton_dvae_pretrain.yaml \
    --test \
    --ckpts mars_transformer_best.pth
```

---

## 可用配置文件

### NTU数据集模型
- `cfgs/NTU_models/gcn_skeleton_tokenizer_25joints.yaml` - GCN骨骼分词器（25关节点）
- `cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml` - 内存优化版本
- `cfgs/NTU_models/skeleton_dvae_pretrain.yaml` - 骨骼DVAE预训练
- `cfgs/NTU_models/simple_ntu_50epochs.yaml` - 简化版50轮训练

### MARS数据集模型
- `cfgs/MARS_models/skeleton_dvae_pretrain.yaml` - 骨骼DVAE预训练
- `cfgs/MARS_models/skeleton_pose_reconstruction.yaml` - 姿态重建
- `cfgs/MARS_models/dvae.yaml` - 基础DVAE

### MMFI数据集模型
- `cfgs/MMFI_models/kinect_skeleton_vqvae.yaml` - Kinect骨骼VQ-VAE
- `cfgs/MMFI_models/semantic_dvae.yaml` - 语义DVAE

---

## 常用工具

### 数据可视化
```bash
# 骨骼可视化
python visualizations/skeleton_visualizer.py

# 码本可视化
python visualizations/codebook_visualizer.py
```

### 数据预处理
```bash
# 提取骨骼特征
python tools/skeleton_extraction_reconstruction_saver.py

# 数据分布检查
python check_data_distribution.py
```

### 标注工具
```bash
# 码本标注
python tools/token_codebook_annotator.py
```

---

## 依赖检查

所需的Python包（已在pb_final环境中安装）：
- PyTorch >= 1.9.0 (已安装: 2.7.1+cu118)
- NumPy >= 1.21.0
- PyYAML >= 5.4.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0
- h5py >= 3.3.0
- opencv-python >= 4.5.0

如需安装缺失依赖：
```bash
pip install -r requirements.txt
```

---

## 常见问题

### Q1: CUDA扩展未编译怎么办？
```bash
cd extensions/chamfer_dist
python setup.py install

cd ../emd
python setup.py install
```

### Q2: 内存不足怎么办？
使用内存优化配置：
```bash
python main.py --config cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml
```

### Q3: 数据集路径配置？
编辑对应的数据集配置文件：
- NTU: `cfgs/dataset_configs/NTU_base.yaml`
- MARS: `cfgs/dataset_configs/MARS.yaml`
- MMFI: `cfgs/dataset_configs/MMFI.yaml`

---

## 与CRSkeleton的区别

MCSkeleton是CRSkeleton的完整副本，包含：
- ✅ 所有模型代码
- ✅ 所有配置文件
- ✅ 预训练模型 (mars_transformer_best.pth)
- ✅ 工具脚本
- ✅ 数据集加载器
- ✅ 可视化工具
- ✅ CUDA扩展（已编译）

可以独立运行，不依赖CRSkeleton。

---

## 项目验证

运行完整性测试确保所有模块正常：
```bash
python test_project_integrity.py
```

预期输出：
```
============================================================
🎉 所有测试通过！项目已准备就绪。
============================================================
```

---

## 联系与支持

- 原项目: CRSkeleton (`/home/uo/myProject/CRSkeleton`)
- 当前项目: MCSkeleton (`/home/uo/myProject/MCSkeleton`)
- 虚拟环境: pb_final

---

**最后更新**: 2025-11-07

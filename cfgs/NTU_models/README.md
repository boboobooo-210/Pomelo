# NTU RGB+D 模型配置文件

## 📁 文件结构

```
cfgs/NTU_models/
├── README.md                           # 本说明文档
├── dvae.yaml                          # DVAE训练配置
├── skeleton_dvae_pretrain.yaml        # 骨架DVAE预训练配置
├── skeleton_point_bert_pretrain.yaml  # Point-BERT预训练配置
└── skeleton_pose_reconstruction.yaml  # 姿态重建配置
```

## 🎯 训练流程

### 阶段1：DVAE预训练
```bash
# 构建骨架点云码本
python main.py --config cfgs/NTU_models/skeleton_dvae_pretrain.yaml
```

**目标**: 构建8192个码字的骨架点云码本
**数据**: NTU RGB+D单人日常动作+康复动作 (89,652个样本)
**增强**: 25关节 → 505点 → 512点 (20.2倍增强)

### 阶段2：Point-BERT预训练
```bash
# 基于码本的自监督预训练
python main.py --config cfgs/NTU_models/skeleton_point_bert_pretrain.yaml
```

**目标**: 学习骨架点云的通用表示
**任务**: 掩码点云建模 (60%掩码率)
**依赖**: 阶段1的DVAE检查点

### 阶段3：姿态重建微调
```bash
# 下游任务微调
python main.py --config cfgs/NTU_models/skeleton_pose_reconstruction.yaml
```

**目标**: 骨架姿态重建任务
**评估**: 关节位置误差、骨长误差、角度误差
**依赖**: 阶段2的Point-BERT检查点

## 📊 数据集配置

### 基础配置
- **数据集**: `cfgs/dataset_configs/NTU_base.yaml`
- **增强配置**: `cfgs/dataset_configs/NTU_augmented.yaml`

### 数据增强策略
- **原始关节**: 25个关节点
- **连接关系**: 24个骨骼连接
- **插值策略**: 每根骨头21个插值点
- **增强结果**: 25 + 24×20 = 505点
- **目标格式**: padding到512点，16组×32点/组

## 🔧 模型配置

### DVAE配置
```yaml
model: {
  NAME: DiscreteVAE,
  group_size: 32,
  num_group: 16,        # 512点 = 16组 × 32点/组
  encoder_dims: 256,
  num_tokens: 8192,     # 码本大小
  tokens_dims: 256,     # 码字维度
  decoder_dims: 256
}
```

### Point-BERT配置
```yaml
model: {
  NAME: Point_BERT,
  trans_dim: 384,
  depth: 12,
  num_heads: 6,
  mask_ratio: 0.6
}
```

## 🎮 使用方法

### 快速开始
```bash
# 1. 检查数据集
python tools/test_ntu_schemes.py

# 2. 开始DVAE预训练
python main.py --config cfgs/NTU_models/skeleton_dvae_pretrain.yaml

# 3. Point-BERT预训练
python main.py --config cfgs/NTU_models/skeleton_point_bert_pretrain.yaml

# 4. 姿态重建微调
python main.py --config cfgs/NTU_models/skeleton_pose_reconstruction.yaml
```

### 自定义配置
```bash
# 使用自定义实验名称
python main.py --config cfgs/NTU_models/skeleton_dvae_pretrain.yaml \
                --exp_name my_ntu_dvae_experiment

# 修改批次大小
python main.py --config cfgs/NTU_models/skeleton_dvae_pretrain.yaml \
                --opts total_bs 64
```

## 📈 实验追踪

### 检查点位置
- **DVAE**: `./experiments/ntu_skeleton_dvae_pretrain/`
- **Point-BERT**: `./experiments/ntu_skeleton_point_bert_pretrain/`
- **姿态重建**: `./experiments/ntu_skeleton_pose_reconstruction/`

### 日志文件
- **训练日志**: `./experiments/{exp_name}/logs/`
- **TensorBoard**: `./experiments/{exp_name}/tb_logs/`
- **可视化**: `./experiments/{exp_name}/vis/`

## 🔍 配置对比

| 特性 | MARS模型 | NTU模型 |
|------|----------|---------|
| **数据类型** | 雷达点云 | 骨架关节 |
| **原始点数** | 64 | 25 |
| **增强后点数** | 550 | 512 |
| **增强策略** | 雷达特征处理 | 骨骼插值 |
| **码本大小** | 8192 | 8192 |
| **分组方式** | 16×32 | 16×32 |

## ⚠️ 注意事项

1. **依赖关系**: 必须按阶段顺序训练
2. **检查点**: 确保前一阶段的检查点存在
3. **数据路径**: 确认NTU数据集路径正确
4. **内存要求**: 建议至少8GB GPU内存
5. **训练时间**: 每个阶段约需要数小时到数天

## 🐛 故障排除

### 常见问题
1. **数据加载失败**: 检查数据路径和文件权限
2. **检查点缺失**: 确认前一阶段训练完成
3. **内存不足**: 减少批次大小或使用梯度累积
4. **收敛问题**: 调整学习率或增加训练轮数

### 调试命令
```bash
# 检查配置文件
python tools/check_config.py --config cfgs/NTU_models/skeleton_dvae_pretrain.yaml

# 测试数据加载
python tools/test_ntu_schemes.py

# 验证模型创建
python tools/test_model.py --config cfgs/NTU_models/skeleton_dvae_pretrain.yaml
```

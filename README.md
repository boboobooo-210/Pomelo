# CRSkeleton - GCN Skeleton Tokenizer

A PyTorch implementation of Graph Convolutional Network (GCN) based skeleton tokenizer for human action recognition.

## Features

- GCN-based skeleton tokenization
- Support for multiple datasets: NTU RGB+D, MARS, MMFI
- Memory-optimized training pipeline
- DVAE (Discrete Variational Autoencoder) integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CRSkeleton.git
cd CRSkeleton
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Usage

### Training

Train the GCN skeleton tokenizer with memory optimization:

```bash
# 激活conda环境
conda activate pb_final

# 训练（内存优化版本 - 适用于32GB内存）
python main.py --config cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml
```

**内存优化特性:**
- ✅ 批大小优化: 4 (配合梯度累积=2，等效批大小=8)
- ✅ 分组重构损失: 对每个语义组单独计算损失
- ✅ 关节权重优化: 头部、手部、脚部关节权重×2
- ✅ GPU内存管理: 自动内存清理和优化分配
- ✅ 数据加载优化: 2个worker进程，减少内存占用

### Supported Datasets

- **NTU RGB+D**: Human action recognition dataset with skeleton data
- **MARS**: Multi-modal action recognition dataset
- **MMFI**: Multi-modal fitness dataset

### Configuration

Model configurations are stored in the `cfgs/` directory:
- `cfgs/NTU_models/` - NTU RGB+D dataset configurations
- `cfgs/MARS_models/` - MARS dataset configurations  
- `cfgs/MMFI_models/` - MMFI dataset configurations

## Project Structure

```
CRSkeleton/
├── main.py                 # Main training script
├── models/                 # Model implementations
│   ├── GCNSkeletonTokenizer.py
│   ├── Tokenizer.py
│   └── dvae.py
├── datasets/              # Dataset implementations
├── cfgs/                  # Configuration files
├── tools/                 # Training utilities
└── utils/                 # Common utilities
```

## Models

### GCNSkeletonTokenizer
- Graph Convolutional Network for skeleton feature extraction
- Tokenization of skeleton sequences
- Integration with DVAE for reconstruction

### DVAE (Discrete Variational Autoencoder)
- Discrete latent space representation
- Reconstruction loss optimization
- KL divergence regularization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on PointNet++ PyTorch implementation
- Inspired by BERT tokenization mechanisms for skeleton data
# Pomelo

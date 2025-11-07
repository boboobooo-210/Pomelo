MARS+Transformer骨架GIF动画说明 (PyTorch版本)
==================================================

文件类型:
- skeleton_torch_comparison_XX.gif: 左右对比动画 (Ground Truth vs PyTorch Prediction)
- skeleton_torch_overlay_XX.gif: 重叠对比动画 (蓝色GT + 红色PyTorch Prediction)

动画参数:
- 帧数: 8 帧/GIF
- 帧率: 2 FPS
- 总GIF数量: 8
- 生成时间: 2025-11-04T07:33:41

模型信息:
- 架构: MARS+Transformer (PyTorch版本)
- 权重文件: mars_transformer_best.pth
- 框架: PyTorch
- 输入格式: (N, C, H, W) - PyTorch标准格式
- 输出维度: 57 (19个关节点的3D坐标)

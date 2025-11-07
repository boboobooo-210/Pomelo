#!/usr/bin/env python3
"""
LLM Token Annotation Exporter
================================
将标注好的token语义导出为LLM友好的格式,支持:
1. 静态Token描述 (单帧姿态语义)
2. Token序列 → 动作语义 (多帧动作理解)
3. 层级化描述 (部位 + 整体)

作者: Skeleton Tokenizer Team
日期: 2025-11-07
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class LLMAnnotationExporter:
    """将码本标注导出为LLM可理解的知识库"""
    
    def __init__(self, project_root: str = "/home/uo/myProject/CRSkeleton"):
        self.project_root = Path(project_root)
        self.token_analysis_dir = self.project_root / "token_analysis"
        self.annotation_path = self.token_analysis_dir / "codebook_annotations.json"
        
        # 语义分组定义
        self.semantic_groups = {
            'head_spine': [0, 1, 2, 3, 20],  # 头、颈椎、脊柱
            'left_arm': [4, 5, 6, 7, 21, 22],
            'right_arm': [8, 9, 10, 11, 23, 24],
            'left_leg': [12, 13, 14, 15],
            'right_leg': [16, 17, 18, 19]
        }
        
        # 中文部位名称映射
        self.part_names_zh = {
            'head_spine': '头部躯干',
            'left_arm': '左臂',
            'right_arm': '右臂',
            'left_leg': '左腿',
            'right_leg': '右腿'
        }
        
        self.annotations = {}
        self.load_annotations()
    
    def load_annotations(self):
        """加载现有标注"""
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_path}")
        
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = data.get('codebook_annotation', {})
        self.metadata = data.get('metadata', {})
        
        print(f"✅ 已加载标注:")
        print(f"   - 总Token数: {self.metadata.get('total_unique_tokens', 0)}")
        print(f"   - 已标注: {self.metadata.get('annotated_tokens', 0)}")
        print(f"   - 最后更新: {self.metadata.get('last_updated', 'Unknown')}")
    
    def export_llm_knowledge_base(self, output_path: str = None) -> Dict:
        """
        导出LLM知识库 (静态Token语义)
        
        格式:
        {
            "token_semantics": {
                "35": {
                    "body_part": "头部躯干",
                    "description": "左倾斜",
                    "token_id": 35,
                    "group": "head_spine",
                    "joints_involved": [0, 1, 2, 3, 20]
                },
                ...
            },
            "body_part_vocabulary": {
                "头部躯干": ["左倾斜", "右倾斜", "前倾", ...],
                ...
            },
            "metadata": {...}
        }
        """
        if output_path is None:
            output_path = self.token_analysis_dir / "llm_token_knowledge_base.json"
        
        knowledge_base = {
            "token_semantics": {},
            "body_part_vocabulary": defaultdict(set),
            "metadata": {
                "source": "MARS Dataset Token Annotations",
                "total_tokens": self.metadata.get('total_unique_tokens', 0),
                "annotation_date": self.metadata.get('last_updated', ''),
                "format_version": "1.0",
                "description": "骨架姿态码本的语义知识库,用于LLM理解骨架token"
            }
        }
        
        # 处理每个部位的token
        for part_name, token_dict in self.annotations.items():
            part_name_zh = self.part_names_zh.get(part_name, part_name)
            joints = self.semantic_groups.get(part_name, [])
            
            for token_id, description in token_dict.items():
                token_id_int = int(token_id)
                
                # 构建token语义条目
                knowledge_base["token_semantics"][token_id] = {
                    "token_id": token_id_int,
                    "body_part": part_name_zh,
                    "body_part_en": part_name,
                    "description": description,
                    "joints_involved": joints,
                    "example_usage": f"当{part_name_zh}处于'{description}'状态时,使用token_{token_id}"
                }
                
                # 构建部位词汇表
                knowledge_base["body_part_vocabulary"][part_name_zh].add(description)
        
        # 转换set为list (JSON序列化需要)
        knowledge_base["body_part_vocabulary"] = {
            k: sorted(list(v)) for k, v in knowledge_base["body_part_vocabulary"].items()
        }
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ LLM知识库已导出: {output_path}")
        print(f"   - Token条目数: {len(knowledge_base['token_semantics'])}")
        print(f"   - 部位词汇类别: {len(knowledge_base['body_part_vocabulary'])}")
        
        return knowledge_base
    
    def export_prompt_templates(self, output_path: str = None) -> Dict:
        """
        导出LLM Prompt模板
        
        包含:
        1. Token → 文本描述 (skeleton token解码为自然语言)
        2. 文本描述 → Token (自然语言编码为skeleton token)
        3. 序列理解 (多帧token序列理解动作)
        """
        if output_path is None:
            output_path = self.token_analysis_dir / "llm_prompt_templates.json"
        
        # 构建Token ID → 描述的快速查询表
        token_to_desc = {}
        for part_name, token_dict in self.annotations.items():
            part_zh = self.part_names_zh.get(part_name, part_name)
            for token_id, desc in token_dict.items():
                token_to_desc[int(token_id)] = {
                    "part": part_zh,
                    "desc": desc
                }
        
        templates = {
            "task_1_decode_single_frame": {
                "name": "单帧姿态解码 (Token → 文本)",
                "description": "将一帧骨架的5个部位token转换为自然语言描述",
                "system_prompt": """你是一个骨架姿态理解专家。用户会给你一组token ID,代表人体骨架的5个部位状态。
你需要将这些token ID转换为自然语言描述。

可用的Token语义知识库:
{token_knowledge_base}

注意:
- 每个token代表一个身体部位的姿态状态
- 描述要简洁准确,避免冗余
- 按照"头部躯干-左臂-右臂-左腿-右腿"顺序组织描述
""",
                "user_prompt_template": """请描述以下骨架姿态:

Token序列: [{head_spine}, {left_arm}, {right_arm}, {left_leg}, {right_leg}]

要求:
1. 先分别描述5个部位的姿态
2. 再总结整体姿态/动作
3. 输出格式:
   部位描述:
   - 头部躯干: xxx
   - 左臂: xxx
   - 右臂: xxx
   - 左腿: xxx
   - 右腿: xxx
   
   整体姿态: xxx
""",
                "example_input": {
                    "head_spine": 117,
                    "left_arm": 178,
                    "right_arm": 375,
                    "left_leg": 489,
                    "right_leg": 608
                },
                "example_output": """部位描述:
- 头部躯干: 正常姿态
- 左臂: 自然垂落
- 右臂: 自然垂落
- 左腿: 站立(直立)
- 右腿: 站立

整体姿态: 标准站立姿势,身体保持直立,双臂自然垂于身体两侧,双腿并拢支撑身体。"""
            },
            
            "task_2_encode_description": {
                "name": "文本描述编码 (文本 → Token)",
                "description": "将自然语言姿态描述转换为对应的token ID",
                "system_prompt": """你是一个骨架姿态编码专家。用户会给你一段自然语言描述,你需要选择最匹配的token ID。

可用的Token词汇表:
{body_part_vocabulary}

Token语义知识库:
{token_knowledge_base}

注意:
- 选择最接近描述的token
- 如果没有完全匹配,选择语义最相近的token
- 必须返回5个部位的token ID
""",
                "user_prompt_template": """请将以下姿态描述编码为token序列:

描述: {action_description}

要求:
1. 分析描述中各个身体部位的状态
2. 从知识库中选择最匹配的token
3. 输出格式:
   {{
     "head_spine": <token_id>,
     "left_arm": <token_id>,
     "right_arm": <token_id>,
     "left_leg": <token_id>,
     "right_leg": <token_id>,
     "confidence": <0-1之间的置信度>,
     "reasoning": "选择理由"
   }}
""",
                "example_input": "一个人站立,左手向侧面抬起,右手自然下垂,身体微微左倾",
                "example_output": {
                    "head_spine": 105,  # 左倾斜
                    "left_arm": 159,    # 左侧抬起
                    "right_arm": 375,   # 自然垂落
                    "left_leg": 489,    # 站立(直立)
                    "right_leg": 608,   # 站立
                    "confidence": 0.85,
                    "reasoning": "描述明确提到左倾、左手侧抬、右手下垂和站立,与对应token语义高度匹配"
                }
            },
            
            "task_3_sequence_understanding": {
                "name": "序列动作理解 (多帧Token → 动作语义)",
                "description": "分析连续多帧的token序列,理解整体动作意图",
                "system_prompt": """你是一个动作序列分析专家。用户会给你连续多帧的骨架token序列,你需要:
1. 理解每一帧的姿态
2. 分析帧间的变化趋势
3. 推断整体动作的语义(如"挥手"、"下蹲"、"行走"等)

可用的Token知识库:
{token_knowledge_base}

分析维度:
- 时序变化: 哪些部位在动,如何动
- 协同模式: 多个部位如何配合
- 动作周期: 是否有重复/周期性
- 意图推断: 这个动作可能在做什么
""",
                "user_prompt_template": """请分析以下骨架动作序列:

帧数: {num_frames}
Token序列:
{token_sequence}

要求:
1. 逐帧描述姿态变化
2. 识别关键动作阶段
3. 推断整体动作类型
4. 输出格式:
   {{
     "frame_analysis": [
       {{"frame": 0, "pose": "...", "key_changes": "..."}},
       ...
     ],
     "action_phases": ["准备阶段", "执行阶段", "恢复阶段"],
     "overall_action": "动作名称",
     "confidence": 0.x,
     "reasoning": "判断依据"
   }}
""",
                "example_input": {
                    "num_frames": 5,
                    "token_sequence": [
                        {"frame": 0, "tokens": [117, 178, 375, 489, 608]},  # 站立
                        {"frame": 1, "tokens": [117, 159, 375, 489, 608]},  # 左手开始抬起
                        {"frame": 2, "tokens": [117, 218, 375, 489, 608]},  # 左手完全侧抬
                        {"frame": 3, "tokens": [117, 159, 375, 489, 608]},  # 左手开始放下
                        {"frame": 4, "tokens": [117, 178, 375, 489, 608]}   # 恢复站立
                    ]
                },
                "example_output": {
                    "frame_analysis": [
                        {"frame": 0, "pose": "标准站立", "key_changes": "初始姿态"},
                        {"frame": 1, "pose": "左手开始抬起", "key_changes": "左臂从自然垂落(178)变为侧抬(159)"},
                        {"frame": 2, "pose": "左手完全侧抬", "key_changes": "左臂达到最高点(218)"},
                        {"frame": 3, "pose": "左手开始下降", "key_changes": "左臂从最高点回落(159)"},
                        {"frame": 4, "pose": "恢复站立", "key_changes": "左臂回到自然垂落(178)"}
                    ],
                    "action_phases": ["准备(F0)", "抬手(F1-F2)", "放下(F3-F4)"],
                    "overall_action": "左手挥手/招手动作",
                    "confidence": 0.92,
                    "reasoning": "左臂完成了一个完整的上抬-下降周期,其他部位保持静止,符合典型的挥手动作模式"
                }
            }
        }
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ LLM Prompt模板已导出: {output_path}")
        print(f"   - 任务模板数: {len(templates)}")
        
        return templates
    
    def generate_sample_sequences_from_dataset(self, 
                                              reconstructed_dir: str = None,
                                              num_samples: int = 10,
                                              min_frames: int = 5,
                                              max_frames: int = 10,
                                              output_path: str = None) -> List[Dict]:
        """
        从MARS重构数据中提取真实的token序列样本
        
        用于:
        1. 生成真实的多帧token序列
        2. 为LLM提供实际的动作案例
        3. 支持后续的GIF可视化标注
        
        参数:
            reconstructed_dir: 重构数据目录
            num_samples: 提取样本数量
            min_frames/max_frames: 序列帧数范围
            output_path: 输出路径
        
        返回:
            样本列表,每个样本包含token序列和元信息
        """
        if reconstructed_dir is None:
            reconstructed_dir = self.project_root / "data" / "MARS" / "reconstructed"
        else:
            reconstructed_dir = Path(reconstructed_dir)
        
        if output_path is None:
            output_path = self.token_analysis_dir / "sample_token_sequences.json"
        
        if not reconstructed_dir.exists():
            print(f"❌ 重构数据目录不存在: {reconstructed_dir}")
            print("   提示: 需要先运行 skeleton_extraction_reconstruction_saver.py")
            return []
        
        # 获取所有.npz文件
        npz_files = sorted(list(reconstructed_dir.glob("*.npz")))
        if len(npz_files) == 0:
            print(f"❌ 未找到重构数据文件 (.npz)")
            return []
        
        print(f"📂 找到 {len(npz_files)} 个重构数据文件")
        
        # 随机采样文件
        sample_indices = np.random.choice(len(npz_files), 
                                         min(num_samples, len(npz_files)), 
                                         replace=False)
        
        samples = []
        for idx in sample_indices:
            npz_file = npz_files[idx]
            try:
                data = np.load(npz_file)
                tokens = data['tokens']  # Shape: (T, 5) - T帧,5个部位
                
                # 随机选择连续帧片段
                total_frames = tokens.shape[0]
                seq_len = np.random.randint(min_frames, min(max_frames, total_frames) + 1)
                start_frame = np.random.randint(0, max(1, total_frames - seq_len + 1))
                token_seq = tokens[start_frame:start_frame + seq_len]
                
                # 构建样本
                sample = {
                    "sample_id": npz_file.stem,
                    "source_file": str(npz_file.name),
                    "frame_range": [start_frame, start_frame + seq_len - 1],
                    "num_frames": seq_len,
                    "token_sequence": [
                        {
                            "frame": i,
                            "head_spine": int(token_seq[i][0]),
                            "left_arm": int(token_seq[i][1]),
                            "right_arm": int(token_seq[i][2]),
                            "left_leg": int(token_seq[i][3]),
                            "right_leg": int(token_seq[i][4])
                        }
                        for i in range(seq_len)
                    ],
                    # 预留字段用于后续人工标注
                    "action_annotation": {
                        "overall_action": "[待标注]",
                        "body_part_actions": {
                            "head_spine": "[待标注]",
                            "left_arm": "[待标注]",
                            "right_arm": "[待标注]",
                            "left_leg": "[待标注]",
                            "right_leg": "[待标注]"
                        },
                        "action_phases": [],
                        "notes": ""
                    }
                }
                
                samples.append(sample)
                
            except Exception as e:
                print(f"⚠️  处理文件失败 {npz_file.name}: {e}")
                continue
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "samples": samples,
                "metadata": {
                    "total_samples": len(samples),
                    "source_dataset": "MARS",
                    "frame_range": [min_frames, max_frames],
                    "generation_date": self.metadata.get('last_updated', ''),
                    "annotation_status": "待标注"
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 样本序列已导出: {output_path}")
        print(f"   - 样本数: {len(samples)}")
        print(f"   - 平均帧数: {np.mean([s['num_frames'] for s in samples]):.1f}")
        
        return samples


def main():
    """主函数: 演示如何使用LLM对接工具"""
    
    print("="*60)
    print("LLM Token Annotation Exporter")
    print("="*60)
    
    # 初始化导出器
    exporter = LLMAnnotationExporter()
    
    # 1. 导出静态Token知识库
    print("\n" + "="*60)
    print("步骤1: 导出Token语义知识库")
    print("="*60)
    knowledge_base = exporter.export_llm_knowledge_base()
    
    # 显示统计
    print("\n📊 知识库统计:")
    for part, vocab in knowledge_base["body_part_vocabulary"].items():
        print(f"   - {part}: {len(vocab)} 种姿态")
    
    # 2. 导出Prompt模板
    print("\n" + "="*60)
    print("步骤2: 导出LLM Prompt模板")
    print("="*60)
    templates = exporter.export_prompt_templates()
    
    print("\n📝 可用任务模板:")
    for task_id, task_info in templates.items():
        print(f"   - {task_info['name']}")
        print(f"     {task_info['description']}")
    
    # 3. 生成样本序列(如果有重构数据)
    print("\n" + "="*60)
    print("步骤3: 提取样本Token序列 (可选)")
    print("="*60)
    samples = exporter.generate_sample_sequences_from_dataset(num_samples=10)
    
    if len(samples) > 0:
        print(f"\n✅ 已提取 {len(samples)} 个样本序列")
        print("   这些序列可用于:")
        print("   - 生成GIF动画进行人工标注")
        print("   - 训练LLM理解动作序列")
        print("   - 验证Token → 文本转换质量")
    
    # 4. 使用建议
    print("\n" + "="*60)
    print("下一步操作建议")
    print("="*60)
    print("""
1. LLM集成方式:
   方式A - API调用 (推荐):
     - 使用OpenAI API / Claude API / 文心一言 等
     - 将知识库注入System Prompt
     - 实时调用LLM进行Token ↔ 文本转换
   
   方式B - Fine-tuning:
     - 使用导出的知识库构建训练数据
     - Fine-tune小型语言模型 (如LLaMA 7B)
     - 本地部署推理

2. GIF动画标注流程:
   a) 使用 sample_token_sequences.json 中的序列
   b) 为每个序列生成骨架动画GIF
   c) 人工标注整体动作语义
   d) 回填到 action_annotation 字段
   e) 构建 Token序列 → 动作语义 的训练数据

3. 测试LLM理解能力:
   - 使用 llm_prompt_templates.json 中的example
   - 测试LLM是否能正确解码token
   - 测试LLM是否能正确编码描述
   - 测试LLM是否能理解动作序列

4. 后续改进方向:
   - 增加更多样化的动作序列样本
   - 标注动作的时间粒度 (快/慢/急促等)
   - 标注动作的情感/意图 (愤怒挥手/友好挥手等)
   - 建立动作语义的层级分类体系
""")
    
    print("\n" + "="*60)
    print("✅ 导出完成!")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"1. {exporter.token_analysis_dir}/llm_token_knowledge_base.json")
    print(f"2. {exporter.token_analysis_dir}/llm_prompt_templates.json")
    if len(samples) > 0:
        print(f"3. {exporter.token_analysis_dir}/sample_token_sequences.json")
    print(f"\n可以直接使用这些文件与LLM进行交互!")


if __name__ == "__main__":
    main()

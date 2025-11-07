#!/usr/bin/env python3
"""
码本到文本映射系统
实现从Token序列到自然语言描述的完整映射
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

class CodebookTextMapper:
    """码本到文本的映射器"""
    
    def __init__(self, mapping_file: str = "codebook_action_mappings.json"):
        self.mapping_file = mapping_file
        self.part_mappings = {}
        self.global_mappings = {}
        self.statistics = {}
        
        # 身体部位信息
        self.part_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        self.part_display_names = ['头部脊柱', '左臂', '右臂', '左腿', '右腿']
        
        # 默认动作模板（用于新映射的初始化）
        self.default_action_templates = {
            'head_spine': [
                "中性姿态", "抬头向上", "低头向下", "左转头部", "右转头部",
                "挺直脊柱", "前倾身体", "后仰身体", "左侧弯曲", "右侧弯曲"
            ],
            'left_arm': [
                "自然下垂", "上举过头", "前伸指向", "侧平举", "弯曲撑腰",
                "交叉胸前", "挥手动作", "背后伸展", "握拳准备", "放松摆动"
            ],
            'right_arm': [
                "自然下垂", "上举过头", "前伸指向", "侧平举", "弯曲撑腰",
                "交叉胸前", "挥手动作", "背后伸展", "握拳准备", "放松摆动"
            ],
            'left_leg': [
                "直立支撑", "微弯准备", "抬起前踏", "侧向迈步", "蹲姿弯曲",
                "后退准备", "踢腿动作", "站立平衡", "交叉站立", "跳跃准备"
            ],
            'right_leg': [
                "直立支撑", "微弯准备", "抬起前踏", "侧向迈步", "蹲姿弯曲", 
                "后退准备", "踢腿动作", "站立平衡", "交叉站立", "跳跃准备"
            ]
        }
        
        # 加载现有映射
        self.load_mappings()
        
    def load_mappings(self) -> bool:
        """加载码本映射表"""
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.part_mappings = data.get('part_mappings', {})
                
                # 处理全局映射的键转换
                global_mappings_raw = data.get('global_mappings', {})
                self.global_mappings = {}
                for key_str, value in global_mappings_raw.items():
                    try:
                        # 将字符串键转换回tuple
                        if key_str.startswith('(') and key_str.endswith(')'):
                            key_tuple = eval(key_str)
                            self.global_mappings[key_tuple] = value
                    except:
                        continue
                        
                self.statistics = data.get('statistics', {})
                
                print(f"✅ 成功加载码本映射表: {len(self.part_mappings)} 个部位映射")
                return True
                
            except Exception as e:
                print(f"⚠️ 加载映射表失败: {e}")
                
        print("📝 创建新的映射表...")
        self._initialize_default_mappings()
        return False
        
    def _initialize_default_mappings(self):
        """初始化默认映射表"""
        for part_name in self.part_names:
            self.part_mappings[part_name] = {}
            
        # 可以预设一些常见的映射
        self._create_sample_mappings()
        
    def _create_sample_mappings(self):
        """创建示例映射（用于演示）"""
        sample_mappings = {
            'head_spine': {
                15: {'semantic': '头部中性', 'confidence': 0.95, 'frequency': 45},
                28: {'semantic': '抬头向上', 'confidence': 0.92, 'frequency': 32},
                45: {'semantic': '低头向下', 'confidence': 0.88, 'frequency': 28}
            },
            'left_arm': {
                32: {'semantic': '自然下垂', 'confidence': 0.94, 'frequency': 55},
                58: {'semantic': '上举过头', 'confidence': 0.97, 'frequency': 38},
                76: {'semantic': '前伸指向', 'confidence': 0.89, 'frequency': 25}
            },
            'right_arm': {
                41: {'semantic': '自然下垂', 'confidence': 0.95, 'frequency': 52},
                65: {'semantic': '上举过头', 'confidence': 0.96, 'frequency': 36},
                119: {'semantic': '挥手动作', 'confidence': 0.88, 'frequency': 22}
            },
            'left_leg': {
                18: {'semantic': '直立支撑', 'confidence': 0.98, 'frequency': 68},
                72: {'semantic': '抬起前踏', 'confidence': 0.89, 'frequency': 31},
                113: {'semantic': '蹲姿弯曲', 'confidence': 0.82, 'frequency': 18}
            },
            'right_leg': {
                23: {'semantic': '直立支撑', 'confidence': 0.97, 'frequency': 65},
                78: {'semantic': '抬起前踏', 'confidence': 0.88, 'frequency': 29},
                126: {'semantic': '蹲姿弯曲', 'confidence': 0.81, 'frequency': 16}
            }
        }
        
        for part, mappings in sample_mappings.items():
            self.part_mappings[part].update(mappings)
            
        # 示例全局映射
        self.global_mappings = {
            (28, 58, 65, 18, 23): {
                'action': '双手举高庆祝动作',
                'confidence': 0.94,
                'category': 'celebration',
                'frequency': 15
            },
            (15, 76, 119, 72, 23): {
                'action': '指向并挥手问候',
                'confidence': 0.91,
                'category': 'greeting', 
                'frequency': 12
            }
        }
        
    def save_mappings(self):
        """保存映射表到文件"""
        mapping_data = {
            'part_mappings': self.part_mappings,
            'global_mappings': {str(k): v for k, v in self.global_mappings.items()},
            'statistics': self.statistics,
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_parts': len(self.part_names),
                'version': '1.0'
            }
        }
        
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 映射表已保存到: {self.mapping_file}")
        
    def map_tokens_to_text(self, token_sequence: List[int]) -> Dict:
        """将Token序列映射为文本描述"""
        if len(token_sequence) != 5:
            return {
                "error": "Token序列长度必须为5",
                "token_sequence": token_sequence
            }
            
        result = {
            'token_sequence': token_sequence,
            'part_descriptions': {},
            'detailed_descriptions': {},
            'overall_action': None,
            'confidence_scores': {},
            'natural_language': '',
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. 解析各部位描述
        total_confidence = 0
        valid_parts = 0
        
        for i, (part_name, display_name) in enumerate(zip(self.part_names, self.part_display_names)):
            token_id = token_sequence[i]
            
            if str(token_id) in self.part_mappings[part_name]:
                mapping = self.part_mappings[part_name][str(token_id)]
                semantic = mapping['semantic']
                confidence = mapping['confidence']
                
                result['part_descriptions'][display_name] = semantic
                result['confidence_scores'][display_name] = confidence
                result['detailed_descriptions'][display_name] = {
                    'token_id': token_id,
                    'action': semantic,
                    'confidence': confidence,
                    'reliability': self._get_reliability_level(confidence),
                    'frequency': mapping.get('frequency', 0)
                }
                
                total_confidence += confidence
                valid_parts += 1
                
            else:
                # 未知Token的处理
                result['part_descriptions'][display_name] = f'未识别动作 (Token: {token_id})'
                result['confidence_scores'][display_name] = 0.0
                result['detailed_descriptions'][display_name] = {
                    'token_id': token_id,
                    'action': '未识别动作',
                    'confidence': 0.0,
                    'reliability': '低',
                    'frequency': 0
                }
                
        # 2. 查找全局动作匹配
        token_tuple = tuple(token_sequence)
        if token_tuple in self.global_mappings:
            global_mapping = self.global_mappings[token_tuple]
            result['overall_action'] = {
                'name': global_mapping['action'],
                'confidence': global_mapping['confidence'],
                'category': global_mapping['category'],
                'type': 'exact_match',
                'frequency': global_mapping.get('frequency', 0)
            }
        else:
            # 生成组合动作描述
            result['overall_action'] = self._generate_composite_action(result['part_descriptions'])
            
        # 3. 计算整体置信度
        avg_confidence = total_confidence / valid_parts if valid_parts > 0 else 0.0
        result['average_confidence'] = avg_confidence
        
        # 4. 生成自然语言描述
        result['natural_language'] = self._generate_natural_language(result)
        
        return result
        
    def _get_reliability_level(self, confidence: float) -> str:
        """根据置信度返回可靠性等级"""
        if confidence >= 0.9:
            return '高'
        elif confidence >= 0.8:
            return '中'
        elif confidence >= 0.6:
            return '中低'
        else:
            return '低'
            
    def _generate_composite_action(self, part_descriptions: Dict) -> Dict:
        """基于局部描述生成组合动作"""
        descriptions = list(part_descriptions.values())
        description_text = ' '.join(descriptions)
        
        # 检测动作模式
        if '上举' in description_text and description_text.count('上举') >= 2:
            return {
                'name': '双臂上举动作',
                'confidence': 0.85,
                'category': 'arm_movement',
                'type': 'pattern_match'
            }
        elif '挥手' in description_text:
            return {
                'name': '挥手问候动作',
                'confidence': 0.82,
                'category': 'greeting',
                'type': 'pattern_match'
            }
        elif '蹲' in description_text and description_text.count('蹲') >= 2:
            return {
                'name': '蹲姿相关动作',
                'confidence': 0.79,
                'category': 'posture_change',
                'type': 'pattern_match'
            }
        elif '迈步' in description_text or '前踏' in description_text:
            return {
                'name': '步行移动动作',
                'confidence': 0.77,
                'category': 'locomotion',
                'type': 'pattern_match'
            }
        elif '低头' in description_text and ('下垂' in description_text or '弯曲' in description_text):
            return {
                'name': '检查观察动作',
                'confidence': 0.74,
                'category': 'examination',
                'type': 'pattern_match'
            }
        else:
            return {
                'name': '复合动作组合',
                'confidence': 0.70,
                'category': 'complex',
                'type': 'composite'
            }
            
    def _generate_natural_language(self, result: Dict) -> str:
        """生成自然语言描述"""
        parts = result['part_descriptions']
        overall = result['overall_action']
        avg_confidence = result.get('average_confidence', 0.0)
        
        # 构建有效的部位描述
        valid_part_texts = []
        for part_name, action in parts.items():
            if '未识别' not in action:
                valid_part_texts.append(f"{part_name}呈现{action}")
                
        # 构建完整描述
        if overall and overall['confidence'] > 0.85:
            # 高置信度的整体动作
            description = f"识别为【{overall['name']}】"
            if valid_part_texts:
                description += f"，具体表现为：{', '.join(valid_part_texts)}"
        else:
            # 基于局部描述的组合
            if valid_part_texts:
                description = f"检测到动作组合：{', '.join(valid_part_texts)}"
                if overall:
                    description += f"，整体判断为{overall['name']}"
            else:
                description = "动作识别置信度较低，建议人工确认"
                
        # 添加置信度信息
        if avg_confidence > 0:
            confidence_text = f"(平均置信度: {avg_confidence:.2f})"
            description += f" {confidence_text}"
            
        return description
        
    def add_token_mapping(self, part_name: str, token_id: int, semantic: str, 
                         confidence: float = 0.8, frequency: int = 1):
        """添加新的Token映射"""
        if part_name not in self.part_names:
            raise ValueError(f"无效的部位名称: {part_name}")
            
        if part_name not in self.part_mappings:
            self.part_mappings[part_name] = {}
            
        self.part_mappings[part_name][str(token_id)] = {
            'semantic': semantic,
            'confidence': confidence,
            'frequency': frequency
        }
        
        print(f"✅ 添加映射: {part_name} Token {token_id} -> {semantic}")
        
    def add_global_mapping(self, token_sequence: List[int], action_name: str,
                          confidence: float = 0.8, category: str = 'custom'):
        """添加新的全局动作映射"""
        if len(token_sequence) != 5:
            raise ValueError("Token序列长度必须为5")
            
        token_tuple = tuple(token_sequence)
        self.global_mappings[token_tuple] = {
            'action': action_name,
            'confidence': confidence,
            'category': category,
            'frequency': 1
        }
        
        print(f"✅ 添加全局映射: {token_sequence} -> {action_name}")
        
    def get_mapping_statistics(self) -> Dict:
        """获取映射统计信息"""
        stats = {
            'part_coverage': {},
            'total_mappings': 0,
            'global_mappings_count': len(self.global_mappings),
            'average_confidence_by_part': {}
        }
        
        for part_name in self.part_names:
            if part_name in self.part_mappings:
                mappings = self.part_mappings[part_name]
                stats['part_coverage'][part_name] = len(mappings)
                stats['total_mappings'] += len(mappings)
                
                # 计算平均置信度
                if mappings:
                    confidences = [m['confidence'] for m in mappings.values()]
                    stats['average_confidence_by_part'][part_name] = sum(confidences) / len(confidences)
                else:
                    stats['average_confidence_by_part'][part_name] = 0.0
            else:
                stats['part_coverage'][part_name] = 0
                stats['average_confidence_by_part'][part_name] = 0.0
                
        return stats
        
    def export_mapping_report(self, output_file: str = "mapping_report.txt"):
        """导出映射报告"""
        stats = self.get_mapping_statistics()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("码本映射系统报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("映射覆盖情况:\n")
            for part_name, count in stats['part_coverage'].items():
                display_name = dict(zip(self.part_names, self.part_display_names))[part_name]
                confidence = stats['average_confidence_by_part'][part_name]
                f.write(f"  {display_name}: {count} 个映射 (平均置信度: {confidence:.3f})\n")
                
            f.write(f"\n总计局部映射: {stats['total_mappings']} 个\n")
            f.write(f"全局动作映射: {stats['global_mappings_count']} 个\n\n")
            
            # 详细映射列表
            f.write("详细映射列表:\n")
            f.write("-" * 30 + "\n")
            
            for part_name, display_name in zip(self.part_names, self.part_display_names):
                f.write(f"\n{display_name}:\n")
                if part_name in self.part_mappings:
                    for token_id, mapping in sorted(self.part_mappings[part_name].items()):
                        f.write(f"  Token {token_id}: {mapping['semantic']} "
                               f"(置信度: {mapping['confidence']:.2f}, "
                               f"频次: {mapping.get('frequency', 0)})\n")
                else:
                    f.write("  暂无映射\n")
                    
        print(f"📊 映射报告已导出到: {output_file}")

if __name__ == "__main__":
    # 测试映射器
    mapper = CodebookTextMapper()
    
    # 测试Token映射
    test_tokens = [28, 58, 65, 18, 23]
    result = mapper.map_tokens_to_text(test_tokens)
    
    print("🧪 测试码本映射:")
    print(f"Token序列: {test_tokens}")
    print(f"自然语言: {result['natural_language']}")
    
    # 保存映射
    mapper.save_mappings()
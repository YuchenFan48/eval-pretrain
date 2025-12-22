"""
数据加载器：负责加载各个benchmark的数据
"""

import json
import csv
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式的数据"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_csv(path: str) -> List[List[str]]:
    """加载CSV格式的数据"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def load_benchmark_data(benchmark_name: str, benchmark_config: dict) -> List[Dict[str, Any]]:
    """
    加载指定benchmark的数据
    
    Args:
        benchmark_name: benchmark名称
        benchmark_config: benchmark配置信息
        
    Returns:
        数据列表
    """
    path = benchmark_config['path']
    format_type = benchmark_config['format']
    
    if format_type == 'jsonl':
        return load_jsonl(path)
    elif format_type == 'csv':
        raw_data = load_csv(path)
        return parse_csv_data(benchmark_name, raw_data)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def parse_csv_data(benchmark_name: str, raw_data: List[List[str]]) -> List[Dict[str, Any]]:
    """
    将CSV原始数据转换为统一的字典格式
    
    Args:
        benchmark_name: benchmark名称
        raw_data: CSV原始数据
        
    Returns:
        解析后的数据列表
    """
    if benchmark_name == 'gpqa':
        return parse_gpqa_csv(raw_data)
    elif benchmark_name == 'mmlu-redux':
        return parse_mmlu_redux_csv(raw_data)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def parse_gpqa_csv(raw_data: List[List[str]]) -> List[Dict[str, Any]]:
    """
    解析GPQA的CSV数据
    CSV格式：第8列是question，9-12列是选项（第一个是正确答案）
    """
    data = []
    cnt = 0
    for row in raw_data:
        if row[7] == 'Question':  # 跳过表头
            continue
        cnt += 1
        question = row[7]
        options = [row[8], row[9], row[10], row[11]]
        
        # 打乱选项顺序（参考opencompass的实现）
        shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']
        pattern = shuffle_patterns[cnt % 4]
        
        item = {'question': question}
        ground_truth = options[0]  # 第一个选项是正确答案
        
        # 根据打乱模式重新排列选项
        for i in range(4):
            item['ABCD'[i]] = options[ord(pattern[i]) - ord('A')]
        
        # 找到正确答案的位置
        for i in range(4):
            if item['ABCD'[i]] == ground_truth:
                item['answer'] = 'ABCD'[i]
                break
        
        data.append(item)
    
    return data


def parse_mmlu_redux_csv(raw_data: List[List[str]]) -> List[Dict[str, Any]]:
    """
    解析MMLU-Redux的CSV数据
    CSV格式：question, choices, answer
    """
    data = []
    
    for i, row in enumerate(raw_data):
        if i == 0:  # 跳过表头
            continue
        if len(row) < 3:
            continue
        
        # 解析choices（可能是列表格式的字符串）
        choices_str = row[1]
        try:
            if choices_str.startswith('['):
                choices = eval(choices_str)
            else:
                choices = [choices_str]
        except:
            choices = [choices_str]
        
        # 解析answer（可能是数字索引）
        answer_str = row[2]
        if answer_str.isdigit():
            answer_idx = int(answer_str)
            # 转换为字母
            answer = chr(65 + answer_idx) if answer_idx < 26 else answer_str
        else:
            answer = answer_str
        
        item = {
            'question': row[0],
            'choices': choices,
            'answer': answer,
        }
        data.append(item)
    
    return data


# 示例使用
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from config import BENCHMARK_CONFIG
    
    # 测试加载bbh
    data = load_benchmark_data('bbh', BENCHMARK_CONFIG['bbh'])
    print(f"BBH数据量: {len(data)}")
    print(f"示例: {data[0]}")
    
    # 测试加载gpqa
    data = load_benchmark_data('gpqa', BENCHMARK_CONFIG['gpqa'])
    print(f"GPQA数据量: {len(data)}")
    print(f"示例: {data[0]}")


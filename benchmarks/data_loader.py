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


def load_json(path: str) -> List[Dict[str, Any]]:
    """加载JSON格式的数据（列表）"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root is not a list.")
    return data


def load_parquet(path: str) -> List[Dict[str, Any]]:
    """加载Parquet格式的数据"""
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(path)
        return table_to_rows(table)
    except ImportError:
        try:
            import pandas as pd  # type: ignore
            return pd.read_parquet(path).to_dict(orient='records')
        except ImportError as exc:
            raise ImportError(
                "缺少读取parquet的依赖，请安装 pyarrow 或 pandas。"
            ) from exc


def table_to_rows(table) -> List[Dict[str, Any]]:
    """将pyarrow Table转换为行列表（兼容旧版本pyarrow）"""
    columns = table.column_names
    data = table.to_pydict()
    rows = []
    for i in range(table.num_rows):
        row = {col: data[col][i] for col in columns}
        rows.append(row)
    return rows


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
    elif format_type == 'json':
        raw_data = load_json(path)
        return parse_json_data(benchmark_name, raw_data)
    elif format_type == 'parquet':
        raw_data = load_parquet(path)
        return parse_parquet_data(benchmark_name, raw_data)
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
    elif benchmark_name == 'simpleqa':
        return parse_simpleqa_csv(raw_data)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def parse_json_data(benchmark_name: str, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将JSON原始数据转换为统一的字典格式
    """
    if benchmark_name == 'nq':
        return parse_nq_json(raw_data)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def parse_parquet_data(benchmark_name: str, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将Parquet原始数据转换为统一的字典格式
    """
    if benchmark_name == 'arc-c':
        return parse_arc_c_parquet(raw_data)
    elif benchmark_name == 'arc-e':
        return parse_arc_c_parquet(raw_data)
    elif benchmark_name == 'hellaswag':
        return parse_hellaswag_parquet(raw_data)
    elif benchmark_name == 'winogrande':
        return parse_winogrande_parquet(raw_data)
    elif benchmark_name == 'piqa':
        return parse_piqa_parquet(raw_data)
    elif benchmark_name == 'drop':
        return parse_drop_parquet(raw_data)
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


def parse_simpleqa_csv(raw_data: List[List[str]]) -> List[Dict[str, Any]]:
    """
    解析SimpleQA的CSV数据
    CSV格式：metadata, problem, answer
    """
    data = []
    for i, row in enumerate(raw_data):
        if i == 0:  # 跳过表头
            continue
        if len(row) < 3:
            continue
        item = {
            'metadata': row[0],
            'question': row[1],
            'answer': row[2],
        }
        data.append(item)
    return data


def parse_arc_c_parquet(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析ARC-C的Parquet数据
    字段: id, question, choices{text,label}, answerKey
    """
    data = []
    
    for row in raw_data:
        choice_map = {}
        choices = row.get('choices', {})
        labels = choices.get('label', []) if isinstance(choices, dict) else []
        texts = choices.get('text', []) if isinstance(choices, dict) else []
        
        for label, text in zip(labels, texts):
            mapped_label = _normalize_choice_label(label)
            choice_map[mapped_label] = text
        
        ordered_choices = [choice_map.get(label, '') for label in ['A', 'B', 'C', 'D']]
        answer = _normalize_choice_label(row.get('answerKey', ''))
        
        data.append({
            'id': row.get('id', ''),
            'question': row.get('question', ''),
            'choices': ordered_choices,
            'answer': answer,
        })
    
    return data


def parse_hellaswag_parquet(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析HellaSwag的Parquet数据
    字段: ctx_a, ctx_b, ctx, endings, label
    """
    data = []
    
    for row in raw_data:
        context = str(row.get('ctx', '') or '').strip()
        if not context:
            ctx_a = str(row.get('ctx_a', '') or '').strip()
            ctx_b = str(row.get('ctx_b', '') or '').strip()
            context = f"{ctx_a} {ctx_b}".strip()
        
        label = row.get('label', '')
        answer = _label_to_abcd(label)
        
        data.append({
            'ind': row.get('ind', ''),
            'context': context,
            'endings': row.get('endings', []),
            'answer': answer,
        })
    
    return data


def parse_winogrande_parquet(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析WinoGrande的Parquet数据
    字段: sentence, option1, option2, answer (1/2)
    """
    data = []
    
    for row in raw_data:
        answer = row.get('answer', '')
        answer_str = str(answer).strip()
        answer_letter = 'A' if answer_str == '1' else 'B' if answer_str == '2' else ''
        
        data.append({
            'sentence': row.get('sentence', ''),
            'option1': row.get('option1', ''),
            'option2': row.get('option2', ''),
            'answer': answer_letter,
        })
    
    return data


def _normalize_choice_label(label: Any) -> str:
    label_str = str(label).strip()
    if label_str.isdigit():
        idx = int(label_str) - 1
        if 0 <= idx < 4:
            return 'ABCD'[idx]
    return label_str.upper()


def _label_to_abcd(label: Any) -> str:
    label_str = str(label).strip()
    if label_str.isdigit():
        idx = int(label_str)
        if 0 <= idx < 4:
            return 'ABCD'[idx]
    if label_str in ['A', 'B', 'C', 'D']:
        return label_str
    return label_str.upper()


def parse_piqa_parquet(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析PIQA的Parquet数据
    字段: goal, sol1, sol2, label (0/1)
    """
    data = []
    
    for row in raw_data:
        label = row.get('label', '')
        label_str = str(label).strip()
        answer = 'A' if label_str == '0' else 'B' if label_str == '1' else ''
        
        data.append({
            'goal': row.get('goal', ''),
            'sol1': row.get('sol1', ''),
            'sol2': row.get('sol2', ''),
            'answer': answer,
        })
    
    return data


def parse_drop_parquet(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析DROP的Parquet数据
    字段: passage, question, answers/answers_spans
    """
    data = []
    
    for row in raw_data:
        answers = collect_drop_answers(row)
        data.append({
            'passage': row.get('passage', ''),
            'question': row.get('question', ''),
            'answers': answers,
        })
    
    return data


def collect_drop_answers(example: Dict[str, Any]) -> List[str]:
    answers: List[str] = []
    spans = example.get('answers_spans', None)
    if isinstance(spans, dict):
        spans = spans.get('spans', [])
    if isinstance(spans, list):
        answers.extend([str(s) for s in spans if s])
    
    ans = example.get('answers', None)
    if isinstance(ans, dict):
        spans_list = ans.get('spans', [])
        if isinstance(spans_list, list):
            answers.extend([str(s) for s in spans_list if s])
        text_list = ans.get('text', [])
        if isinstance(text_list, list):
            answers.extend([str(s) for s in text_list if s])
        number = ans.get('number', '')
        if number:
            answers.append(str(number))
        date = ans.get('date', {})
        date_text = _format_drop_date(date)
        if date_text:
            answers.append(date_text)
    elif isinstance(ans, list):
        answers.extend([str(a) for a in ans if a])
    elif isinstance(ans, str):
        answers.append(ans)
    
    answer = example.get('answer', None)
    if isinstance(answer, dict):
        spans_list = answer.get('spans', [])
        if isinstance(spans_list, list):
            answers.extend([str(s) for s in spans_list if s])
        number = answer.get('number', '')
        if number:
            answers.append(str(number))
        date_text = _format_drop_date(answer.get('date', {}))
        if date_text:
            answers.append(date_text)
    
    validated = example.get('validated_answers', None)
    if isinstance(validated, dict):
        numbers = validated.get('number', [])
        if isinstance(numbers, list):
            answers.extend([str(n) for n in numbers if n])
        dates = validated.get('date', [])
        if isinstance(dates, list):
            for d in dates:
                date_text = _format_drop_date(d)
                if date_text:
                    answers.append(date_text)
        spans = validated.get('spans', [])
        if isinstance(spans, list):
            for span_group in spans:
                if isinstance(span_group, list):
                    answers.extend([str(s) for s in span_group if s])

    deduped = []
    seen = set()
    for a in answers:
        if a not in seen:
            deduped.append(a)
            seen.add(a)
    return deduped


def _format_drop_date(date_obj: Any) -> str:
    if not isinstance(date_obj, dict):
        return ""
    date_parts = [date_obj.get('day', ''), date_obj.get('month', ''), date_obj.get('year', '')]
    date_text = " ".join([str(p).strip() for p in date_parts if str(p).strip()])
    return date_text


def parse_nq_json(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解析Natural Questions的JSON数据
    字段: questions, answers, contexts
    """
    data = []
    
    for row in raw_data:
        questions = row.get('questions', [])
        question = ''
        if questions and isinstance(questions, list):
            question = questions[0].get('input_text', '')
        answers = row.get('answers', [])
        answer_texts = []
        for ans in answers:
            text = ans.get('span_text', '')
            if text:
                answer_texts.append(text)
        data.append({
            'id': row.get('id', ''),
            'question': question,
            'context': row.get('contexts', ''),
            'answers': answer_texts,
        })
    
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

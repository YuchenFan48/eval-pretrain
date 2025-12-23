"""
评估器：负责评估各个benchmark的结果
根据 prompt 格式优化的答案提取函数
"""

import re
from typing import List, Dict, Any, Tuple


def extract_answer_bbh(response: str) -> str:
    """
    从BBH的回答中提取答案
    Prompt格式: "Q: ... A: Let's think step by step. ... So the answer is X."
    """
    # 移除多余的换行和空格
    ans = response.strip()
    
    # 尝试匹配 "So the answer is X" 或 "the answer is X"
    patterns = [
        r'[Ss]o the answer is[:\s]+([^\n\.]+)',
        r'[Tt]he answer is[:\s]+([^\n\.]+)',
        r'answer is[:\s]+([^\n\.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, ans)
        if match:
            ans = match.group(1).strip()
            break
    else:
        # 如果没有匹配到，取第一行
        ans = ans.split('\n')[0].strip()
    
    # 移除末尾的句号
    if ans.endswith('.'):
        ans = ans[:-1].strip()
    
    # 尝试匹配加粗的答案 **X**
    match = re.search(r'\*\*(.*?)\*\*', ans)
    if match:
        return match.group(1).strip()
    
    return ans.strip()


def extract_answer_mmlu_pro(response: str) -> str:
    """
    从MMLU-Pro的回答中提取答案选项
    Prompt格式: "Answer: Let's think step by step. ... The answer is (X)."
    """
    # 尝试多种模式匹配
    patterns = [
        r'[Tt]he answer is[:\s]*\(?([A-P])\)?',  # "The answer is (A)" 或 "The answer is A"
        r'answer is[:\s]*\(?([A-P])\)?',          # "answer is (A)"
        r'\(([A-P])\)',                            # "(A)"
        r'^([A-P])[\.:\s]',                        # "A." 或 "A:"
        r'[Cc]orrect answer[:\s]+\(?([A-P])\)?',  # "correct answer is A"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果没有匹配到，尝试找到最后一个出现的选项字母
    matches = re.findall(r'\b([A-P])\b', response.upper())
    if matches:
        return matches[0]
    
    return ""


def extract_answer_mmlu(response: str) -> str:
    """
    从MMLU的回答中提取答案选项
    Prompt格式（新）: "Answer: Let's think step by step. ... The answer is X."
    """
    response = response.strip()
    
    # 尝试多种模式（CoT 格式优先）
    patterns = [
        r'[Tt]he answer is[:\s]+\(?([A-D])\)?',  # "The answer is A" 或 "(A)"
        r'answer is[:\s]+\(?([A-D])\)?',          # "answer is A"
        r'^([A-D])[\.:\s]',                        # "A." 或 "A:" 在开头
        r'\(([A-D])\)',                            # "(A)"
        r'[Aa]nswer[:\s]+([A-D])',                 # "Answer: A"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 找最后一个出现的 A-D（CoT 可能在中间提到其他选项）
    matches = re.findall(r'\b([A-D])\b', response.upper())
    if matches:
        return matches[0]
    
    return ""


def extract_answer_gpqa(response: str) -> str:
    """
    从GPQA的回答中提取答案
    Prompt格式: 'Format your response as follows: "The correct answer is (insert answer here)"'
    """
    patterns = [
        r'[Tt]he correct answer is[:\s]*\(([A-D])\)',  # "The correct answer is (A)"
        r'[Tt]he correct answer is[:\s]*([A-D])',       # "The correct answer is A"
        r'correct answer[:\s]*\(?([A-D])\)?',           # "correct answer is A"
        r'\(([A-D])\)',                                  # "(A)"
        r'^([A-D])[\.:\s]',                              # "A." 在开头
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 找第一个出现的 A-D
    for char in response.upper():
        if char in 'ABCD':
            return char
    
    return ""


def extract_answer_gsm8k(response: str) -> str:
    """
    从GSM8K回答中提取答案
    Prompt格式: "Answer: ... The answer is X"
    """
    # 移除 "Question:" 之后的内容（避免提取到后续问题的数字）
    response = response.split('Question:')[0]
    
    # 尝试匹配 "The answer is X"
    patterns = [
        r'[Tt]he answer is[:\s]+(-?\d+(?:,?\d+)*(?:\.\d+)?)',
        r'answer is[:\s]+(-?\d+(?:,?\d+)*(?:\.\d+)?)',
        r'####\s*(-?\d+(?:,?\d+)*(?:\.\d+)?)',  # GSM8K 标准格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            # 移除逗号
            return match.group(1).replace(',', '')
    
    # 提取最后一个数字
    numbers = re.findall(r'(-?\d+(?:,?\d+)*(?:\.\d+)?)', response)
    if numbers:
        return numbers[0].replace(',', '')
    
    return ""


def extract_answer_math(response: str) -> str:
    """
    从MATH回答中提取答案
    Prompt格式: "Solution: ... $\\boxed{X}$ ... Final Answer: The final answer is $X$."
    """
    # 优先级1: 提取 \\boxed{} 中的内容（处理嵌套的花括号）
    match = re.search(r'\\boxed\{(.+?)\}(?:[^\}]|$)', response)
    if match:
        ans = match.group(1).strip()
        # 如果内容包含 \frac 等命令，继续查找完整的表达式
        if '\\frac' in ans and ans.count('{') != ans.count('}'):
            # 手动匹配嵌套的花括号
            start_idx = response.find('\\boxed{')
            if start_idx != -1:
                start_idx += len('\\boxed{')
                brace_count = 1
                end_idx = start_idx
                while end_idx < len(response) and brace_count > 0:
                    if response[end_idx] == '{':
                        brace_count += 1
                    elif response[end_idx] == '}':
                        brace_count -= 1
                    end_idx += 1
                if brace_count == 0:
                    return response[start_idx:end_idx-1].strip()
        return ans
    
    # 优先级2: 提取 "Final Answer: The final answer is X"
    patterns = [
        r'[Ff]inal [Aa]nswer[:\s]+[Tt]he final answer is[:\s]+\$?([^\$\.\n]+)\$?',
        r'[Ff]inal [Aa]nswer[:\s]+\$?([^\$\.\n]+)\$?',
        r'[Tt]he final answer is[:\s]+\$?([^\$\.\n]+)\$?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            ans = match.group(1).strip()
            # 移除 "I hope it is correct"
            ans = ans.replace('I hope it is correct', '').strip()
            if ans:
                return ans
    
    # 优先级3: 提取最后一个在 $ $ 之间的内容
    dollar_matches = re.findall(r'\$([^\$]+)\$', response)
    if dollar_matches:
        return dollar_matches[0].strip()
    
    # 优先级4: 提取最后一个数字或表达式
    # 尝试匹配分数、小数、整数等
    number_patterns = [
        r'(-?\d+/\d+)',           # 分数
        r'(-?\d+\.\d+)',          # 小数
        r'(-?\d+)',               # 整数
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[0]
    
    return ""

def extract_answer_multiple_choice(response: str, options: str = 'ABCD') -> str:
    """
    强化版多选题答案提取，解决 CoT 场景下的稳定性问题
    """
    if not response:
        return ""
    
    # 2. 优先级模式匹配 (从最强特征到弱特征)
    patterns = [
        # 匹配 **A** 或 (A) 或 【A】
        rf'\*\*?\(?([{options}])\)?\*\*?', 
        rf'\[([{options}])\]',
        # 匹配 "The answer is A" 或 "Answer: A"
        rf'[Tt]he answer is[:\s]+\(?([{options}])\)?',
        rf'[Aa]nswer is[:\s]+\(?([{options}])\)?',
        rf'[Aa]nswer[:\s]+\(?([{options}])\)?',
        # 匹配 "Choice A"
        rf'[Cc]hoice\s+([{options}])',
    ]

    for pattern in patterns:
        # 在整个文本中找最后一个匹配项（通常是最终结论）
        matches = list(re.finditer(pattern, response))
        if matches:
            return matches[0].group(1).upper()

    # 3. 兜底策略 1: 查找最后一行出现的孤立字母
    last_line = response.strip().split('\n')[0]
    match = re.search(rf'\b([{options}])\b', last_line)
    if match:
        return match.group(1).upper()

    # 4. 兜底策略 2: 查找全文中最后一个出现的选项字母
    # 注意：这里要加上 \b 边界触发，防止匹配到单词内部的字母
    matches = re.findall(rf'\b([{options}])\b', response)
    if matches:
        return matches[0].upper()

    return ""


def extract_answer_nq(response: str) -> str:
    """
    从NQ回答中提取答案文本
    """
    if not response:
        return ""
    response = response.strip()
    patterns = [
        r'[Tt]he answer is[:\s]+([^\n\.]+)',
        r'[Aa]nswer[:\s]+([^\n\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
    last_line = response.split('\n')[-1].strip()
    return last_line if last_line else response


def extract_answer_drop(response: str) -> str:
    """
    从DROP回答中提取答案文本
    """
    if not response:
        return ""
    response = response.strip()
    patterns = [
        r'[Tt]he answer is[:\s]+([^\n\.]+)',
        r'[Aa]nswer[:\s]+([^\n\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
    last_line = response.split('\n')[-1].strip()
    return last_line if last_line else response

def extract_answer_mmlu_redux(response: str) -> str:
    """
    专门为 MMLU-Redux 优化的提取器
    """
    # MMLU-Redux 通常是 ABCD，但为了兼容性保留
    return extract_answer_multiple_choice(response, 'ABCD')


def normalize_choice_ref(ref: Any, options: str = 'ABCD', prefer_one_based: bool = False) -> str:
    """
    标准化多选题参考答案（支持字母与数字索引）
    """
    if isinstance(ref, int):
        if prefer_one_based and 1 <= ref <= len(options):
            return options[ref - 1]
        if 0 <= ref < len(options):
            return options[ref]
        if 1 <= ref <= len(options):
            return options[ref - 1]
        return str(ref).upper()

    ref_str = str(ref).strip()
    if ref_str.isdigit():
        idx = int(ref_str)
        if prefer_one_based and 1 <= idx <= len(options):
            return options[idx - 1]
        if 0 <= idx < len(options):
            return options[idx]
        if 1 <= idx <= len(options):
            return options[idx - 1]
        return ref_str.upper()

    if len(ref_str) == 1:
        return ref_str.upper()

    return ref_str.upper()


def normalize_math_answer(answer: str) -> str:
    """
    标准化数学答案
    移除LaTeX格式、空格、单位等
    """
    if not answer:
        return ""
    
    answer = str(answer).strip()
    
    # 移除常见的LaTeX命令
    latex_removals = ['\\$', '$', '\\text', '\\mathrm', '\\mathbf', '\\', ',']
    for removal in latex_removals:
        answer = answer.replace(removal, '')
    
    # 移除花括号
    answer = answer.replace('{', '').replace('}', '')
    
    # 移除空格
    answer = answer.replace(' ', '')
    
    # 移除常见单位
    units = ['dollars', 'dollar', 'mph', 'inches', 'inch', 'ft', 'feet', 
             'cm', 'meters', 'meter', 'km', 'pounds', 'pound', 'kg', 'grams', 'gram']
    answer_lower = answer.lower()
    for unit in units:
        answer_lower = answer_lower.replace(unit, '')
    
    return answer_lower.strip()


def normalize_nq_answer(text: str) -> str:
    """
    NQ标准化：小写、去标点、去冠词、去多余空格
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_answer_text(response: str) -> str:
    """
    通用文本答案提取器
    """
    if not response:
        return ""
    response = response.strip()
    patterns = [
        r'[Tt]he answer is[:\s]+([^\n\.]+)',
        r'[Aa]nswer[:\s]+([^\n\.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
    last_line = response.split('\n')[-1].strip()
    return last_line if last_line else response


def _get_arc_answer_text(ref: Any, item: Dict[str, Any]) -> str:
    """
    将ARC参考答案映射为选项文本
    """
    question = item.get('question', '')
    if isinstance(question, dict):
        choices = question.get('choices', item.get('choices', []))
    else:
        choices = item.get('choices', item.get('options', []))
    
    if choices and isinstance(choices[0], dict):
        label_to_text = {str(c.get('label', '')).strip().upper(): c.get('text', '') for c in choices}
        ref_label = normalize_choice_ref(ref, 'ABCDE', prefer_one_based=True)
        return label_to_text.get(ref_label, '')
    
    if isinstance(choices, list):
        ref_label = normalize_choice_ref(ref, 'ABCDE', prefer_one_based=True)
        if len(ref_label) == 1 and ref_label in 'ABCDE':
            idx = ord(ref_label) - ord('A')
            if 0 <= idx < len(choices):
                return str(choices[idx])
    
    return str(ref)


def _get_gpqa_answer_text(ref: Any, item: Dict[str, Any]) -> str:
    """
    将GPQA参考答案映射为选项文本
    """
    label = normalize_choice_ref(ref, 'ABCD', prefer_one_based=False)
    if label == 'A':
        return item.get('A', '')
    if label == 'B':
        return item.get('B', '')
    if label == 'C':
        return item.get('C', '')
    if label == 'D':
        return item.get('D', '')
    return str(ref)


def _get_choices_answer_text(ref: Any, choices: List[Any]) -> str:
    if not choices:
        return str(ref)
    options = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]
    ref_label = normalize_choice_ref(ref, options, prefer_one_based=True)
    if len(ref_label) == 1 and ref_label in options:
        idx = ord(ref_label) - ord('A')
        if 0 <= idx < len(choices):
            return str(choices[idx])
    return str(ref)


def normalize_drop_answer(text: str) -> str:
    """
    DROP标准化：小写、去标点、去冠词、去多余空格
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_drop_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_drop_answer(prediction).split()
    gold_tokens = normalize_drop_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_drop_em(prediction: str, gold: str) -> float:
    return 1.0 if normalize_drop_answer(prediction) == normalize_drop_answer(gold) else 0.0


def is_math_equal(pred: str, ref: str) -> bool:
    """
    判断两个数学答案是否相等
    """
    pred_norm = normalize_math_answer(pred)
    ref_norm = normalize_math_answer(ref)
    
    # 字符串完全匹配
    if pred_norm == ref_norm:
        return True
    
    # 尝试作为数字比较
    try:
        # 处理分数
        if '/' in pred_norm and '/' in ref_norm:
            pred_parts = pred_norm.split('/')
            ref_parts = ref_norm.split('/')
            if len(pred_parts) == 2 and len(ref_parts) == 2:
                pred_val = float(pred_parts[0]) / float(pred_parts[1])
                ref_val = float(ref_parts[0]) / float(ref_parts[1])
                return abs(pred_val - ref_val) < 1e-6
        
        # 处理普通数字
        pred_float = float(pred_norm)
        ref_float = float(ref_norm)
        return abs(pred_float - ref_float) < 1e-6
    except:
        pass
    
    return False


def evaluate_benchmark(
    benchmark_name: str,
    predictions: List[str],
    references: List[Any],
    data_items: List[Dict[str, Any]] = None
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    评估benchmark的结果
    
    Args:
        benchmark_name: benchmark名称
        predictions: 模型预测结果列表
        references: 参考答案列表
        data_items: 原始数据项（可选，用于某些需要额外信息的评估）
        
    Returns:
        (准确率, 详细结果列表)
    """
    if len(predictions) != len(references):
        raise ValueError(f"predictions和references长度不匹配: {len(predictions)} vs {len(references)}")
    
    correct = 0
    details = []
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_norm = None
        ref_norm = None
        ref_norm_list = None
        # 根据benchmark类型提取答案
        if benchmark_name == 'bbh':
            extracted_pred = extract_answer_bbh(pred)
            is_correct = (extracted_pred.lower() == str(ref).lower())
            
        elif benchmark_name == 'mmlu-pro':
            extracted_pred = extract_answer_text(pred)
            options = data_items[i].get('options', [])
            ref_answer = _get_choices_answer_text(ref, options)
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)
                
        elif benchmark_name == 'mmlu':
            extracted_pred = extract_answer_text(pred)
            choices = data_items[i].get('choices', [])
            ref_answer = _get_choices_answer_text(ref, choices)
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)
        
        elif benchmark_name == 'mmlu-redux':
            extracted_pred = extract_answer_text(pred)
            choices = data_items[i].get('choices', [])
            ref_answer = _get_choices_answer_text(ref, choices)
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)
            
        elif benchmark_name == 'gpqa':
            extracted_pred = extract_answer_text(pred)
            ref_answer = _get_gpqa_answer_text(ref, data_items[i])
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)
            
        elif benchmark_name == 'supergpqa':
            extracted_pred = extract_answer_multiple_choice(pred, 'ABCDEFGHIJ')
            is_correct = (extracted_pred == str(ref).upper())
            
        elif benchmark_name == 'gsm8k':
            extracted_pred = extract_answer_gsm8k(pred)
            # GSM8K的答案需要从#### xxx中提取
            if isinstance(ref, str):
                if '####' in ref:
                    ref_answer = ref.split('####')[1].strip().replace(',', '')
                else:
                    ref_answer = ref.strip().replace(',', '')
            else:
                ref_answer = str(ref).replace(',', '')
            is_correct = is_math_equal(extracted_pred, ref_answer)
            
        elif benchmark_name == 'math':
            pred =  pred.split('Problem:')[0]
            extracted_pred = extract_answer_math(pred)
            if extracted_pred.endswith('.'):
                extracted_pred = extracted_pred[:-1]
            is_correct = is_math_equal(extracted_pred, str(ref))

        elif benchmark_name == 'hellaswag':
            extracted_pred = extract_answer_text(pred)
            endings = data_items[i].get('endings', [])
            ref_answer = _get_choices_answer_text(ref, endings)
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)

        elif benchmark_name == 'arc-c':
            extracted_pred = extract_answer_text(pred)
            ref_answer = _get_arc_answer_text(ref, data_items[i])
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)

        elif benchmark_name == 'arc-e':
            extracted_pred = extract_answer_text(pred)
            ref_answer = _get_arc_answer_text(ref, data_items[i])
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)

        elif benchmark_name == 'winogrande':
            extracted_pred = extract_answer_text(pred)
            choices = [data_items[i].get('option1', ''), data_items[i].get('option2', '')]
            ref_answer = _get_choices_answer_text(ref, choices)
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm = normalize_nq_answer(ref_answer)
            is_correct = (pred_norm == ref_norm)

        elif benchmark_name == 'piqa':
            extracted_pred = extract_answer_multiple_choice(pred, 'AB')
            ref_answer = normalize_choice_ref(ref, 'AB', prefer_one_based=False)
            is_correct = (extracted_pred == ref_answer)

        elif benchmark_name == 'nq':
            extracted_pred = extract_answer_nq(pred)
            if isinstance(ref, list):
                gold_answers = [str(r) for r in ref if r is not None]
            else:
                gold_answers = [str(ref)]
            pred_norm = normalize_nq_answer(extracted_pred)
            ref_norm_list = [normalize_nq_answer(ans) for ans in gold_answers]
            is_correct = pred_norm in ref_norm_list

        elif benchmark_name == 'drop':
            extracted_pred = extract_answer_drop(pred)
            if isinstance(ref, list):
                gold_answers = [str(r) for r in ref if r is not None]
            else:
                gold_answers = [str(ref)]
            if not gold_answers:
                gold_answers = [""]
            em_scores = [compute_drop_em(extracted_pred, ga) for ga in gold_answers]
            f1_scores = [compute_drop_f1(extracted_pred, ga) for ga in gold_answers]
            best_em = max(em_scores) if em_scores else 0.0
            best_f1 = max(f1_scores) if f1_scores else 0.0
            is_correct = (best_em == 1.0)
            
        else:
            # 默认：直接字符串匹配
            extracted_pred = pred.strip()
            is_correct = (extracted_pred == str(ref))
        
        if is_correct:
            correct += 1
        
        detail = {
            'index': i,
            'prediction': pred,
            'extracted_answer': extracted_pred,
            'reference': ref,
            'correct': is_correct
        }
        if pred_norm is not None:
            detail['normalized_prediction'] = pred_norm
        if ref_norm is not None:
            detail['normalized_reference'] = ref_norm
        if ref_norm_list is not None:
            detail['normalized_references'] = ref_norm_list
        if benchmark_name == 'drop':
            detail['em'] = best_em
            detail['f1'] = best_f1
        details.append(detail)
    
    accuracy = correct / len(predictions) * 100 if predictions else 0.0
    
    return accuracy, details


# 测试代码
if __name__ == '__main__':
    print("=== 测试 BBH 提取 ===")
    test_cases = [
        ("Let's think step by step. ... So the answer is True.", "True"),
        ("Analysis... The answer is False", "False"),
        ("...reasoning... So the answer is 42", "42"),
    ]
    
    for response, expected in test_cases:
        extracted = extract_answer_bbh(response)
        match = extracted.lower() == expected.lower()
        print(f"✓" if match else "✗", f"Extracted: '{extracted}', Expected: '{expected}'")
    
    print("\n=== 测试 MMLU-Pro 提取 ===")
    test_cases = [
        ("Let's think step by step... The answer is (A).", "A"),
        ("After analysis, the answer is (H).", "H"),
        ("The correct answer is B", "B"),
    ]
    
    for response, expected in test_cases:
        extracted = extract_answer_mmlu_pro(response)
        match = extracted == expected
        print(f"✓" if match else "✗", f"Extracted: '{extracted}', Expected: '{expected}'")
    
    print("\n=== 测试 MMLU 提取 ===")
    test_cases = [
        ("A", "A"),
        ("The answer is B", "B"),
        ("C.", "C"),
    ]
    
    for response, expected in test_cases:
        extracted = extract_answer_mmlu(response)
        match = extracted == expected
        print(f"✓" if match else "✗", f"Extracted: '{extracted}', Expected: '{expected}'")
    
    print("\n=== 测试 GSM8K 提取 ===")
    test_cases = [
        ("... The answer is 42", "42"),
        ("calculation... answer is 1,234", "1234"),
        ("#### 567", "567"),
    ]
    
    for response, expected in test_cases:
        extracted = extract_answer_gsm8k(response)
        match = extracted == expected
        print(f"✓" if match else "✗", f"Extracted: '{extracted}', Expected: '{expected}'")
    
    print("\n=== 测试 MATH 提取 ===")
    test_cases = [
        ("Solution: ... $\\boxed{42}$. Final Answer: ...", "42"),
        ("... Final Answer: The final answer is $[2,5)$. I hope it is correct.", "[2,5)"),
        ("... $\\boxed{-\\frac{2}{3}}$.", "-\\frac{2}{3}"),
    ]
    
    for response, expected in test_cases:
        extracted = extract_answer_math(response)
        # 对于MATH，使用normalize比较
        match = normalize_math_answer(extracted) == normalize_math_answer(expected)
        print(f"✓" if match else "✗", f"Extracted: '{extracted}', Expected: '{expected}'")

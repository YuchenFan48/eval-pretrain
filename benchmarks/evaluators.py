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
        return matches[-1]
    
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
        return matches[-1]
    
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
        return numbers[-1].replace(',', '')
    
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
        return dollar_matches[-1].strip()
    
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
            return matches[-1]
    
    return ""


def extract_answer_multiple_choice(response: str, options: str = 'ABCD') -> str:
    """
    强化版多选题答案提取，解决 CoT 场景下的稳定性问题
    """
    if not response:
        return ""

    # 1. 预处理：只取最后一部分，防止被 Question 重复或长篇推理干扰
    # 通常答案在最后 200 个字符内
    footer = response[-200:] if len(response) > 200 else response
    
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
            return matches[-1].group(1).upper()

    # 3. 兜底策略 1: 查找最后一行出现的孤立字母
    last_line = response.strip().split('\n')[-1]
    match = re.search(rf'\b([{options}])\b', last_line)
    if match:
        return match.group(1).upper()

    # 4. 兜底策略 2: 查找全文中最后一个出现的选项字母
    # 注意：这里要加上 \b 边界触发，防止匹配到单词内部的字母
    matches = re.findall(rf'\b([{options}])\b', response)
    if matches:
        return matches[-1].upper()

    return ""

def extract_answer_mmlu_redux(response: str) -> str:
    """
    专门为 MMLU-Redux 优化的提取器
    """
    # MMLU-Redux 通常是 ABCD，但为了兼容性保留
    return extract_answer_multiple_choice(response, 'ABCD')


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
        # 根据benchmark类型提取答案
        if benchmark_name == 'bbh':
            extracted_pred = extract_answer_bbh(pred)
            is_correct = (extracted_pred.lower() == str(ref).lower())
            
        elif benchmark_name == 'mmlu-pro':
            extracted_pred = extract_answer_mmlu_pro(pred)
            # MMLU-Pro的答案可能是字母或索引
            if isinstance(ref, str):
                is_correct = (extracted_pred == ref.upper())
            elif isinstance(ref, int):
                # 如果ref是索引，转换为字母
                ref_letter = 'ABCDEFGHIJKLMNOP'[ref]
                is_correct = (extracted_pred == ref_letter)
            else:
                is_correct = (extracted_pred == str(ref).upper())
                
        elif benchmark_name == 'mmlu':
            extracted_pred = extract_answer_mmlu(pred)
            # 处理不同格式的参考答案
            if isinstance(ref, str):
                ref_answer = ref.upper() if len(ref) == 1 else ref
            elif isinstance(ref, int):
                ref_answer = 'ABCD'[ref]
            else:
                ref_answer = str(ref).upper()
            is_correct = (extracted_pred == ref_answer)
        
        elif benchmark_name == 'mmlu-redux':
            # 1. 尝试更严格的提取
            extracted_pred = extract_answer_multiple_choice(pred, 'ABCD')
            
            # 2. 针对 Redux 可能存在的特殊情况（如参考答案是 index）进行处理
            if isinstance(ref, int):
                ref_answer = 'ABCD'[ref]
            else:
                ref_answer = str(ref).upper().strip()
            
            is_correct = (extracted_pred == ref_answer)
            
        elif benchmark_name == 'gpqa':
            extracted_pred = extract_answer_gpqa(pred)
            if isinstance(ref, str):
                ref_answer = ref.upper() if len(ref) == 1 else ref
            elif isinstance(ref, int):
                ref_answer = 'ABCD'[ref]
            else:
                ref_answer = str(ref).upper()
            is_correct = (extracted_pred == ref_answer)
            
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
            
        else:
            # 默认：直接字符串匹配
            extracted_pred = pred.strip()
            is_correct = (extracted_pred == str(ref))
        
        if is_correct:
            correct += 1
        
        details.append({
            'index': i,
            'prediction': pred,
            'extracted_answer': extracted_pred,
            'reference': ref,
            'correct': is_correct
        })
    
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

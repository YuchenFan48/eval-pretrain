"""
代码提取模块：从模型输出中提取代码
"""

import re
from typing import Optional, List


def select_best_code_block(blocks: List[str]) -> Optional[str]:
    """
    在多个代码块中选择最可能的主代码块。
    """
    if not blocks:
        return None

    def score(block: str) -> int:
        score_val = len(block)
        if 'def ' in block or 'class ' in block:
            score_val += 1000
        return score_val

    return max(blocks, key=score).strip()


def extract_code_from_markdown(text: str, language: str = 'python') -> Optional[str]:
    """
    从 Markdown 代码块中提取代码
    
    Args:
        text: 包含代码的文本
        language: 编程语言
        
    Returns:
        提取的代码，如果未找到则返回 None
    """
    # 尝试匹配 ```python ... ``` 格式
    pattern = rf'```{language}\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return select_best_code_block(matches)
    
    # 尝试匹配 ```...``` 格式（没有语言标识）
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return select_best_code_block(matches)
    
    return None


def extract_python_code(text: str) -> str:
    """
    从文本中提取 Python 代码
    
    优先级：
    1. [BEGIN]...[DONE] 标记（用于 MBPP 等）
    2. Markdown 代码块
    3. 函数定义（def ...）
    4. 类定义（class ...）
    5. 原始文本（去除解释性文字）
    
    Args:
        text: 包含代码的文本
        
    Returns:
        提取的代码
    """
    # 0. 清理 answer 标记（如 [/ANSWER]、[ANSWER] 等）
    text = re.sub(r'\[/?ANSWER\]', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # 1. 尝试提取 [DONE] 之前的代码（用于 MBPP 格式）
    # 按 [DONE] 分割，取第一个代码块
    if '[DONE]' in text.upper():
        # 按 [DONE] 分割（大小写不敏感）
        parts = re.split(r'\[DONE\]', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # 取第一个 [DONE] 之前的部分
            first_code = parts[0].strip()
            
            # 清理可能存在的 [BEGIN] 标记（来自 few-shot 示例）
            first_code = re.sub(r'\[BEGIN\]\s*', '', first_code, flags=re.IGNORECASE).strip()
            
            return first_code
    
    # 2. 尝试从 Markdown 代码块提取
    code = extract_code_from_markdown(text, 'python')
    if code:
        return code
    
    # 3. 尝试提取函数定义
    # 匹配 def ... 开始的代码块
    func_pattern = r'(def\s+\w+\s*\([^)]*\)[^:]*:(?:.*?)(?=\n(?:def\s|\Z|\nclass\s)))'
    matches = re.findall(func_pattern, text, re.DOTALL)
    
    if matches:
        # 返回所有匹配的函数（可能有多个）
        return '\n\n'.join(matches)
    
    # 4. 尝试提取类定义
    class_pattern = r'(class\s+\w+[^:]*:(?:.*?)(?=\n(?:class\s|\Z)))'
    matches = re.findall(class_pattern, text, re.DOTALL)
    
    if matches:
        return '\n\n'.join(matches)
    
    # 5. 移除常见的解释性前缀和后缀
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过明显的解释性文本
        if any(stripped.lower().startswith(prefix) for prefix in [
            'here',  # 'Here is', 'Here\'s'
            'this',  # 'This code', 'This function'
            'the ',  # 'The solution', 'The function'
            'you',   # 'You can', 'You should'
            'let',   # 'Let me', 'Let\'s'
            'note:', 'explanation:', 'output:',
        ]):
            continue
        
        # 如果是代码行
        if stripped and (
            stripped.startswith('def ') or 
            stripped.startswith('class ') or
            stripped.startswith('import ') or
            stripped.startswith('from ') or
            in_code and (stripped[0] in ' \t' or stripped[0].isalnum() or stripped[0] in '#@"\'()[]{},.:=+-*/<>')
        ):
            code_lines.append(line)
            in_code = True
        elif in_code and not stripped:
            # 保留代码中的空行
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # 6. 如果都失败了，返回原始文本
    return text.strip()


def extract_code_by_language(text: str, language: str) -> str:
    """
    根据编程语言提取代码
    
    Args:
        text: 包含代码的文本
        language: 编程语言
        
    Returns:
        提取的代码
    """
    if language.lower() in ['python', 'py', 'python3']:
        return extract_python_code(text)
    
    # 清理 answer 标记
    text = re.sub(r'\[/?ANSWER\]', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # 对于其他语言，尝试从 Markdown 提取
    code = extract_code_from_markdown(text, language)
    if code:
        return code
    
    # 否则返回原始文本
    return text.strip()


def clean_code(code: str) -> str:
    """
    清理代码（移除多余的空行、注释等）
    
    Args:
        code: 原始代码
        
    Returns:
        清理后的代码
    """
    lines = code.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        stripped = line.rstrip()
        
        # 跳过连续的空行（只保留一个）
        if not stripped:
            if not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
            continue
        
        cleaned_lines.append(stripped)
        prev_empty = False
    
    # 移除开头和结尾的空行
    while cleaned_lines and not cleaned_lines[0]:
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


# 测试代码
if __name__ == '__main__':
    # 测试用例1: Markdown 代码块
    test1 = """
Here's a Python function to solve this problem:

```python
def remove_Occ(s, ch):
    first = s.find(ch)
    if first == -1:
        return s
    s = s[:first] + s[first+1:]
    last = s.rfind(ch)
    if last != -1:
        s = s[:last] + s[last+1:]
    return s
```

This function removes the first and last occurrence of a character.
"""
    
    print("测试用例1 (Markdown):")
    code1 = extract_python_code(test1)
    print(code1)
    print("\n" + "="*60 + "\n")
    
    # 测试用例2: 直接函数定义
    test2 = """
Here is the solution:

def sort_matrix(M):
    result = sorted(M, key=sum)
    return result

You can test it with the examples provided.
"""
    
    print("测试用例2 (直接定义):")
    code2 = extract_python_code(test2)
    print(code2)
    print("\n" + "="*60 + "\n")
    
    # 测试用例3: 混合文本
    test3 = """
To solve this problem, we need to:
1. Use the sorted function
2. Pass a key function

def sort_matrix(M):
    return sorted(M, key=sum)

This will work correctly.
"""
    
    print("测试用例3 (混合文本):")
    code3 = extract_python_code(test3)
    print(code3)

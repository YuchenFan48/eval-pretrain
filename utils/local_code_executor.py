"""
本地代码执行器：不依赖 Docker 的代码评测模块
用于在没有 Docker 环境时进行代码评测
"""

import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class LocalCodeExecutor:
    """本地代码执行器（不使用 Docker）"""
    
    def __init__(
        self,
        max_workers: int = 16,
        timeout: int = 30,
    ):
        """
        初始化本地代码执行器
        
        Args:
            max_workers: 最大并发worker数量
            timeout: 单个样本的超时时间（秒）
        """
        self.max_workers = max_workers
        self.timeout = timeout
    
    def run_code_with_tests(
        self,
        code: str,
        test_code: str,
        language: str = 'python',
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行代码并执行测试
        
        Args:
            code: 要测试的代码
            test_code: 测试代码
            language: 编程语言
            timeout: 超时时间（秒）
            
        Returns:
            执行结果字典
        """
        if timeout is None:
            timeout = self.timeout
        
        # 合并代码和测试
        full_code = f"{code}\n\n{test_code}"
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py' if language == 'python' else '',
            delete=False
        ) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # 执行代码
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'status': 'Success' if result.returncode == 0 else 'Failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0,
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'Timeout',
                'returncode': -1,
                'stdout': '',
                'stderr': f'Execution timeout after {timeout} seconds',
                'passed': False,
            }
        except Exception as e:
            return {
                'status': 'Error',
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'passed': False,
            }
        finally:
            # 删除临时文件
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def evaluate_mbpp(
        self,
        task_id: str,
        code: str,
        test_list: List[str],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        评测 MBPP 代码
        
        Args:
            task_id: 任务ID
            code: 生成的代码
            test_list: 测试列表
            timeout: 超时时间
            
        Returns:
            评测结果
        """
        # 构建测试代码
        test_code = '\n'.join(test_list)
        
        # 执行代码
        result = self.run_code_with_tests(code, test_code, timeout=timeout)
        
        return {
            'id': str(task_id),
            'accepted': result['passed'],
            'extracted_code': code,
            'status': result['status'],
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'tests': [{
                'passed': result['passed'],
                'error': result['stderr'] if not result['passed'] else None,
            }],
        }
    
    def evaluate_humaneval(
        self,
        task_id: str,
        code: str,
        test_code: str,
        entry_point: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        评测 HumanEval 代码
        
        Args:
            task_id: 任务ID
            code: 生成的代码
            test_code: 测试代码
            entry_point: 入口函数名
            timeout: 超时时间
            
        Returns:
            评测结果
        """
        # 构建完整测试
        full_test = f"{test_code}\ncheck({entry_point})"
        
        # 执行代码
        result = self.run_code_with_tests(code, full_test, timeout=timeout)
        
        return {
            'id': str(task_id),
            'accepted': result['passed'],
            'extracted_code': code,
            'status': result['status'],
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'tests': [{
                'passed': result['passed'],
                'error': result['stderr'] if not result['passed'] else None,
            }],
        }
    
    def _evaluate_with_index(
        self,
        index: int,
        dataset: str,
        task_id: str,
        completion: str,
        test_data: Dict[str, Any],
        timeout: int,
    ) -> tuple:
        """
        评测代码（带索引）
        
        Returns:
            (索引, 评测结果)
        """
        try:
            if dataset == 'mbpp':
                result = self.evaluate_mbpp(
                    task_id=task_id,
                    code=completion,
                    test_list=test_data.get('test_list', []),
                    timeout=timeout,
                )
            elif dataset in ['humaneval', 'humanevalplus']:
                result = self.evaluate_humaneval(
                    task_id=task_id,
                    code=completion,
                    test_code=test_data.get('test', ''),
                    entry_point=test_data.get('entry_point', ''),
                    timeout=timeout,
                )
            else:
                result = {
                    'id': str(task_id),
                    'accepted': False,
                    'error': f'Unsupported dataset: {dataset}',
                }
            
            return (index, result)
            
        except Exception as e:
            return (index, {
                'id': str(task_id),
                'accepted': False,
                'error': str(e),
            })
    
    def batch_evaluate(
        self,
        dataset: str,
        task_ids: List[str],
        completions: List[str],
        test_data_list: List[Dict[str, Any]],
        show_progress: bool = True,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量评测代码
        
        Args:
            dataset: 数据集名称
            task_ids: 任务ID列表
            completions: 代码补全列表
            test_data_list: 测试数据列表
            show_progress: 是否显示进度
            max_workers: 最大并发worker数量
            timeout: 超时时间
            
        Returns:
            评测结果列表
        """
        if timeout is None:
            timeout = self.timeout
        
        total = len(task_ids)
        workers = max_workers if max_workers is not None else self.max_workers
        
        print(f"使用本地执行器评测 {total} 个代码样本（并发: {workers}）")
        
        # 初始化结果列表
        results = [None] * total
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    self._evaluate_with_index,
                    i,
                    dataset,
                    task_id,
                    completion,
                    test_data,
                    timeout
                ): i for i, (task_id, completion, test_data) in enumerate(
                    zip(task_ids, completions, test_data_list)
                )
            }
            
            # 使用tqdm显示进度
            if show_progress:
                pbar = tqdm(total=total, desc="代码评测（本地）", unit="samples")
            
            # 收集结果
            completed = 0
            success = 0
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    
                    # 统计成功数
                    if result.get('accepted', False):
                        success += 1
                    
                    if show_progress:
                        pbar.update(1)
                        pbar.set_postfix({
                            '完成': f'{completed}/{total}',
                            '通过': f'{success}/{completed}'
                        })
                except Exception as e:
                    index = futures[future]
                    print(f"\n✗ 评测索引 {index} 时出错: {e}")
                    results[index] = {
                        'id': task_ids[index],
                        'accepted': False,
                        'error': str(e),
                    }
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        return results


# 测试代码
if __name__ == '__main__':
    # 创建执行器
    executor = LocalCodeExecutor(max_workers=4, timeout=10)
    
    # 测试 MBPP 样例
    print("测试 MBPP 样例...")
    code = """
def remove_Occ(s, ch):
    first = s.find(ch)
    if first == -1:
        return s
    s = s[:first] + s[first+1:]
    last = s.rfind(ch)
    if last != -1:
        s = s[:last] + s[last+1:]
    return s
"""
    
    test_list = [
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"',
        'assert remove_Occ("PHP","P") == "H"',
    ]
    
    result = executor.evaluate_mbpp(
        task_id='11',
        code=code,
        test_list=test_list,
    )
    
    print(f"结果: {'通过' if result['accepted'] else '失败'}")
    print(f"状态: {result['status']}")
    if result['stderr']:
        print(f"错误: {result['stderr']}")
    
    print("\n测试完成！")


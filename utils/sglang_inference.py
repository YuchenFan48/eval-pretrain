"""
SGLang推理模块：使用sglang进行模型推理
支持并发推理以提高速度
"""

import os
import time
import subprocess
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
from tqdm import tqdm

from .base_inference import BaseInference



class SGLangInference(BaseInference):
    """SGLang推理类（支持并发）"""
    
    def __init__(
        self,
        model_path: str,
        host: str = '127.0.0.1',
        port: int = 30000,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_workers: int = 32,  # 并发worker数量
    ):
        """
        初始化SGLang推理器（仅API模式）
        
        Args:
            model_path: 模型路径（保留用于记录）
            host: API服务器地址
            port: API服务器端口
            temperature: 采样温度
            max_tokens: 最大生成token数
            max_workers: 最大并发worker数量
        """
        super().__init__(temperature, max_tokens, max_workers)
        
        self.model_path = model_path
        self.host = host
        self.port = port
        self.api_url = f"http://{host}:{port}/generate"
        
        # 创建session以复用连接，避免连接堆积
        self.session = requests.Session()
        # 设置连接池大小
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def check_server(self, max_retries: int = 5) -> bool:
        """
        检查服务器是否正常运行
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            服务器是否正常运行
        """
        for i in range(max_retries):
            try:
                response = self.session.get(f"http://{self.host}:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 120,
    ) -> str:
        """
        生成单个回答
        
        Args:
            prompt: 输入prompt
            temperature: 采样温度（可选）
            max_tokens: 最大生成token数（可选）
            timeout: 请求超时时间（秒）
            
        Returns:
            生成的文本
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_tok,
            }
        }
        
        try:
            response = self.session.post(self.api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result.get('text', '')
        except Exception as e:
            print(f"生成失败: {e}")
            return ""
    
    def _generate_with_index(
        self,
        index: int,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 120,
    ) -> tuple:
        """
        生成单个回答（带索引）
        
        Args:
            index: 样本索引
            prompt: 输入prompt
            temperature: 采样温度（可选）
            max_tokens: 最大生成token数（可选）
            timeout: 请求超时时间（秒）
            
        Returns:
            (索引, 生成的文本)
        """
        result = self.generate(prompt, temperature, max_tokens, timeout)
        return (index, result)
    
    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        batch_size: int = 8,
        show_progress: bool = True,
        max_workers: Optional[int] = None,
        timeout: int = 120,
    ) -> List[str]:
        """
        批量生成回答（并发版本）
        
        Args:
            prompts: 输入prompt列表
            temperature: 采样温度（可选）
            max_tokens: 最大生成token数（可选）
            batch_size: 批处理大小（已弃用，现在使用max_workers控制并发）
            show_progress: 是否显示进度
            max_workers: 最大并发worker数量（可选，默认使用初始化时的值）
            timeout: 请求超时时间（秒）
            
        Returns:
            生成的文本列表
        """
        total = len(prompts)
        workers = max_workers if max_workers is not None else self.max_workers
        
        print(f"使用 {workers} 个并发workers处理 {total} 个prompts")
        
        # 初始化结果列表
        results = [None] * total
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    self._generate_with_index,
                    i,
                    prompt,
                    temperature,
                    max_tokens,
                    timeout
                ): i for i, prompt in enumerate(prompts)
            }
            
            # 使用tqdm显示进度
            if show_progress:
                pbar = tqdm(total=total, desc="推理进度", unit="samples")
            
            # 收集结果
            completed = 0
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
                        pbar.set_postfix({'完成': f'{completed}/{total}'})
                except Exception as e:
                    index = futures[future]
                    print(f"\n✗ 处理索引 {index} 时出错: {e}")
                    results[index] = ""
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        return results
    
    def batch_generate_chunked(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        chunk_size: int = 100,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        timeout: int = 120,
    ) -> List[str]:
        """
        分块批量生成回答（适用于超大数据集）
        
        Args:
            prompts: 输入prompt列表
            temperature: 采样温度（可选）
            max_tokens: 最大生成token数（可选）
            chunk_size: 每块的大小
            max_workers: 最大并发worker数量（可选）
            show_progress: 是否显示进度
            timeout: 请求超时时间（秒）
            
        Returns:
            生成的文本列表
        """
        total = len(prompts)
        results = []
        
        print(f"使用分块模式处理 {total} 个prompts，每块 {chunk_size} 个")
        
        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = prompts[chunk_start:chunk_end]
            
            print(f"\n处理块 [{chunk_start}:{chunk_end}] ({len(chunk)} samples)")
            
            chunk_results = self.batch_generate(
                prompts=chunk,
                temperature=temperature,
                max_tokens=max_tokens,
                show_progress=show_progress,
                max_workers=max_workers,
                timeout=timeout,
            )
            
            results.extend(chunk_results)
        
        return results
    
    def compute_logprob(
        self,
        text: str,
        timeout: int = 120,
    ) -> dict:
        """
        计算文本的对数概率
        
        Args:
            text: 输入文本
            timeout: 请求超时时间（秒）
            
        Returns:
            包含对数概率等信息的字典
        """
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0.0,  # 使用贪心解码
                "max_new_tokens": 1,  # 生成1个token以触发logprob计算
            },
            "return_logprob": True,  # 返回对数概率
            "logprob_start_len": 0,  # 从第0个token开始返回logprob
        }
        
        try:
            response = self.session.post(self.api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            # 从结果中提取logprob信息
            meta_info = result.get('meta_info', {})
            raw_input_token_logprobs = meta_info.get('input_token_logprobs', [])
            
            # SGLang返回的格式是: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
            # 我们需要提取每个元素的第一个值（logprob）和第二个值（token_id）
            input_token_logprobs = []
            input_tokens = []
            
            for item in raw_input_token_logprobs:
                if isinstance(item, list) and len(item) >= 2:
                    # 提取logprob（第一个元素）
                    input_token_logprobs.append(item[0])
                    # 提取token_id（第二个元素）
                    input_tokens.append(item[1])
                elif isinstance(item, (int, float)):
                    # 如果直接是数字，就直接使用
                    input_token_logprobs.append(item)
            
            return {
                'logprobs': input_token_logprobs,
                'token_ids': input_tokens,
                'prompt_tokens': len(input_tokens),
            }
        except Exception as e:
            print(f"计算logprob失败: {e}")
            return {
                'logprobs': [],
                'token_ids': [],
                'prompt_tokens': 0,
            }
    
    def compute_logprob_with_prompt(
        self,
        prompt: str,
        answer: str,
        timeout: int = 120,
    ) -> dict:
        """
        计算文本的对数概率（支持分离prompt和answer）
        只返回answer部分的logprobs，但需要完整text作为上下文
        
        Args:
            prompt: prompt文本
            answer: answer文本
            timeout: 请求超时时间（秒）
            
        Returns:
            包含answer部分对数概率等信息的字典
        """
        try:
            # 完整文本
            full_text = "<|begin_text|>" + prompt + answer + "<|end_text|>"
            
            # 根据文本长度动态调整timeout
            # 估算：每1000个字符需要约10秒
            estimated_time = max(timeout, len(full_text) / 1000 * 10)
            actual_timeout = min(estimated_time, 600)  # 最大600秒
            
            # 计算完整文本的logprobs
            full_result = self.compute_logprob(full_text, int(actual_timeout))
            
            if not full_result['logprobs']:
                return {
                    'logprobs': [],
                    'token_ids': [],
                    'prompt_tokens': 0,
                    'answer_tokens': 0,
                }
            
            # 计算prompt部分的token数量（使用较短的timeout）
            prompt_timeout = min(timeout, 120)
            prompt_result = self.compute_logprob(prompt, prompt_timeout)
            prompt_token_count = len(prompt_result['token_ids'])
            
            # 提取answer部分的logprobs（跳过prompt的token）
            all_logprobs = full_result['logprobs']
            all_token_ids = full_result['token_ids']
            #将all_token_ids decode出来
            
            # 注意：input_token_logprobs的第一个元素通常是None或0（因为第一个token没有前文）
            # 所以answer部分应该从prompt_token_count开始取
            answer_logprobs = all_logprobs[prompt_token_count:]
            answer_token_ids = all_token_ids[prompt_token_count:]
            
            return {
                'logprobs': answer_logprobs,
                'token_ids': answer_token_ids,
                'prompt_tokens': prompt_token_count,
                'answer_tokens': len(answer_token_ids),
            }
            
        except Exception as e:
            # 超时或其他错误，返回空结果
            if 'timeout' in str(e).lower():
                print(f"⚠️  样本超时 (文本长度: {len(prompt)+len(answer)} chars)")
            else:
                print(f"⚠️  计算失败: {e}")
            
            return {
                'logprobs': [],
                'token_ids': [],
                'prompt_tokens': 0,
                'answer_tokens': 0,
            }
    
    def _compute_logprob_with_index(
        self,
        index: int,
        text: str,
        timeout: int = 120,
    ) -> tuple:
        """
        计算文本的对数概率（带索引）
        
        Args:
            index: 样本索引
            text: 输入文本
            timeout: 请求超时时间（秒）
            
        Returns:
            (索引, logprob结果字典)
        """
        text = "<|begin_text|>" + text + "<|end_text|>"
        result = self.compute_logprob(text, timeout)
        return (index, result)
    
    def _compute_logprob_with_prompt_and_index(
        self,
        index: int,
        prompt: str,
        answer: str,
        timeout: int = 120,
    ) -> tuple:
        """
        计算文本的对数概率（带索引，支持prompt/answer分离）
        
        Args:
            index: 样本索引
            prompt: prompt文本
            answer: answer文本
            timeout: 请求超时时间（秒）
            
        Returns:
            (索引, logprob结果字典)
        """

        result = self.compute_logprob_with_prompt(prompt, answer, timeout)
        return (index, result)
    
    def batch_compute_ppl(
        self,
        texts: List[str],
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        timeout: int = 120,
    ) -> List[dict]:
        """
        批量计算困惑度（Perplexity）
        
        Args:
            texts: 输入文本列表
            max_workers: 最大并发worker数量（可选）
            show_progress: 是否显示进度
            timeout: 请求超时时间（秒）
            
        Returns:
            包含PPL信息的字典列表
        """
        total = len(texts)
        workers = max_workers if max_workers is not None else self.max_workers
        
        print(f"使用 {workers} 个并发workers计算 {total} 个文本的PPL")
        
        # 初始化结果列表
        results = [None] * total
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    self._compute_logprob_with_index,
                    i,
                    text,
                    timeout
                ): i for i, text in enumerate(texts)
            }
            
            # 使用tqdm显示进度
            if show_progress:
                pbar = tqdm(total=total, desc="计算PPL", unit="samples")
            
            # 收集结果
            completed = 0
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
                        pbar.set_postfix({'完成': f'{completed}/{total}'})
                except Exception as e:
                    index = futures[future]
                    print(f"\n✗ 处理索引 {index} 时出错: {e}")
                    results[index] = {
                        'logprobs': [],
                        'token_ids': [],
                        'prompt_tokens': 0,
                    }
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        return results
    
    def batch_compute_ppl_with_prompt(
        self,
        prompts: List[str],
        answers: List[str],
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        timeout: int = 120,
    ) -> List[dict]:
        """
        批量计算困惑度（Perplexity），支持prompt/answer分离
        只计算answer部分的PPL
        
        Args:
            prompts: prompt文本列表
            answers: answer文本列表
            max_workers: 最大并发worker数量（可选）
            show_progress: 是否显示进度
            timeout: 请求超时时间（秒）
            
        Returns:
            包含answer部分PPL信息的字典列表
        """
        if len(prompts) != len(answers):
            raise ValueError(f"prompts和answers长度不匹配: {len(prompts)} vs {len(answers)}")
        
        total = len(prompts)
        workers = max_workers if max_workers is not None else self.max_workers
        
        print(f"使用 {workers} 个并发workers计算 {total} 个文本的PPL（只计算answer部分）")
        
        # 初始化结果列表
        results = [None] * total
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    self._compute_logprob_with_prompt_and_index,
                    i,
                    prompt,
                    answer,
                    timeout
                ): i for i, (prompt, answer) in enumerate(zip(prompts, answers))
            }
            
            # 使用tqdm显示进度
            if show_progress:
                pbar = tqdm(total=total, desc="计算PPL（answer部分）", unit="samples")
            
            # 收集结果
            completed = 0
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
                        pbar.set_postfix({'完成': f'{completed}/{total}'})
                except Exception as e:
                    index = futures[future]
                    print(f"\n✗ 处理索引 {index} 时出错: {e}")
                    results[index] = {
                        'logprobs': [],
                        'token_ids': [],
                        'prompt_tokens': 0,
                        'answer_tokens': 0,
                    }
                    completed += 1
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        return results

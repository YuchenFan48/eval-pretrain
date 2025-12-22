"""
OpenAI API推理模块：使用OpenAI API进行模型推理
支持并发推理以提高速度
"""

import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("警告: 未安装openai库，请运行: pip install openai")
    OpenAI = None

from .base_inference import BaseInference


class OpenAIInference(BaseInference):
    """OpenAI API推理类（支持并发）"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_workers: int = 32,
    ):
        """
        初始化OpenAI推理器
        
        Args:
            api_key: OpenAI API密钥
            model: 模型名称（如 "gpt-4", "gpt-3.5-turbo"）
            base_url: API基础URL（可选，用于自定义API端点）
            temperature: 采样温度
            max_tokens: 最大生成token数
            max_workers: 最大并发worker数量
        """
        super().__init__(temperature, max_tokens, max_workers)
        
        if OpenAI is None:
            raise ImportError("请安装openai库: pip install openai")
        
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
        # 初始化OpenAI客户端
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        print(f"✓ OpenAI推理器初始化成功")
        print(f"  - 模型: {model}")
        if base_url:
            print(f"  - API地址: {base_url}")
        print(f"  - 并发workers: {max_workers}")
    
    def check_server(self) -> bool:
        """
        检查API是否可用
        
        Returns:
            API是否可用
        """
        try:
            # 尝试发送一个简单的请求
            response = self.client.completions.create(
                model=self.model,
                prompt="test",
                max_tokens=1,
                timeout=10,
            )
            print(response)
            return True
        except Exception as e:
            print(f"API连接失败: {e}")
            return False
    
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
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tok,
                timeout=timeout,
            )
            return response.choices[0].text
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
    
    def batch_compute_ppl(
        self,
        texts: List[str],
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        timeout: int = 120,
    ) -> List[dict]:
        """
        批量计算困惑度（Perplexity）
        
        注意：此功能需要API支持logprobs参数，标准OpenAI API可能不支持
        
        Args:
            texts: 输入文本列表
            max_workers: 最大并发worker数量（可选）
            show_progress: 是否显示进度
            timeout: 请求超时时间（秒）
            
        Returns:
            包含PPL信息的字典列表
        """
        print("警告: OpenAI后端的PPL计算功能可能不被支持，因为标准OpenAI API不返回logprobs")
        print("如果您使用的是兼容OpenAI的API（如vLLM），请确保API支持logprobs参数")
        
        total = len(texts)
        workers = max_workers if max_workers is not None else self.max_workers
        
        print(f"使用 {workers} 个并发workers计算 {total} 个文本的PPL")
        
        # 初始化结果列表 - 返回空结果
        results = []
        for _ in range(total):
            results.append({
                'logprobs': [],
                'token_ids': [],
                'prompt_tokens': 0,
            })
        
        print("✗ OpenAI后端暂不支持PPL计算")
        return results


# 使用示例
if __name__ == '__main__':
    # 创建推理器
    inference = OpenAIInference(
        api_key="your-api-key",
        model="gpt-4",
        temperature=0.0,
        max_tokens=512,
        max_workers=32,
    )
    
    # 检查API
    if not inference.check_server():
        print("API不可用")
        exit(1)
    
    # 测试单个生成
    prompt = "What is the capital of France?"
    response = inference.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # 测试并发批量生成
    prompts = [
        f"What is {i}+{i}?" for i in range(10)
    ]
    
    print("\n测试并发批量生成...")
    start_time = time.time()
    responses = inference.batch_generate(
        prompts, 
        max_workers=5,
        show_progress=True
    )
    elapsed = time.time() - start_time
    
    print(f"\n完成 {len(responses)} 个推理")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"平均速度: {len(responses)/elapsed:.2f} samples/s")

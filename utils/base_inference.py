"""
推理基类：定义统一的推理接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseInference(ABC):
    """推理基类，定义统一的接口"""
    
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_workers: int = 32,
    ):
        """
        初始化推理器
        
        Args:
            temperature: 采样温度
            max_tokens: 最大生成token数
            max_workers: 最大并发worker数量
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        批量生成回答
        
        Args:
            prompts: 输入prompt列表
            temperature: 采样温度（可选）
            max_tokens: 最大生成token数（可选）
            batch_size: 批处理大小
            show_progress: 是否显示进度
            max_workers: 最大并发worker数量（可选）
            timeout: 请求超时时间（秒）
            
        Returns:
            生成的文本列表
        """
        pass
    
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
    
    def check_server(self) -> bool:
        """
        检查服务器/API是否可用
        
        Returns:
            是否可用
        """
        return True
    
    def start_server(self, wait_time: int = 30):
        """
        启动服务器（如果需要）
        
        Args:
            wait_time: 等待时间（秒）
        """
        pass
    
    def stop_server(self):
        """停止服务器（如果需要）"""
        pass


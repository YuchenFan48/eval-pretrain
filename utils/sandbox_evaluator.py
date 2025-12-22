"""
Sandbox Fusion 代码评测模块
用于评测代码生成 benchmark（MBPP, HumanEval 等）
"""

import requests
import json
import subprocess
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class SandboxEvaluator:
    """Sandbox Fusion 代码评测器"""
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 8080,
        max_workers: int = 16,
        docker_image: str = 'volcengine/sandbox-fusion:server-20250609',
    ):
        """
        初始化 Sandbox 评测器
        
        Args:
            host: Sandbox 服务器地址
            port: Sandbox 服务器端口
            max_workers: 最大并发worker数量
            docker_image: Docker 镜像名称
        """
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.docker_image = docker_image
        self.base_url = f"http://{host}:{port}"
        self.run_code_url = f"{self.base_url}/run_code"
        self.get_prompts_url = f"{self.base_url}/get_prompts"
        self.submit_url = f"{self.base_url}/submit"
        self.container_name = f"sandbox-fusion-{port}"
        self.process = None
    
    def start_server(self, wait_time: int = 30):
        """
        启动 Sandbox 服务器（使用 Docker）
        
        Args:
            wait_time: 等待服务器启动的时间（秒）
        """
        print(f"正在启动 Sandbox 服务器...")
        print(f"  Docker 镜像: {self.docker_image}")
        print(f"  端口: {self.port}")
        
        # 检查是否已有同名容器在运行
        try:
            check_cmd = f"docker ps -a --filter name={self.container_name} --format '{{{{.Names}}}}'"
            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if self.container_name in result.stdout:
                print(f"  发现已存在的容器 {self.container_name}，正在删除...")
                subprocess.run(
                    f"docker rm -f {self.container_name}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        except Exception as e:
            print(f"  检查已存在容器时出错: {e}")
        
        # 构建 Docker 命令
        cmd = [
            "docker", "run",
            "-d",  # 后台运行
            "--name", self.container_name,
            "-p", f"{self.port}:{self.port}",
            self.docker_image
        ]
        
        try:
            # 启动容器
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            container_id = result.stdout.strip()
            print(f"  ✓ 容器已启动: {container_id[:12]}")
            
            # 等待服务器启动
            print(f"  等待服务器启动（{wait_time}秒）...")
            time.sleep(wait_time)
            
            # 检查服务器是否启动成功
            if self.check_server():
                print(f"✓ Sandbox 服务器启动成功: {self.base_url}")
                print(f"✓ 并发workers数量: {self.max_workers}")
            else:
                print("✗ Sandbox 服务器启动失败")
                self.stop_server()
                raise RuntimeError("Sandbox 服务器启动失败")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Docker 启动失败: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            raise RuntimeError(f"无法启动 Sandbox 容器: {e}")
        except Exception as e:
            print(f"✗ 启动 Sandbox 时出错: {e}")
            raise
    
    def stop_server(self):
        """停止 Sandbox 服务器"""
        print(f"正在停止 Sandbox 服务器...")
        
        try:
            # 停止并删除容器
            subprocess.run(
                f"docker rm -f {self.container_name}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            print(f"✓ Sandbox 服务器已停止")
        except subprocess.TimeoutExpired:
            print("⚠ 停止容器超时")
        except Exception as e:
            print(f"⚠ 停止容器时出错: {e}")
    
    def check_server(self, max_retries: int = 5) -> bool:
        """
        检查 Sandbox 服务器是否正常运行
        
        Args:
            max_retries: 最大重试次数
            
        Returns:
            服务器是否正常运行
        """
        for i in range(max_retries):
            try:
                # 尝试获取 MBPP 的 prompts（作为健康检查）
                response = requests.post(
                    self.get_prompts_url,
                    json={"dataset": "mbpp", "config": {}},
                    timeout=5
                )
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    def run_code(
        self,
        code: str,
        language: str = 'python',
        timeout: int = 30,
    ) -> dict:
        """
        运行代码
        
        Args:
            code: 要运行的代码
            language: 编程语言
            timeout: 超时时间（秒）
            
        Returns:
            运行结果字典
        """
        payload = {
            'code': code,
            'language': language,
        }
        
        try:
            response = requests.post(
                self.run_code_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'status': 'Error',
                'message': str(e),
                'run_result': None,
            }
    
    def get_prompts(
        self,
        dataset: str,
        config: Optional[dict] = None,
    ) -> List[dict]:
        """
        获取数据集的所有 prompts
        
        Args:
            dataset: 数据集名称（如 'mbpp', 'humaneval'）
            config: 可选配置
            
        Returns:
            prompts 列表
        """
        if config is None:
            config = {}
        
        payload = {
            'dataset': dataset,
            'config': config,
        }
        
        try:
            response = requests.post(
                self.get_prompts_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取 prompts 失败: {e}")
            return []
    
    def submit_completion(
        self,
        dataset: str,
        task_id: str,
        completion: str,
        config: Optional[dict] = None,
        timeout: int = 30,
    ) -> dict:
        """
        提交代码补全并评测
        
        Args:
            dataset: 数据集名称
            task_id: 任务ID
            completion: 模型生成的代码
            config: 可选配置
            timeout: 超时时间（秒）
            
        Returns:
            评测结果字典
        """
        if config is None:
            config = {}
        
        payload = {
            'dataset': dataset,
            'id': str(task_id),
            'completion': completion,
            'config': config,
        }
        
        try:
            response = requests.post(
                self.submit_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                'id': str(task_id),
                'accepted': False,
                'error': str(e),
            }
    
    def _submit_with_index(
        self,
        index: int,
        dataset: str,
        task_id: str,
        completion: str,
        config: Optional[dict],
        timeout: int,
    ) -> tuple:
        """
        提交代码补全（带索引）
        
        Returns:
            (索引, 评测结果)
        """
        result = self.submit_completion(
            dataset=dataset,
            task_id=task_id,
            completion=completion,
            config=config,
            timeout=timeout
        )
        return (index, result)
    
    def batch_evaluate(
        self,
        dataset: str,
        task_ids: List[str],
        completions: List[str],
        config: Optional[dict] = None,
        show_progress: bool = True,
        max_workers: Optional[int] = None,
        timeout: int = 30,
    ) -> List[dict]:
        """
        批量评测代码
        
        Args:
            dataset: 数据集名称
            task_ids: 任务ID列表
            completions: 代码补全列表
            config: 可选配置
            show_progress: 是否显示进度
            max_workers: 最大并发worker数量
            timeout: 超时时间（秒）
            
        Returns:
            评测结果列表
        """
        if len(task_ids) != len(completions):
            raise ValueError(
                f"task_ids 和 completions 长度不匹配: {len(task_ids)} vs {len(completions)}"
            )
        
        total = len(task_ids)
        workers = max_workers if max_workers is not None else self.max_workers
        
        if config is None:
            config = {}
        
        print(f"使用 {workers} 个并发workers评测 {total} 个代码样本")
        
        # 初始化结果列表
        results = [None] * total
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    self._submit_with_index,
                    i,
                    dataset,
                    task_id,
                    completion,
                    config,
                    timeout
                ): i for i, (task_id, completion) in enumerate(zip(task_ids, completions))
            }
            
            # 使用tqdm显示进度
            if show_progress:
                pbar = tqdm(total=total, desc="代码评测", unit="samples")
            
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
    # 创建评测器
    evaluator = SandboxEvaluator(
        host='127.0.0.1',
        port=8080,
        max_workers=16,
    )
    
    # 检查服务器
    print("检查 Sandbox 服务器...")
    if evaluator.check_server():
        print("✓ Sandbox 服务器运行正常")
    else:
        print("✗ 无法连接到 Sandbox 服务器")
        print("\n请先启动 Sandbox 服务器:")
        print("docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609")
        exit(1)
    
    # 测试运行代码
    print("\n测试运行代码...")
    code = 'print("Hello, Sandbox!")'
    result = evaluator.run_code(code, language='python')
    print(f"运行结果: {result['run_result']['stdout']}")
    
    # 测试获取 prompts
    print("\n测试获取 MBPP prompts...")
    prompts = evaluator.get_prompts('mbpp')
    print(f"获取了 {len(prompts)} 个 prompts")
    
    # 测试提交评测
    if prompts:
        print("\n测试提交评测...")
        first_prompt = prompts[0]
        print(f"题目 {first_prompt['id']}: {first_prompt['prompt'][:50]}...")
        
        # 提交一个简单的代码
        completion = """
def remove_Occ(s, char):
    first_occ = s.find(char)
    last_occ = s.rfind(char)
    
    if first_occ == -1 or first_occ == last_occ:
        return s
    
    # Remove the first occurrence
    s = s[:first_occ] + s[first_occ + 1:]
    
    # Adjust the index for the last occurrence
    last_occ -= 1
    
    # Remove the last occurrence
    s = s[:last_occ] + s[last_occ + 1:]
    
    return s
"""
        result = evaluator.submit_completion('mbpp', first_prompt['id'], completion)
        print(f"评测结果: {'✓ 通过' if result['accepted'] else '✗ 失败'}")


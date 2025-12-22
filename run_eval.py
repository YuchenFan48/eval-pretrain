"""
主评测脚本：运行benchmark评测
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

from config import BENCHMARK_CONFIG, SGLANG_CONFIG, OPENAI_CONFIG, PPL_DATASET_CONFIG, SANDBOX_CONFIG
from benchmarks.data_loader import load_benchmark_data, load_jsonl
from prompts.prompt_templates import build_prompt
from benchmarks.evaluators import evaluate_benchmark
from utils.sglang_inference import SGLangInference
from utils.openai_inference import OpenAIInference
from utils.sandbox_evaluator import SandboxEvaluator
from utils.local_code_executor import LocalCodeExecutor
from utils.code_extractor import extract_code_by_language
import numpy as np


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行benchmark评测')
    
    parser.add_argument(
        '--backend',
        type=str,
        default='sglang',
        choices=['sglang', 'openai'],
        help='推理后端：sglang (本地模型) 或 openai (OpenAI API)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=False,
        help='模型文件路径（使用sglang时必需）'
    )
    
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=None,
        choices=list(BENCHMARK_CONFIG.keys()) + ['all'],
        help='要评测的benchmark，可以指定多个，或使用"all"评测所有'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='结果输出目录'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=SGLANG_CONFIG['host'],
        help='SGLang服务器地址'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=SGLANG_CONFIG['port'],
        help='SGLang服务器端口'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='采样温度'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='最大生成token数'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批处理大小（已弃用，使用--max-workers代替）'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=32,
        help='最大并发worker数量（推荐：32-64）'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='每个benchmark的最大样本数（用于快速测试）'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='分块处理大小（用于超大数据集，避免内存问题）'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='单个请求的超时时间（秒）'
    )

    parser.add_argument(
        '--sample-k',
        type=int,
        default=1,
        help='每个prompt采样次数（>1时会重复生成并对结果求均值）'
    )
    
    parser.add_argument(
        '--no-sandbox-server',
        action='store_true',
        help='不启动Sandbox服务器（假设服务器已在运行）'
    )
    
    # OpenAI相关参数
    parser.add_argument(
        '--openai-api-key',
        type=str,
        default=OPENAI_CONFIG['api_key'],
        help='OpenAI API密钥（使用openai后端时必需）'
    )
    
    parser.add_argument(
        '--openai-base-url',
        type=str,
        default=OPENAI_CONFIG['base_url'],
        help='OpenAI API基础URL（可选，用于自定义API端点）'
    )
    
    parser.add_argument(
        '--openai-model',
        type=str,
        default=OPENAI_CONFIG['model'],
        help='OpenAI模型名称（如 gpt-4, gpt-3.5-turbo）'
    )
    
    # PPL评测参数
    parser.add_argument(
        '--ppl-datasets',
        type=str,
        nargs='+',
        default=None,
        choices=list(PPL_DATASET_CONFIG.keys()) + ['all'],
        help='要评测PPL的数据集，可以指定多个，或使用"all"评测所有'
    )
    
    parser.add_argument(
        '--ppl-mode',
        type=str,
        default='answer-only',
        choices=['answer-only', 'full-text'],
        help='PPL计算模式：answer-only（只计算answer部分）或 full-text（计算完整prompt+answer）'
    )
    
    return parser.parse_args()


def generate_predictions_k(
    prompts: List[str],
    inference,
    args: argparse.Namespace,
    stage_label: str = "",
) -> List[List[str]]:
    """
    对同一批prompts进行k次采样生成
    """
    sample_k = max(1, getattr(args, 'sample_k', 1))
    all_predictions = []

    for sample_idx in range(sample_k):
        if sample_k > 1:
            label = f"（{stage_label}）" if stage_label else ""
            print(f"   采样第 {sample_idx + 1}/{sample_k} 次{label}...")

        if args.chunk_size:
            predictions = inference.batch_generate_chunked(
                prompts=prompts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                chunk_size=args.chunk_size,
                max_workers=args.max_workers,
                show_progress=True,
                timeout=args.timeout,
            )
        else:
            predictions = inference.batch_generate(
                prompts=prompts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_workers=args.max_workers,
                show_progress=True,
                timeout=args.timeout,
            )

        all_predictions.append(predictions)

    return all_predictions


def run_benchmark_eval(
    benchmark_name: str,
    inference,  # BaseInference类型（可以是SGLangInference或OpenAIInference）
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    运行单个benchmark的评测
    
    Args:
        benchmark_name: benchmark名称
        inference: SGLang推理器
        args: 命令行参数
        
    Returns:
        评测结果字典
    """
    print(f"\n{'='*60}")
    print(f"开始评测: {benchmark_name}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print("1. 加载数据...")
    config = BENCHMARK_CONFIG[benchmark_name]
    data = load_benchmark_data(benchmark_name, config)
    
    # 限制样本数量（用于快速测试）
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"   ✓ 加载了 {len(data)} 条数据")
    sample_k = max(1, getattr(args, 'sample_k', 1))
    
    # 2. 构建prompts
    print("2. 构建prompts...")
    prompts = []
    for item in data:
        try:
            prompt = build_prompt(benchmark_name, item)
            prompts.append(prompt)
        except Exception as e:
            print(f"   ✗ 构建prompt失败: {e}")
            prompts.append("")
    
    print(f"   ✓ 构建了 {len(prompts)} 个prompts")

    # 3. 批量推理（并发）
    if sample_k > 1:
        print(f"3. 批量推理（使用 {args.max_workers} 个并发workers，每个prompt采样 {sample_k} 次）...")
    else:
        print(f"3. 批量推理（使用 {args.max_workers} 个并发workers）...")

    predictions_samples = generate_predictions_k(prompts, inference, args)
    print(f"   ✓ 完成 {len(predictions_samples[0])} 个预测")
    
    # 4. 提取参考答案
    print("4. 提取参考答案...")
    references = []
    for item in data:
        if benchmark_name == 'bbh':
            references.append(item['target'])
        elif benchmark_name == 'mmlu-pro':
            references.append(item['answer'])
        elif benchmark_name == 'mmlu-redux':
            references.append(item['answer'])
        elif benchmark_name == 'gpqa':
            references.append(item['answer'])
        elif benchmark_name == 'gsm8k':
            references.append(item['answer'])
        elif benchmark_name == 'math':
            references.append(item.get('answer', item.get('solution', '')))
        elif benchmark_name == 'mmlu':
            references.append(item['answer'])
        elif benchmark_name == 'supergpqa':
            references.append(item['answer_letter'])
        else:
            references.append("")
    
    print(f"   ✓ 提取了 {len(references)} 个参考答案")
    
    # 5. 评估
    print("5. 评估结果...")
    if sample_k == 1:
        predictions = predictions_samples[0]
        accuracy, details = evaluate_benchmark(
            benchmark_name=benchmark_name,
            predictions=predictions,
            references=references,
            data_items=data
        )

        correct_count = sum(1 for d in details if d['correct'])
        print(f"   ✓ 准确率: {accuracy:.2f}%")

        # 6. 返回结果
        result = {
            'benchmark': benchmark_name,
            'total': len(data),
            'correct': correct_count,
            'accuracy': accuracy,
            'details': details,
            'timestamp': datetime.now().isoformat(),
        }
    else:
        accuracy_samples = []
        correct_samples = []
        details_samples = []

        for predictions in predictions_samples:
            accuracy, details = evaluate_benchmark(
                benchmark_name=benchmark_name,
                predictions=predictions,
                references=references,
                data_items=data
            )
            correct_count = sum(1 for d in details if d['correct'])
            accuracy_samples.append(accuracy)
            correct_samples.append(correct_count)
            details_samples.append(details)

        avg_accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0
        avg_correct = float(np.mean(correct_samples)) if correct_samples else 0.0
        print(f"   ✓ 平均准确率: {avg_accuracy:.2f}% (采样{sample_k}次)")

        result = {
            'benchmark': benchmark_name,
            'total': len(data),
            'correct': avg_correct,
            'accuracy': avg_accuracy,
            'details': details_samples[0] if details_samples else [],
            'details_samples': details_samples,
            'accuracy_samples': accuracy_samples,
            'correct_samples': correct_samples,
            'sample_k': sample_k,
            'timestamp': datetime.now().isoformat(),
        }
    
    return result


def save_results(results: List[Dict[str, Any]], args: argparse.Namespace):
    """
    保存评测结果
    
    Args:
        results: 评测结果列表
        args: 命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 根据后端类型生成模型名称
    if args.backend == 'sglang':
        model_name = os.path.basename(args.model_path)
    elif args.backend == 'openai':
        # 使用 OpenAI 模型名称，移除路径中的斜杠
        model_name = args.openai_model.replace('/', '_')
    else:
        model_name = 'unknown'
    
    output_file = os.path.join(args.output_dir, f'{model_name}_{timestamp}.json')
    
    # 保存完整结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 完整结果已保存到: {output_file}")
    
    # 保存汇总结果
    summary_file = os.path.join(args.output_dir, f'{model_name}_{timestamp}_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        if args.backend == 'sglang':
            f.write(f"模型: {args.model_path}\n")
        elif args.backend == 'openai':
            f.write(f"模型: {args.openai_model}\n")
            if args.openai_base_url:
                f.write(f"API地址: {args.openai_base_url}\n")
        f.write(f"后端: {args.backend}\n")
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            if result.get('type') == 'crux-o':
                # CRUX-O 特殊格式
                output_acc = result['direct_output']['accuracy']
                input_acc = result['direct_input']['accuracy']
                avg_acc = result['average_accuracy']
                sample_k = result['direct_output'].get('sample_k', 1)
                if sample_k > 1:
                    f.write(f"{result['benchmark']:15} | "
                           f"Output准确率: {output_acc:6.2f}% | "
                           f"Input准确率: {input_acc:6.2f}% | "
                           f"平均: {avg_acc:6.2f}% | "
                           f"采样: {sample_k}\n")
                else:
                    f.write(f"{result['benchmark']:15} | "
                           f"Output准确率: {output_acc:6.2f}% | "
                           f"Input准确率: {input_acc:6.2f}% | "
                           f"平均: {avg_acc:6.2f}%\n")
            elif 'correct' in result:
                # 常规 benchmark
                if result.get('sample_k', 1) > 1:
                    f.write(f"{result['benchmark']:15} | "
                           f"准确率: {result['accuracy']:6.2f}% | "
                           f"平均正确/总数: {result['correct']:6.1f}/{result['total']:4} | "
                           f"采样: {result['sample_k']}\n")
                else:
                    f.write(f"{result['benchmark']:15} | "
                           f"准确率: {result['accuracy']:6.2f}% | "
                           f"正确/总数: {result['correct']:4}/{result['total']:4}\n")
            elif 'passed' in result:
                # 代码生成 benchmark
                if result.get('sample_k', 1) > 1:
                    f.write(f"{result['benchmark']:15} | "
                           f"通过率: {result['accuracy']:6.2f}% | "
                           f"平均通过/总数: {result['passed']:6.1f}/{result['total']:4} | "
                           f"采样: {result['sample_k']}\n")
                else:
                    f.write(f"{result['benchmark']:15} | "
                           f"通过率: {result['accuracy']:6.2f}% | "
                           f"通过/总数: {result['passed']:4}/{result['total']:4}\n")
        
        # 计算平均准确率
        total_accuracy = 0.0
        for r in results:
            if r.get('type') == 'crux-o':
                total_accuracy += r['average_accuracy']
            else:
                total_accuracy += r['accuracy']
        avg_accuracy = total_accuracy / len(results)
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"平均准确率: {avg_accuracy:.2f}%\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ 汇总结果已保存到: {summary_file}")


def save_ppl_results(results: List[Dict[str, Any]], args: argparse.Namespace):
    """
    保存PPL评测结果
    
    Args:
        results: PPL评测结果列表
        args: 命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 根据后端类型生成模型名称
    if args.backend == 'sglang':
        model_name = os.path.basename(args.model_path)
    elif args.backend == 'openai':
        model_name = args.openai_model.replace('/', '_')
    else:
        model_name = 'unknown'
    
    output_file = os.path.join(args.output_dir, f'{model_name}_ppl_{timestamp}.json')
    
    # 保存完整结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ PPL完整结果已保存到: {output_file}")
    
    # 保存汇总结果
    summary_file = os.path.join(args.output_dir, f'{model_name}_ppl_{timestamp}_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        if args.backend == 'sglang':
            f.write(f"模型: {args.model_path}\n")
        elif args.backend == 'openai':
            f.write(f"模型: {args.openai_model}\n")
            if args.openai_base_url:
                f.write(f"API地址: {args.openai_base_url}\n")
        f.write(f"后端: {args.backend}\n")
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            ppl_mode_str = result.get('ppl_mode', 'answer-only')
            mode_label = "答" if ppl_mode_str == 'answer-only' else "全"
            if result['avg_ppl'] is not None:
                f.write(f"{result['dataset']:20} [{mode_label}] | "
                       f"平均PPL: {result['avg_ppl']:8.2f} | "
                       f"中位数PPL: {result['median_ppl']:8.2f} | "
                       f"有效/总数: {result['valid']:4}/{result['total']:4}\n")
            else:
                f.write(f"{result['dataset']:20} [{mode_label}] | PPL计算失败\n")
        
        # 计算平均PPL
        valid_results = [r for r in results if r['avg_ppl'] is not None]
        if valid_results:
            avg_ppl = sum(r['avg_ppl'] for r in valid_results) / len(valid_results)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"平均PPL: {avg_ppl:.2f}\n")
            f.write("=" * 80 + "\n")
    
    print(f"✓ PPL汇总结果已保存到: {summary_file}")


def extract_prompt_and_answer(item: Dict[str, Any], dataset_name: str) -> tuple:
    """
    从数据项中提取prompt和answer
    
    Args:
        item: 数据项
        dataset_name: 数据集名称
        
    Returns:
        (prompt, answer) 元组
    """
    # 获取source，如果没有就用dataset_name
    source = item.get('source', dataset_name)
    
    # 根据不同的数据集提取prompt和answer
    if source == 'math' or 'MATH' in dataset_name.upper():
        # MATH数据集：problem是prompt，solution是answer
        prompt = item.get('problem', '')
        answer = item.get('solution', '')
        
    elif source == 'gsm8k' or 'GSM8K' in dataset_name.upper():
        # GSM8K数据集：question是prompt，answer是answer
        prompt = item.get('question', '')
        answer = item.get('answer', '')
        
    elif source == 'gpqa' or 'GPQA' in dataset_name.upper():
        # GPQA数据集：Question是prompt，Correct Answer是answer
        prompt = item.get('Question', item.get('question', ''))
        answer = item.get('Correct Answer', item.get('answer', ''))
        
    elif source == 'mmlu-pro' or 'MMLU-PRO' in dataset_name.upper():
        # MMLU-PRO数据集：question是prompt，对应的选项文本是answer
        prompt = item.get('question', '')
        # answer是选项的实际文本，不是选项字母
        options = item.get('options', [])
        answer_index = item.get('answer_index', -1)
        if 0 <= answer_index < len(options):
            answer = options[answer_index]
        else:
            answer = item.get('answer', '')
    
    elif source == 'mmlu-redux' or 'MMLU-REDUX' in dataset_name.upper():
        # MMLU-Redux数据集：类似MMLU-Pro
        prompt = item.get('question', item.get('problem', ''))
        answer = item.get('answer', '')
    
    elif source == 'supergpqa' or 'SUPERGPQA' in dataset_name.upper():
        # SuperGPQA数据集：类似GPQA
        prompt = item.get('Question', item.get('question', ''))
        answer = item.get('Correct Answer', item.get('answer', ''))
            
    elif source == 'mbpp' or (dataset_name.upper() == 'MBPP_PPL'):
        # MBPP数据集：任务描述是prompt（从text中提取），code是answer
        text = item.get('text', '')
        code = item.get('code', '')
        # 从text中提取prompt部分（去掉code部分）
        if code and code in text:
            prompt = text.replace(code, '').strip()
        else:
            # 如果找不到code，尝试从text开头提取
            prompt = text.split('\n')[0] if text else ''
        answer = code
    
    elif source == 'mbppplus' or 'MBPPPLUS' in dataset_name.upper():
        # MBPP+数据集：直接有prompt字段，code是answer
        prompt = item.get('prompt', item.get('text', ''))
        answer = item.get('code', '')
        
    elif source == 'bbh' or 'BBH' in dataset_name.upper():
        # BBH数据集：input是prompt，target是answer
        prompt = item.get('input', '')
        answer = item.get('target', '')
        
    elif source == 'crux-o' or 'CRUX' in dataset_name.upper():
        # CRUX数据集：code + input是prompt，output是answer
        code = item.get('code', '')
        input_data = item.get('input', '')
        prompt = f"{code} {input_data}".strip()
        answer = item.get('output', '')
        
    elif 'humaneval' in source.lower() or 'HUMANEVAL' in dataset_name.upper():
        # HumanEval数据集：通常有prompt字段和canonical_solution字段
        prompt = item.get('prompt', '')
        answer = item.get('canonical_solution', item.get('code', ''))
        
    else:
        # 默认情况：尝试从text字段中分离
        text = item.get('text', '')
        # 移除 <|begin_text|> 和 <|end_text|> 标记
        text = text.replace('<|begin_text|>', '').replace('<|end_text|>', '').strip()
        
        # 如果有明确的字段，使用它们
        if 'question' in item and 'answer' in item:
            prompt = item['question']
            answer = item['answer']
        elif 'problem' in item and 'solution' in item:
            prompt = item['problem']
            answer = item['solution']
        else:
            # 否则整个text作为prompt，answer为空（这会导致计算失败）
            prompt = text
            answer = ''
            print(f"   ⚠ 警告: 无法识别数据集 {dataset_name} 的prompt/answer格式")
    
    return prompt, answer


def run_ppl_eval(
    dataset_name: str,
    inference,  # BaseInference类型
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    运行单个数据集的PPL评测
    
    Args:
        dataset_name: PPL数据集名称
        inference: 推理器
        args: 命令行参数
        
    Returns:
        PPL评测结果字典
    """
    print(f"\n{'='*60}")
    print(f"开始PPL评测: {dataset_name}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print("1. 加载数据...")
    config = PPL_DATASET_CONFIG[dataset_name]
    data = load_jsonl(config['path'])
    
    # 限制样本数量（用于快速测试）
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"   ✓ 加载了 {len(data)} 条数据")
    
    # 2. 提取prompt和answer
    print("2. 提取prompt和answer...")
    prompts = []
    answers = []
    for item in data:
        prompt, answer = extract_prompt_and_answer(item, dataset_name)
        if prompt and answer:  # 只保留有效的样本
            prompts.append(prompt)
            answers.append(answer)
    
    print(f"   ✓ 提取了 {len(prompts)} 个有效的prompt/answer对")
    
    # 3. 计算PPL（根据模式选择）
    ppl_mode = getattr(args, 'ppl_mode', 'answer-only')
    
    if ppl_mode == 'answer-only':
        print(f"3. 计算PPL（使用 {args.max_workers} 个并发workers，只计算answer部分）...")
        
        # 检查推理器是否支持PPL计算
        if not hasattr(inference, 'batch_compute_ppl_with_prompt'):
            print("   ✗ 当前推理后端不支持带prompt的PPL计算")
            return {
                'dataset': dataset_name,
                'total': len(prompts),
                'avg_ppl': None,
                'median_ppl': None,
                'ppl_mode': ppl_mode,
                'timestamp': datetime.now().isoformat(),
            }
        
        logprob_results = inference.batch_compute_ppl_with_prompt(
            prompts=prompts,
            answers=answers,
            max_workers=args.max_workers,
            show_progress=True,
            timeout=args.timeout,
        )
    else:  # full-text
        print(f"3. 计算PPL（使用 {args.max_workers} 个并发workers，计算完整prompt+answer）...")
        
        # 拼接完整文本
        full_texts = [p + a for p, a in zip(prompts, answers)]
        
        logprob_results = inference.batch_compute_ppl(
            texts=full_texts,
            max_workers=args.max_workers,
            show_progress=True,
            timeout=args.timeout,
        )
    
    print(f"   ✓ 完成 {len(logprob_results)} 个文本的PPL计算（模式：{ppl_mode}）")
    
        # 4. 计算PPL统计信息
    print("4. 计算PPL统计信息...")
    ppls = []

    # 用于全局 token 级别的 PPL 统计
    total_log_prob = 0.0
    total_tokens = 0

    # 调试：检查前几个结果
    for idx in range(min(2, len(logprob_results))):
        result = logprob_results[idx]
        print(f"\n   调试 - 样本 {idx}:")
        print(f"      result keys: {list(result.keys())}")
        print(f"      logprobs: {result.get('logprobs', [])}")
        print(f"      answer_tokens: {result.get('answer_tokens', 0)}")
        print(f"      prompt_tokens: {result.get('prompt_tokens', 0)}")

    for idx, logprob_result in enumerate(logprob_results):
        logprobs = logprob_result.get('logprobs', [])

        # 关键改进：排除第一个 token（对齐 Transformers 的逻辑）
        # 因为第一个 token 没有前文可以预测，不应参与 PPL 计算
        # 参考：labels[:, 0] = -100 和 shift 操作
        if logprobs and len(logprobs) > 1:
            # 跳过第一个 token 的 logprob
            logprobs_to_use = logprobs[1:]
        else:
            logprobs_to_use = logprobs

        # 1) 样本级 PPL：每个样本独立计算
        if logprobs_to_use:
            valid_logprobs = []
            for lp in logprobs_to_use:
                if lp is not None and isinstance(lp, (int, float)):
                    valid_logprobs.append(lp)

            if valid_logprobs:
                # PPL = exp(-mean(log_probs))
                # 对齐 Transformers: ppl_value = torch.exp(-avg_log_prob).item()
                avg_logprob = np.mean(valid_logprobs)
                ppl = np.exp(-avg_logprob)
                ppls.append(ppl)
            else:
                ppls.append(None)
        else:
            ppls.append(None)

        # 2) 全局 token 级 PPL：累加所有 token 的 log_prob
        # 对齐 Transformers: total_log_prob += valid_log_probs.sum().item()
        for lp in logprobs_to_use:
            if lp is not None and isinstance(lp, (int, float)):
                total_log_prob += lp
                total_tokens += 1

    # 过滤掉None值（样本级 PPL）
    valid_ppls = [p for p in ppls if p is not None]

    # 样本级统计
    if valid_ppls:
        avg_ppl = np.mean(valid_ppls)
        median_ppl = np.median(valid_ppls)
        mode_text = "仅answer部分" if ppl_mode == 'answer-only' else "完整prompt+answer"
        print(f"   ✓ 平均PPL（按样本，{mode_text}）: {avg_ppl:.2f}")
        print(f"   ✓ 中位数PPL（按样本，{mode_text}）: {median_ppl:.2f}")
    else:
        avg_ppl = None
        median_ppl = None
        print("   ✗ 无有效的样本级PPL结果")

    # 全局 token 级 PPL（对方写法）
    if total_tokens == 0:
        global_ppl = float('inf')
        print("   ✗ 全局PPL：无有效token，设为 inf")
    else:
        avg_neg_log_prob = - total_log_prob / total_tokens
        # 防止exp溢出，和对方一样加一个 clamp
        clamped_avg = min(avg_neg_log_prob, 700.0)
        global_ppl = float(np.exp(clamped_avg))
        mode_text = "仅answer部分" if ppl_mode == 'answer-only' else "完整prompt+answer"
        print(f"   ✓ 全局PPL（按token加权，{mode_text}）: {global_ppl:.2f}")

    
    # 5. 返回结果
    result = {
        'dataset': dataset_name,
        'source': config['source'],
        'total': len(prompts),
        'valid': len(valid_ppls),
        'avg_ppl': float(avg_ppl) if avg_ppl is not None else None,
        'median_ppl': float(median_ppl) if median_ppl is not None else None,
        'global_ppl': float(global_ppl) if global_ppl is not None else None,
        'ppl_mode': ppl_mode,
        'ppls': ppls,
        'timestamp': datetime.now().isoformat(),
    }
    
    return result


def run_code_eval(
    benchmark_name: str,
    inference,  # BaseInference类型
    code_evaluator,  # SandboxEvaluator 或 LocalCodeExecutor
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    运行代码生成 benchmark 的评测
    
    Args:
        benchmark_name: benchmark名称
        inference: 推理器
        code_evaluator: 代码评测器（Sandbox 或本地执行器）
        args: 命令行参数
        
    Returns:
        评测结果字典
    """
    print(f"\n{'='*60}")
    print(f"开始代码评测: {benchmark_name}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    print("1. 加载数据...")
    config = BENCHMARK_CONFIG[benchmark_name]
    data = load_benchmark_data(benchmark_name, config)
    
    # 限制样本数量（用于快速测试）
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"   ✓ 加载了 {len(data)} 条数据")
    
    sample_k = args.sample_k
    # ============ CRUX-O 特殊处理 ============
    if benchmark_name == 'crux-o':
        # CRUX-O 需要每条数据跑两种 prompt
        task_types = ['direct_output', 'direct_input']
        all_results = {}
        
        for task_type in task_types:
            print(f"\n--- 任务类型: {task_type} ---")
            
            # 2. 构建prompts
            print("2. 构建prompts...")
            prompts = []
            
            for item in data:
                try:
                    prompt = build_prompt(benchmark_name, item, task_type=task_type)
                    prompts.append(prompt)
                except Exception as e:
                    print(f"   ✗ 构建prompt失败: {e}")
                    prompts.append("")
            
            print(f"   ✓ 构建了 {len(prompts)} 个prompts")
            
            # 3. 批量推理
            if sample_k > 1:
                print(f"3. 批量推理（使用 {args.max_workers} 个并发workers，每个prompt采样 {sample_k} 次）...")
            else:
                print(f"3. 批量推理（使用 {args.max_workers} 个并发workers）...")

            predictions_samples = generate_predictions_k(
                prompts,
                inference,
                args,
                stage_label=task_type,
            )
            print(f"   ✓ 完成 {len(predictions_samples[0])} 个预测")

            # 4. 评估结果
            print("4. 评估结果...")
            if sample_k == 1:
                predictions = predictions_samples[0]
                correct_count = 0
                details = []

                for idx, (item, pred) in enumerate(zip(data, predictions)):
                    # 提取 [ANSWER] 和 [/ANSWER] 之间的内容
                    answer_text = pred
                    if '[ANSWER]' in pred and '[/ANSWER]' in pred:
                        start_idx = pred.find('[ANSWER]') + len('[ANSWER]')
                        end_idx = pred.find('[/ANSWER]')
                        answer_text = pred[start_idx:end_idx].strip()

                    # 根据任务类型提取答案
                    if task_type == 'direct_output':
                        # 从 "assert f(...) == xxx" 中提取 xxx
                        if '==' in answer_text:
                            predicted = answer_text.split('==')[-1].strip()
                        else:
                            predicted = answer_text

                        expected = item.get('output', '')

                    else:  # direct_input
                        # 从 "assert f(xxx) == ..." 中提取 xxx
                        if 'assert f(' in answer_text and ') ==' in answer_text:
                            start = answer_text.find('assert f(') + len('assert f(')
                            end = answer_text.find(') ==')
                            predicted = answer_text[start:end].strip()
                        else:
                            predicted = answer_text

                        expected = item.get('input', '')

                    # 比较结果
                    is_correct = (predicted == expected)
                    if is_correct:
                        correct_count += 1

                    details.append({
                        'task_id': item.get('id', f'sample_{idx}'),
                        'task_type': task_type,
                        'model_response': pred,  # 保存模型完整回复
                        'extracted_answer': answer_text,  # 提取的答案文本
                        'prediction': predicted,  # 解析后的预测值
                        'expected': expected,  # 期望值
                        'correct': is_correct,
                    })

                accuracy = (correct_count / len(data) * 100) if len(data) > 0 else 0.0
                print(f"   ✓ {task_type} 准确率: {accuracy:.2f}% ({correct_count}/{len(data)})")

                all_results[task_type] = {
                    'total': len(data),
                    'correct': correct_count,
                    'accuracy': accuracy,
                    'details': details,
                }
            else:
                accuracy_samples = []
                correct_samples = []
                details_samples = []

                for predictions in predictions_samples:
                    correct_count = 0
                    details = []

                    for idx, (item, pred) in enumerate(zip(data, predictions)):
                        # 提取 [ANSWER] 和 [/ANSWER] 之间的内容
                        answer_text = pred
                        if '[ANSWER]' in pred and '[/ANSWER]' in pred:
                            start_idx = pred.find('[ANSWER]') + len('[ANSWER]')
                            end_idx = pred.find('[/ANSWER]')
                            answer_text = pred[start_idx:end_idx].strip()

                        # 根据任务类型提取答案
                        if task_type == 'direct_output':
                            if '==' in answer_text:
                                predicted = answer_text.split('==')[-1].strip()
                            else:
                                predicted = answer_text

                            expected = item.get('output', '')

                        else:
                            if 'assert f(' in answer_text and ') ==' in answer_text:
                                start = answer_text.find('assert f(') + len('assert f(')
                                end = answer_text.find(') ==')
                                predicted = answer_text[start:end].strip()
                            else:
                                predicted = answer_text

                            expected = item.get('input', '')

                        is_correct = (predicted == expected)
                        if is_correct:
                            correct_count += 1

                        details.append({
                            'task_id': item.get('id', f'sample_{idx}'),
                            'task_type': task_type,
                            'model_response': pred,
                            'extracted_answer': answer_text,
                            'prediction': predicted,
                            'expected': expected,
                            'correct': is_correct,
                        })

                    accuracy = (correct_count / len(data) * 100) if len(data) > 0 else 0.0
                    accuracy_samples.append(accuracy)
                    correct_samples.append(correct_count)
                    details_samples.append(details)

                avg_accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0
                avg_correct = float(np.mean(correct_samples)) if correct_samples else 0.0
                print(f"   ✓ {task_type} 平均准确率: {avg_accuracy:.2f}% (采样{sample_k}次)")

                all_results[task_type] = {
                    'total': len(data),
                    'correct': avg_correct,
                    'accuracy': avg_accuracy,
                    'details': details_samples[0] if details_samples else [],
                    'details_samples': details_samples,
                    'accuracy_samples': accuracy_samples,
                    'correct_samples': correct_samples,
                    'sample_k': sample_k,
                }
        
        # 5. 计算平均准确率
        avg_accuracy = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
        if sample_k > 1:
            print(f"\n   ✓ 平均准确率: {avg_accuracy:.2f}% (采样{sample_k}次)")
        else:
            print(f"\n   ✓ 平均准确率: {avg_accuracy:.2f}%")
        
        # 6. 返回 CRUX-O 结果
        result = {
            'benchmark': benchmark_name,
            'type': 'crux-o',
            'total': len(data),
            'direct_output': all_results['direct_output'],
            'direct_input': all_results['direct_input'],
            'average_accuracy': avg_accuracy,
            'sample_k': sample_k,
            'timestamp': datetime.now().isoformat(),
        }
        
        return result
    
    # ============ 其他代码生成 benchmark（标准流程）============
    # 2. 构建prompts 和收集测试数据
    print("2. 构建prompts...")
    prompts = []
    task_ids = []
    test_data_list = []  # 用于本地执行器
    
    for item in data:
        try:
            prompt = build_prompt(benchmark_name, item)
            prompts.append(prompt)
            
            # 获取task_id
            task_id = item.get('task_id', item.get('id', ''))
            task_ids.append(str(task_id))
            
            # 收集测试数据（用于本地执行器）
            test_data = {
                'test_list': item.get('test_list', []),
                'test': item.get('test', ''),
                'entry_point': item.get('entry_point', ''),
            }
            # print(f"   ✓ 收集了 {len(test_data.get('test_list', []))} 个测试数据")
            test_data_list.append(test_data)
            
        except Exception as e:
            print(f"   ✗ 构建prompt失败: {e}")
            prompts.append("")
            task_ids.append("")
            test_data_list.append({})
    
    print(f"   ✓ 构建了 {len(prompts)} 个prompts")
    
    # 3. 批量推理（生成代码）
    if sample_k > 1:
        print(f"3. 批量推理（使用 {args.max_workers} 个并发workers，每个prompt采样 {sample_k} 次）...")
    else:
        print(f"3. 批量推理（使用 {args.max_workers} 个并发workers）...")

    predictions_samples = generate_predictions_k(prompts, inference, args)
    print(f"   ✓ 完成 {len(predictions_samples[0])} 个预测")

    # 4. 从预测中提取代码 + 评测
    print("4. 从预测中提取代码...")
    language = config.get('language', 'python')

    if sample_k == 1:
        predictions = predictions_samples[0]
        completions = []

        for pred in predictions:
            code = extract_code_by_language(pred, language)
            completions.append(code)

        print(f"   ✓ 提取了 {len(completions)} 段代码")

        # 5. 评测代码
        if isinstance(code_evaluator, LocalCodeExecutor):
            # 使用本地执行器
            print(f"5. 使用本地执行器评测代码（使用 {code_evaluator.max_workers} 个并发workers）...")
            sandbox_dataset = config.get('sandbox_dataset', benchmark_name)

            results = code_evaluator.batch_evaluate(
                dataset=sandbox_dataset,
                task_ids=task_ids,
                completions=completions,
                test_data_list=test_data_list,
                show_progress=True,
                max_workers=code_evaluator.max_workers,
                timeout=code_evaluator.timeout,
            )
        else:
            # 使用 Sandbox
            print(f"5. 使用 Sandbox 评测代码（使用 {SANDBOX_CONFIG['max_workers']} 个并发workers）...")
            sandbox_dataset = config.get('sandbox_dataset', benchmark_name)

            results = code_evaluator.batch_evaluate(
                dataset=sandbox_dataset,
                task_ids=task_ids,
                completions=completions,
                config={},
                show_progress=True,
                max_workers=SANDBOX_CONFIG['max_workers'],
                timeout=SANDBOX_CONFIG['timeout'],
            )

        print(f"   ✓ 完成 {len(results)} 个样本的评测")

        # 6. 统计结果并添加模型回复
        print("6. 统计结果...")
        total = len(results)
        passed = sum(1 for r in results if r.get('accepted', False))
        accuracy = (passed / total * 100) if total > 0 else 0.0

        # 为每个结果添加模型的原始回复和提取的代码
        enhanced_details = []
        for i, result in enumerate(results):
            enhanced_result = result.copy()
            if i < len(predictions):
                enhanced_result['model_response'] = predictions[i]  # 模型完整回复
            if i < len(completions):
                enhanced_result['extracted_code'] = completions[i]  # 提取的代码
            enhanced_details.append(enhanced_result)

        print(f"   ✓ 通过率: {accuracy:.2f}% ({passed}/{total})")

        # 7. 返回结果
        result = {
            'benchmark': benchmark_name,
            'type': 'code',
            'total': total,
            'passed': passed,
            'accuracy': accuracy,
            'details': enhanced_details,
            'timestamp': datetime.now().isoformat(),
        }
    else:
        accuracy_samples = []
        passed_samples = []
        details_samples = []
        total = 0

        for predictions in predictions_samples:
            completions = []
            for pred in predictions:
                code = extract_code_by_language(pred, language)
                completions.append(code)

            if isinstance(code_evaluator, LocalCodeExecutor):
                print(f"5. 使用本地执行器评测代码（使用 {code_evaluator.max_workers} 个并发workers）...")
                sandbox_dataset = config.get('sandbox_dataset', benchmark_name)

                results = code_evaluator.batch_evaluate(
                    dataset=sandbox_dataset,
                    task_ids=task_ids,
                    completions=completions,
                    test_data_list=test_data_list,
                    show_progress=True,
                    max_workers=code_evaluator.max_workers,
                    timeout=code_evaluator.timeout,
                )
            else:
                print(f"5. 使用 Sandbox 评测代码（使用 {SANDBOX_CONFIG['max_workers']} 个并发workers）...")
                sandbox_dataset = config.get('sandbox_dataset', benchmark_name)

                results = code_evaluator.batch_evaluate(
                    dataset=sandbox_dataset,
                    task_ids=task_ids,
                    completions=completions,
                    config={},
                    show_progress=True,
                    max_workers=SANDBOX_CONFIG['max_workers'],
                    timeout=SANDBOX_CONFIG['timeout'],
                )

            total = len(results)
            passed = sum(1 for r in results if r.get('accepted', False))
            accuracy = (passed / total * 100) if total > 0 else 0.0

            enhanced_details = []
            for i, result_item in enumerate(results):
                enhanced_result = result_item.copy()
                if i < len(predictions):
                    enhanced_result['model_response'] = predictions[i]
                if i < len(completions):
                    enhanced_result['extracted_code'] = completions[i]
                enhanced_details.append(enhanced_result)

            accuracy_samples.append(accuracy)
            passed_samples.append(passed)
            details_samples.append(enhanced_details)

        avg_accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0
        avg_passed = float(np.mean(passed_samples)) if passed_samples else 0.0
        print(f"6. 统计结果...")
        print(f"   ✓ 平均通过率: {avg_accuracy:.2f}% (采样{sample_k}次)")

        result = {
            'benchmark': benchmark_name,
            'type': 'code',
            'total': total,
            'passed': avg_passed,
            'accuracy': avg_accuracy,
            'details': details_samples[0] if details_samples else [],
            'details_samples': details_samples,
            'accuracy_samples': accuracy_samples,
            'passed_samples': passed_samples,
            'sample_k': sample_k,
            'timestamp': datetime.now().isoformat(),
        }
    
    return result


def print_summary(results: List[Dict[str, Any]]):
    """
    打印评测结果汇总
    
    Args:
        results: 评测结果列表
    """
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)
    
    for result in results:
        if result.get('sample_k', 1) > 1:
            print(f"{result['benchmark']:15} | "
                  f"准确率: {result['accuracy']:6.2f}% | "
                  f"平均正确/总数: {result['correct']:6.1f}/{result['total']:4} | "
                  f"采样: {result['sample_k']}")
        else:
            print(f"{result['benchmark']:15} | "
                  f"准确率: {result['accuracy']:6.2f}% | "
                  f"正确/总数: {result['correct']:4}/{result['total']:4}")
    
    # 计算平均准确率
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    print("=" * 60)
    print(f"平均准确率: {avg_accuracy:.2f}%")
    print("=" * 60)


def print_ppl_summary(results: List[Dict[str, Any]]):
    """
    打印PPL评测结果汇总
    
    Args:
        results: PPL评测结果列表
    """
    print("\n" + "=" * 60)
    print("PPL评测结果汇总")
    print("=" * 60)
    
    for result in results:
        ppl_mode_str = result.get('ppl_mode', 'answer-only')
        mode_label = "答" if ppl_mode_str == 'answer-only' else "全"
        if result['avg_ppl'] is not None:
            print(f"{result['dataset']:20} [{mode_label}] | "
                  f"平均PPL: {result['avg_ppl']:8.2f} | "
                  f"中位数PPL: {result['median_ppl']:8.2f} | "
                  f"有效/总数: {result['valid']:4}/{result['total']:4}")
        else:
            print(f"{result['dataset']:20} [{mode_label}] | PPL计算失败")
    
    # 计算平均PPL
    valid_results = [r for r in results if r['avg_ppl'] is not None]
    if valid_results:
        avg_ppl = sum(r['avg_ppl'] for r in valid_results) / len(valid_results)
        print("=" * 60)
        print(f"平均PPL: {avg_ppl:.2f}")
        print("=" * 60)


def main():
    """主函数"""
    args = parse_args()

    if args.sample_k < 1:
        print("✗ 错误: --sample-k 必须 >= 1")
        sys.exit(1)
    
    # 确定要评测的benchmarks
    if args.benchmarks:
        if 'all' in args.benchmarks:
            benchmarks = list(BENCHMARK_CONFIG.keys())
        else:
            benchmarks = args.benchmarks
    else:
        benchmarks = []
    
    # 分离代码benchmark和其他benchmark
    code_benchmarks = []
    normal_benchmarks = []
    
    for bm in benchmarks:
        config = BENCHMARK_CONFIG.get(bm, {})
        if config.get('type') == 'code':
            code_benchmarks.append(bm)
        else:
            normal_benchmarks.append(bm)
    
    # 确定要评测的PPL数据集
    if args.ppl_datasets:
        if 'all' in args.ppl_datasets:
            ppl_datasets = list(PPL_DATASET_CONFIG.keys())
        else:
            ppl_datasets = args.ppl_datasets
    else:
        ppl_datasets = []
    
    # 如果三者都没有指定，报错
    if not normal_benchmarks and not code_benchmarks and not ppl_datasets:
        print("✗ 错误: 请至少指定 --benchmarks 或 --ppl-datasets 中的一个")
        sys.exit(1)
    
    print(f"推理后端: {args.backend}")
    if normal_benchmarks:
        print(f"评测benchmarks (常规): {', '.join(normal_benchmarks)}")
    if code_benchmarks:
        print(f"评测benchmarks (代码): {', '.join(code_benchmarks)}")
    if ppl_datasets:
        print(f"评测PPL数据集: {', '.join(ppl_datasets)}")
    print(f"输出目录: {args.output_dir}")
    
    # 根据后端类型创建推理器
    if args.backend == 'sglang':
        # 检查必需参数
        if not args.model_path:
            print("✗ 使用sglang后端时必须指定 --model-path")
            sys.exit(1)
        
        print(f"模型路径: {args.model_path}")
        
        # 创建SGLang推理器（支持并发）
        inference = SGLangInference(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
        )
    
    elif args.backend == 'openai':
        # 检查必需参数
        if not args.openai_api_key:
            print("✗ 使用openai后端时必须指定 --openai-api-key")
            sys.exit(1)
        
        print(f"OpenAI模型: {args.openai_model}")
        if args.openai_base_url:
            print(f"API地址: {args.openai_base_url}")
        
        # 创建OpenAI推理器（支持并发）
        inference = OpenAIInference(
            api_key=args.openai_api_key,
            model=args.openai_model,
            base_url=args.openai_base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
        )
    
    else:
        print(f"✗ 不支持的后端: {args.backend}")
        sys.exit(1)
    
    # 初始化 sandbox 为 None（在 try 块外，避免 UnboundLocalError）
    sandbox = None
    
    try:
        # 检查API可用性
        if args.backend == 'sglang':
            print("检查SGLang API连接...")
            if not inference.check_server():
                print("✗ 无法连接到SGLang API")
                print(f"请确保SGLang服务器已启动: http://{args.host}:{args.port}")
                sys.exit(1)
            print("✓ SGLang API连接成功")
        elif args.backend == 'openai':
            print("检查OpenAI API连接...")
            if not inference.check_server():
                print("✗ 无法连接到OpenAI API")
                sys.exit(1)
            print("✓ OpenAI API连接成功")
        
        # 初始化代码评测器（如果需要）
        if code_benchmarks:
            # 检查是否有 Docker
            import shutil
            has_docker = shutil.which('docker') is not None
            
            if has_docker and not args.no_sandbox_server:
                # 尝试使用 Sandbox
                print("\n初始化 Sandbox 评测器...")
                try:
                    sandbox = SandboxEvaluator(
                        host=SANDBOX_CONFIG['host'],
                        port=SANDBOX_CONFIG['port'],
                        max_workers=SANDBOX_CONFIG['max_workers'],
                        docker_image=SANDBOX_CONFIG.get('docker_image', 'volcengine/sandbox-fusion:server-20250609'),
                    )
                    
                    # 启动 Sandbox 服务器
                    sandbox.start_server(wait_time=30)
                    
                except Exception as e:
                    print(f"⚠ Sandbox 启动失败: {e}")
                    print("⚠ 切换到本地代码执行器...")
                    sandbox = LocalCodeExecutor(
                        max_workers=SANDBOX_CONFIG['max_workers'],
                        timeout=SANDBOX_CONFIG['timeout'],
                    )
            elif has_docker and args.no_sandbox_server:
                # 手动模式：检查 Sandbox 是否已运行
                print("\n检查 Sandbox 服务器...")
                sandbox = SandboxEvaluator(
                    host=SANDBOX_CONFIG['host'],
                    port=SANDBOX_CONFIG['port'],
                    max_workers=SANDBOX_CONFIG['max_workers'],
                    docker_image=SANDBOX_CONFIG.get('docker_image', 'volcengine/sandbox-fusion:server-20250609'),
                )
                
                if not sandbox.check_server():
                    print("⚠ 无法连接到 Sandbox 服务器")
                    print("⚠ 切换到本地代码执行器...")
                    sandbox = LocalCodeExecutor(
                        max_workers=SANDBOX_CONFIG['max_workers'],
                        timeout=SANDBOX_CONFIG['timeout'],
                    )
                else:
                    print("✓ Sandbox 服务器连接成功")
            else:
                # 没有 Docker，使用本地执行器
                print("\n⚠ 未检测到 Docker，使用本地代码执行器...")
                print("⚠ 注意：本地执行器不提供沙箱隔离，请确保代码来源可信！")
                sandbox = LocalCodeExecutor(
                    max_workers=SANDBOX_CONFIG['max_workers'],
                    timeout=SANDBOX_CONFIG['timeout'],
                )
                print(f"✓ 本地执行器初始化完成（并发: {sandbox.max_workers}, 超时: {sandbox.timeout}s）")
        
        # 运行常规benchmark评测
        results = []
        if normal_benchmarks:
            for benchmark_name in normal_benchmarks:
                try:
                    result = run_benchmark_eval(benchmark_name, inference, args)
                    results.append(result)
                except Exception as e:
                    print(f"✗ 评测 {benchmark_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 打印和保存结果
            if results:
                print_summary(results)
                save_results(results, args)
            else:
                print("✗ 没有成功完成的常规benchmark评测")
        
        # 运行代码benchmark评测
        code_results = []
        if code_benchmarks:
            for benchmark_name in code_benchmarks:
                try:
                    result = run_code_eval(benchmark_name, inference, sandbox, args)
                    code_results.append(result)
                except Exception as e:
                    print(f"✗ 代码评测 {benchmark_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 打印和保存代码评测结果
            if code_results:
                print("\n" + "=" * 60)
                print("代码评测结果汇总")
                print("=" * 60)
                
                for result in code_results:
                    if result.get('type') == 'crux-o':
                        # CRUX-O 特殊格式：显示三个分数
                        output_acc = result['direct_output']['accuracy']
                        input_acc = result['direct_input']['accuracy']
                        avg_acc = result['average_accuracy']
                        sample_k = result['direct_output'].get('sample_k', 1)
                        if sample_k > 1:
                            print(f"{result['benchmark']:15} | "
                                  f"Output准确率: {output_acc:6.2f}% | "
                                  f"Input准确率: {input_acc:6.2f}% | "
                                  f"平均: {avg_acc:6.2f}% | "
                                  f"采样: {sample_k}")
                        else:
                            print(f"{result['benchmark']:15} | "
                                  f"Output准确率: {output_acc:6.2f}% | "
                                  f"Input准确率: {input_acc:6.2f}% | "
                                  f"平均: {avg_acc:6.2f}%")
                    else:
                        # 其他代码生成 benchmark
                        if result.get('sample_k', 1) > 1:
                            print(f"{result['benchmark']:15} | "
                                  f"通过率: {result['accuracy']:6.2f}% | "
                                  f"平均通过/总数: {result['passed']:6.1f}/{result['total']:4} | "
                                  f"采样: {result['sample_k']}")
                        else:
                            print(f"{result['benchmark']:15} | "
                                  f"通过率: {result['accuracy']:6.2f}% | "
                                  f"通过/总数: {result['passed']:4}/{result['total']:4}")
                
                # 计算平均通过率
                total_accuracy = 0.0
                for r in code_results:
                    if r.get('type') == 'crux-o':
                        total_accuracy += r['average_accuracy']
                    else:
                        total_accuracy += r['accuracy']
                avg_accuracy = total_accuracy / len(code_results)
                print("=" * 60)
                print(f"平均通过率: {avg_accuracy:.2f}%")
                print("=" * 60)
                
                # 保存结果
                save_results(code_results, args)
            else:
                print("✗ 没有成功完成的代码benchmark评测")
        
        # 运行PPL评测
        ppl_results = []
        if ppl_datasets:
            for dataset_name in ppl_datasets:
                try:
                    result = run_ppl_eval(dataset_name, inference, args)
                    ppl_results.append(result)
                except Exception as e:
                    print(f"✗ PPL评测 {dataset_name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 打印和保存PPL结果
            if ppl_results:
                print_ppl_summary(ppl_results)
                save_ppl_results(ppl_results, args)
            else:
                print("✗ 没有成功完成的PPL评测")
    
    finally:
        # 关闭连接
        if args.backend == 'sglang' and hasattr(inference, 'close'):
            inference.close()
        
        # 停止 Sandbox 服务器（如果是本脚本启动的）
        if sandbox is not None and isinstance(sandbox, SandboxEvaluator) and not args.no_sandbox_server:
            sandbox.stop_server()


if __name__ == '__main__':
    main()

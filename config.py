"""
配置文件：定义所有benchmark的数据路径和基本信息
"""

BENCHMARK_CONFIG = {
    'bbh': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/bbh_all.jsonl',
        'format': 'jsonl',
        'fields': {'input': 'input', 'target': 'target'},
        'type': 'generation',  # 需要生成答案
    },
    'mmlu-pro': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/mmlu-pro_all.jsonl',
        'format': 'jsonl',
        'fields': {'question': 'question', 'options': 'options', 'answer': 'answer'},
        'type': 'multiple_choice',
    },
    'mmlu-redux': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/mmlu-redux-all.csv',
        'format': 'csv',
        'fields': {'question': 'question', 'choices': 'choices', 'answer': 'answer'},
        'type': 'multiple_choice',
    },
    'gpqa': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/gpqa/gpqa_main.csv',
        'format': 'csv',
        'fields': {},  # CSV格式，第8列是question，9-12列是选项
        'type': 'multiple_choice',
    },
    'gsm8k': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/gsm8k_all.jsonl_temp/test-00000-of-00001.jsonl',
        'format': 'jsonl',
        'fields': {'question': 'question', 'answer': 'answer'},
        'type': 'math',
    },
    'math': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/MATH/test.jsonl',
        'format': 'jsonl',
        'fields': {'problem': 'problem', 'solution': 'solution', 'answer': 'answer'},
        'type': 'math',
    },
    'mmlu': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/mmlu_all.jsonl_temp/test-00000-of-00001.jsonl',
        'format': 'jsonl',
        'fields': {'question': 'question', 'choices': 'choices', 'answer': 'answer'},
        'type': 'multiple_choice',
    },
    'supergpqa': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/SuperGPQA/SuperGPQA-all.jsonl',
        'format': 'jsonl',
        'fields': {'question': 'question', 'options': 'options', 'answer': 'answer_letter'},
        'type': 'multiple_choice',
    },
    'hellaswag': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/hellaswag.parquet',
        'format': 'parquet',
        'fields': {'ctx': 'ctx', 'endings': 'endings', 'label': 'label'},
        'type': 'multiple_choice',
    },
    'arc-c': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/arc-c.parquet',
        'format': 'parquet',
        'fields': {'question': 'question', 'choices': 'choices', 'answer': 'answerKey'},
        'type': 'multiple_choice',
    },
    'arc-e': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/arc-e.parquet',
        'format': 'parquet',
        'fields': {'question': 'question', 'choices': 'choices', 'answer': 'answerKey'},
        'type': 'multiple_choice',
    },
    'winogrande': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/winogrand.parquet',
        'format': 'parquet',
        'fields': {'sentence': 'sentence', 'option1': 'option1', 'option2': 'option2', 'answer': 'answer'},
        'type': 'multiple_choice',
    },
    'piqa': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/piqa.parquet',
        'format': 'parquet',
        'fields': {'goal': 'goal', 'sol1': 'sol1', 'sol2': 'sol2', 'label': 'label'},
        'type': 'multiple_choice',
    },
    'nq': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/nq.json',
        'format': 'json',
        'fields': {'question': 'question', 'context': 'context', 'answers': 'answers'},
        'type': 'reading',
    },
    'simpleqa': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/simple_qa_test_set.csv',
        'format': 'csv',
        'fields': {'question': 'question', 'answer': 'answer'},
        'type': 'reading',
    },
    'drop': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/drop.parquet',
        'format': 'parquet',
        'fields': {'passage': 'passage', 'question': 'question', 'answers': 'answers'},
        'type': 'reading',
    },
    # 代码生成 benchmarks
    'mbpp': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/mbpp_all.jsonl_temp/test-00000-of-00001.jsonl',
        'format': 'jsonl',
        'fields': {'text': 'text', 'code': 'code', 'test_list': 'test_list'},
        'type': 'code',
        'language': 'python',
        'sandbox_dataset': 'mbpp',
    },
    'mbppplus': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/mbppplus_all.jsonl_temp/test-00000-of-00001-d5781c9c51e02795.jsonl',
        'format': 'jsonl',
        'fields': {'prompt': 'prompt', 'code': 'code', 'test': 'test', 'test_list': 'test_list'},
        'type': 'code',
        'language': 'python',
        'sandbox_dataset': 'mbpp',  # 使用 mbpp 评测方式
    },
    'humaneval': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/Humanevalplus_all.jsonl_temp/test-00000-of-00001-5973903632b82d40.jsonl',
        'format': 'jsonl',
        'fields': {'prompt': 'prompt', 'canonical_solution': 'canonical_solution', 'test': 'test'},
        'type': 'code',
        'language': 'python',
        'sandbox_dataset': 'humaneval',
    },
    'crux-o': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/crux-o/test.jsonl',
        'format': 'jsonl',
        'fields': {'code': 'code', 'input': 'input', 'output': 'output'},
        'type': 'code',
        'language': 'python',
        'sandbox_dataset': 'crux-o',  # 特殊处理
    },
}

# PPL数据集配置
PPL_DATASET_CONFIG = {
    'bbh_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/bbh_all_ppl.jsonl',
        'format': 'jsonl',
        'source': 'bbh',
    },
    'crux-o_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/crux-o_ppl.jsonl',
        'format': 'jsonl',
        'source': 'crux-o',
    },
    'gpqa_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/gpqa_main_ppl.jsonl',
        'format': 'jsonl',
        'source': 'gpqa',
    },
    'gsm8k_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/gsm8k_ppl.jsonl',
        'format': 'jsonl',
        'source': 'gsm8k',
    },
    'humanevalplus_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/humanevalplus_ppl.jsonl',
        'format': 'jsonl',
        'source': 'humanevalplus',
    },
    'math_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/MATH_ppl.jsonl',
        'format': 'jsonl',
        'source': 'math',
    },
    'mbpp_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/mbpp_ppl.jsonl',
        'format': 'jsonl',
        'source': 'mbpp',
    },
    'mbppplus_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/mbppplus_all_ppl.jsonl',
        'format': 'jsonl',
        'source': 'mbppplus',
    },
    'mmlu-pro_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/mmlu-pro_all_ppl.jsonl',
        'format': 'jsonl',
        'source': 'mmlu-pro',
    },
    'mmlu-redux_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/mmlu-redux-all_ppl.jsonl',
        'format': 'jsonl',
        'source': 'mmlu-redux',
    },
    'supergpqa_ppl': {
        'path': '/mnt/cephfs/users/yuchenfan/benchmark-eval/dataset/ppl_dataset/SuperGPQA-all_ppl.jsonl',
        'format': 'jsonl',
        'source': 'supergpqa',
    },
}

# SGLANG配置
SGLANG_CONFIG = {
    'host': '10.120.5.128',
    'port': 8000,
    'temperature': 0.0,  # 对于评测，通常使用贪心解码
    'max_tokens': 2048
}

# OpenAI配置
OPENAI_CONFIG = {
    'api_key': 'noneed',  # 需要设置API密钥
    'base_url': "http://10.120.4.10:8000/v1",  # 可选，用于自定义API端点（如OpenAI兼容的API）
    'model': '/mnt/cephfs/users/yuchenfan/model/Qwen3-0.6B',  # 默认模型
    'temperature': 0.0,  # 对于评测，通常使用贪心解码
    'max_tokens': 2048,
}

# Sandbox Fusion 配置
SANDBOX_CONFIG = {
    'host': '127.0.0.1',
    'port': 8080,
    'max_workers': 16,  # 代码评测的并发数
    'timeout': 30,  # 单个样本的超时时间（秒）
    # 使用国内镜像加速
    'docker_image': 'vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609',
    # 国际版镜像：
    # 'docker_image': 'volcengine/sandbox-fusion:server-20250609',
}

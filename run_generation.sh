#!/bin/bash

################################################################################
# 自动化多模型评测脚本
# 用法: ./run_all_evals.sh [all|reasoning|code] [样本数]
################################################################################

# 1. 配置区域
# 在这里添加你想评测的模型路径列表
MODELS=(
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0188415-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0196607-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0180223-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0172031-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0163839-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0155647-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0147455-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0139263-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0122879-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0114687-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0106495-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0094207-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_00786015-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0077823-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0069631-hf"
    "/mnt/cephfs/users/yuchenfan/qwen-kda-fixed-data-aug-checked/iter_0057343-hf"
)

TASK_TYPE="${1:-all}"
MAX_SAMPLES="${2:-}"
HOST="0.0.0.0"
PORT="30000"
MAX_WORKERS=2048
OUTPUT_DIR="./results"
LOG_DIR="./logs"
mkdir -p $OUTPUT_DIR $LOG_DIR

# tmux 会话名称
TMUX_SESSION="eval_server"

# 任务定义
# REASONING_TASKS="math gsm8k gpqa mmlu-pro mmlu-redux mmlu"
REASONING_TASKS="mmlu-redux"
# CODE_TASKS="mbpp mbppplus humaneval"

case $TASK_TYPE in
    all) BENCHMARKS="$REASONING_TASKS $CODE_TASKS" ;;
    reasoning) BENCHMARKS="$REASONING_TASKS" ;;
    code) BENCHMARKS="$CODE_TASKS" ;;
    *) echo "错误: 未知任务类型"; exit 1 ;;
esac

# 2. 函数：等待 Server 就绪
wait_for_server() {
    local url="http://127.0.0.1:$PORT/v1/models"
    echo "等待 sglang server 就绪 ($url)..."
    while ! curl -s $url > /dev/null; do
        sleep 5
        echo -n "."
    done
    echo -e "\nServer 已启动!"
}

# 3. 循环遍历模型
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CURRENT_LOG="$LOG_DIR/${MODEL_NAME}_${TIMESTAMP}.log"
    
    echo "================================================================"
    echo "正在处理模型: $MODEL_NAME"
    echo "模型路径: $MODEL_PATH"
    echo "日志文件: $CURRENT_LOG"
    echo "================================================================"

    # a. 清理旧的 tmux 会话（防止残留）
    tmux kill-session -t $TMUX_SESSION 2>/dev/null
    sleep 2

    # b. 使用 tmux 启动 sglang server
    echo "正在启动 sglang server..."
    tmux new-session -d -s $TMUX_SESSION "python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host $HOST \
        --port $PORT \
        --log-level warning \
        --dp 8"

    # c. 等待服务可用
    wait_for_server

    # d. 构建评测命令
    CMD="python run_eval.py \
        --backend openai \
        --openai-model $MODEL_PATH \
        --benchmarks $BENCHMARKS \
        --max-workers $MAX_WORKERS \
        --output-dir $OUTPUT_DIR \
        --openai-base-url http://127.0.0.1:$PORT/v1"

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max-samples $MAX_SAMPLES"
    fi

    # e. 执行评测并记录日志
    echo "开始执行评测..."
    echo "命令: $CMD"
    eval "$CMD" 2>&1 | tee "$CURRENT_LOG"

    # f. 评测结束，关闭 server
    echo "评测完成，关闭 $MODEL_NAME 的 Server..."
    tmux kill-session -t $TMUX_SESSION
    
    # 给系统一点时间释放显存
    sleep 10
done

echo "所有模型评测任务已完成！"

#!/bin/bash

# Example usage:
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./eval_openllm.sh "meta-llama/Meta-Llama-3-8B-Instruct" "tensor_parallel_size=4,max_model_len=4096,add_bos_token=True,gpu_memory_utilization=0.8"

export MODEL_DIR=${1}
export MODEL_ARGS=${2}

declare -A tasks_fewshot=(
    ["arc_challenge"]=25
    ["winogrande"]=5
    ["truthfulqa_mc2"]=0
    ["hellaswag"]=10
    ["mmlu"]=5
    ["gsm8k"]=5
)

for TASK in "${!tasks_fewshot[@]}"; do
    NUM_FEWSHOT=${tasks_fewshot[$TASK]}
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_DIR,$MODEL_ARGS \
        --tasks ${TASK} \
        --num_fewshot ${NUM_FEWSHOT} \
        --write_out \
        --show_config \
        --device cuda \
        --batch_size 16 \
        --output_path="results/"${TASK}
done

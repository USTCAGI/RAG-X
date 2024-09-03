#!/bin/bash

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false

# 指定使用的 GPU（例如，使用第一个 GPU，即 cuda:0）
export CUDA_VISIBLE_DEVICES=7

# 模型路径
MODEL_PATH="Meta-Llama-3.1-8B-Instruct"
# MODEL_PATH="qwen2"

# 运行 vllm serve 命令
vllm serve $MODEL_PATH --dtype auto --port 8000 --max-model-len 42736 #32768 #42736

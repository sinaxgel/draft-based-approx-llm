#!/usr/bin/env bash
set -uo pipefail

LOG=/home/ubuntu/zhangpengcheng/logs/milebench-specpc-pilot.log
STATUS=/home/ubuntu/zhangpengcheng/logs/milebench-specpc-pilot.status
OUTPUT=/home/ubuntu/zhangpengcheng/outputs/milebench-specpc-pilot/results.json

mkdir -p \
    /home/ubuntu/zhangpengcheng/logs \
    /home/ubuntu/zhangpengcheng/outputs/milebench-specpc-pilot \
    /home/ubuntu/zhangpengcheng/cache/huggingface \
    /home/ubuntu/zhangpengcheng/cache/torch_extensions

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/ubuntu/zhangpengcheng/draft-based-approx-llm${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/home/ubuntu/zhangpengcheng/cache/huggingface
export TORCH_EXTENSIONS_DIR=/home/ubuntu/zhangpengcheng/cache/torch_extensions
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

{
    echo "RUNNING $(date --iso-8601=seconds)"
    echo "host=$(hostname) physical_gpus=${CUDA_VISIBLE_DEVICES} conda_env=zpc-spec"
    echo "output=${OUTPUT}"
} > "${STATUS}"

cd /home/ubuntu/zhangpengcheng/draft-based-approx-llm || exit 90
/home/ubuntu/miniconda3/bin/conda run --no-capture-output -n zpc-spec \
    python scripts/vlm_milebench_specpc_pilot.py \
        --resume \
        --output "${OUTPUT}" 2>&1 | tee -a "${LOG}"
code=${PIPESTATUS[0]}
if [[ ${code} -eq 0 ]]; then state=PASSED; else state=FAILED; fi
echo "${state} $(date --iso-8601=seconds) exit_code=${code}" | tee "${STATUS}"
exit "${code}"

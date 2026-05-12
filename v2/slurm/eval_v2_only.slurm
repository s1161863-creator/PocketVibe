#!/bin/bash
#SBATCH --job-name=pv_v2only
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j_v2only.out
#SBATCH --error=logs/%j_v2only.err

# =================================================================
# PocketVibe V2+ — V2 单模型细粒度推理（5 条高难度题目 + 100 分制）
# 提交: sbatch slurm/eval_v2_only.slurm
# 监控: squeue -u $USER   /   tail -f logs/*_v2only.out
# 与 slurm/eval_compare.slurm 可以同时提交 (两张卡并行)
# =================================================================

source ~/venvs/pv-train/bin/activate
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

cd ~/PocketVibe
mkdir -p logs data/eval

echo "===== V2+ 单模型评测开始 $(date) ====="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ ! -d outputs/qlora-v2-run1/final_adapter ]; then
    echo "❌ V2 adapter 不存在：outputs/qlora-v2-run1/final_adapter"
    exit 1
fi
echo "✅ V2 adapter: outputs/qlora-v2-run1/final_adapter"

python scripts/04_inference_test_v2plus.py
EXIT=$?

echo "===== V2+ 单模型评测结束 $(date) ====="
echo "退出码: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo "--- 输出文件 ---"
    ls -lah data/eval/v2plus_* 2>/dev/null
fi

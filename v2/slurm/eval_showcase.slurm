#!/bin/bash
#SBATCH --job-name=pv_showcase
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j_showcase.out
#SBATCH --error=logs/%j_showcase.err

# ===================================================================
# PocketVibe V2+ — Showcase 对比 (V1 vs V2, 2 道 V2 强项题)
# ===================================================================
# 占 1 张 A16，总耗时 ~40-50 min（V1 2题 + V2 2题，Best-of-3 每题 3 次采样）
# 与 eval_v2_only.slurm / eval_compare.slurm 可**并行**（互不冲突）
# ===================================================================

source /opt/apps/python-envs/week07/bin/activate
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=1

cd ~/PocketVibe
mkdir -p logs data/eval

# 显存监控（每 60 秒 1 条，用于报告附录）
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 60 \
    > logs/${SLURM_JOB_ID}_showcase_gpu.csv &
MON_PID=$!

echo "===== Showcase 开始 $(date) ====="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
nvidia-smi

python scripts/06_showcase_v2_strengths.py

RC=$?
kill $MON_PID 2>/dev/null
echo "===== Showcase 结束 $(date)  退出码=$RC ====="
exit $RC

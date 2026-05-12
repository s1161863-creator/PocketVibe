#!/bin/bash
#SBATCH --job-name=pv_v2_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err

# =================================================================
# PocketVibe V2 — V1 vs V2 两版 LoRA 对比评测作业
# 提交方式：sbatch slurm/eval.slurm
# 监控方式：squeue -u $USER
#           tail -f logs/*_eval.out
# =================================================================

# ── 激活训练时用的 venv ──
source ~/venvs/pv-train/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python -V)"

# ── 模型缓存 ──
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

cd ~/PocketVibe
mkdir -p logs data/eval

echo "===== 评测开始 $(date) ====="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ── 前置检查：两个 adapter 必须都在 ──
if [ ! -d outputs/qlora-run1/final_adapter ]; then
    echo "❌ 错误：V1 adapter 不存在（outputs/qlora-run1/final_adapter）"
    exit 1
fi
if [ ! -d outputs/qlora-v2-run1/final_adapter ]; then
    echo "❌ 错误：V2 adapter 不存在（outputs/qlora-v2-run1/final_adapter）"
    exit 1
fi
echo "✅ V1 adapter: outputs/qlora-run1/final_adapter"
echo "✅ V2 adapter: outputs/qlora-v2-run1/final_adapter"
echo ""

# ── 运行对比评测 ──
python scripts/05_eval_compare_v1_vs_v2.py

EVAL_EXIT=$?
echo ""
echo "===== 评测结束 $(date) ====="
echo "退出码：$EVAL_EXIT"

if [ $EVAL_EXIT -eq 0 ]; then
    echo ""
    echo "--- 输出文件 ---"
    ls -lah data/eval/compare_v1v2_* 2>/dev/null
fi

#!/bin/bash
#SBATCH --job-name=pv_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%j_compare.out
#SBATCH --error=logs/%j_compare.err

# =================================================================
# PocketVibe V2+ — V1 vs V2 细粒度对比评测（5 条高难度题 + 100 分制）
#
# V1 = ~/PocketVibe/outputs/qlora-run1/final_adapter
#      (原始交接版 / 与本地 C:\Users\Lenovo\Desktop\Enoch 仓库一致)
# V2 = ~/PocketVibe/outputs/qlora-v2-run1/final_adapter
#      (本工作 Version2 训练 / SLURM job 1442 成果)
#
# 提交: sbatch slurm/eval_compare.slurm
# 监控: squeue -u $USER   /   tail -f logs/*_compare.out
# 与 slurm/eval_v2_only.slurm 可以同时提交（两张 A16 卡并行）
# =================================================================

source ~/venvs/pv-train/bin/activate
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

cd ~/PocketVibe
mkdir -p logs data/eval

echo "===== V1 vs V2 对比评测开始 $(date) ====="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ ! -d outputs/qlora-run1/final_adapter ]; then
    echo "❌ V1 adapter 不存在：outputs/qlora-run1/final_adapter"
    exit 1
fi
if [ ! -d outputs/qlora-v2-run1/final_adapter ]; then
    echo "❌ V2 adapter 不存在：outputs/qlora-v2-run1/final_adapter"
    exit 1
fi
echo "✅ V1 adapter: outputs/qlora-run1/final_adapter"
echo "✅ V2 adapter: outputs/qlora-v2-run1/final_adapter"

# ── 打印两个 adapter 的配置做身份自证 ──
echo ""
echo "--- V1 adapter_config.json ---"
cat outputs/qlora-run1/final_adapter/adapter_config.json 2>/dev/null | head -30
echo ""
echo "--- V2 adapter_config.json ---"
cat outputs/qlora-v2-run1/final_adapter/adapter_config.json 2>/dev/null | head -30
echo ""

python scripts/05_eval_compare_v1_vs_v2_v2plus.py
EXIT=$?

echo "===== V1 vs V2 对比评测结束 $(date) ====="
echo "退出码: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo "--- 输出文件 ---"
    ls -lah data/eval/compare_v1v2p_* 2>/dev/null
fi

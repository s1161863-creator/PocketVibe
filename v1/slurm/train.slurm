#!/bin/bash
#SBATCH --job-name=pv_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

source ~/venvs/pv-train/bin/activate
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PV_USE_4BIT=${PV_USE_4BIT:-0}
export PV_MAX_SEQ_LENGTH=${PV_MAX_SEQ_LENGTH:-2048}
export PV_GRAD_ACC_STEPS=${PV_GRAD_ACC_STEPS:-8}
export PV_USE_GRAD_CHECKPOINTING=${PV_USE_GRAD_CHECKPOINTING:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd ~/PocketVibe
mkdir -p logs

nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 60 > logs/gpu_mem.csv &

echo "===== 开始训练 $(date) ====="
echo "Python: $(which python)"
python -V
python scripts/03_train_qlora.py
echo "===== 训练结束 $(date) ====="

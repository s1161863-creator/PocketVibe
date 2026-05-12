#!/bin/bash
#SBATCH --job-name=pv_serve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_serve.out
#SBATCH --error=logs/%j_serve.err

source ~/venvs/pv-train/bin/activate

export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=1

cd ~/PocketVibe
mkdir -p logs

echo "===== 启动推理 API $(date) ====="
python scripts/06_serve_api.py
echo "===== 服务结束 $(date) ====="

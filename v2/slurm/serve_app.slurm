#!/bin/bash
#SBATCH --job-name=pv_app
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_app.out
#SBATCH --error=logs/%j_app.err

# =================================================================
# PocketVibe Gradio Web Demo 后端
# 提交: sbatch slurm/serve_app.slurm
# 访问: 在本地执行 SSH 隧道
#   ssh -N -L 7860:<compute-node>:7860 student07@aaillm.eduhk.hk
# 然后浏览器打开 http://localhost:7860
# =================================================================

source ~/venvs/pv-train/bin/activate

cd ~/PocketVibe
mkdir -p logs

# 模型路径 (与 app.py 内默认一致, 可按需覆盖)
export PV_BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
export PV_V1_ADAPTER="$HOME/PocketVibe/outputs/qlora-run1/final_adapter"
export PV_V2_ADAPTER="$HOME/PocketVibe/outputs/qlora-v2-run1/final_adapter"

# 打印计算节点名, 方便做 SSH 端口转发
echo "===== Gradio server starting at $(date) ====="
echo "Compute node: $(hostname)"
echo "Tunnel cmd:  ssh -N -L 7860:$(hostname):7860 $USER@aaillm.eduhk.hk"
echo "=============================================="

python app/app.py --device cuda --host 0.0.0.0 --port 7860

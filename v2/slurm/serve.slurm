#!/bin/bash
#SBATCH --job-name=pv_v2_serve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_serve.out
#SBATCH --error=logs/%j_serve.err

# =================================================================
# PocketVibe V2 — 推理服务作业脚本
# 提交方式：sbatch slurm/serve.slurm
# 查看端口：cat logs/*_serve.out | grep "port\|PORT\|Uvicorn"
# 前端访问：http://localhost:8000/docs （SSH 端口转发后）
#
# SSH 端口转发（本地运行，把 HPC:8000 映射到本机:8000）：
#   ssh -L 8000:localhost:8000 student40@aaillm.eduhk.hk
# =================================================================

# ── 激活环境 ──
source /opt/apps/python-envs/week07/bin/activate

# ── 模型缓存（离线模式）──
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=1

# ── 服务端口（可修改）──
export PV_PORT=8000

# ── 进入项目目录 ──
cd ~/PocketVibe_v2
mkdir -p logs

echo "===== 推理服务启动 $(date) ====="
echo "SLURM Job ID : $SLURM_JOB_ID"
echo "节点         : $SLURMD_NODENAME"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "服务端口     : $PV_PORT"
echo ""

# ── 确认适配器存在 ──
ADAPTER=~/PocketVibe_v2/outputs/qlora-v2-run1/final_adapter
if [ ! -d "$ADAPTER" ]; then
    echo "❌ 错误：找不到 LoRA 适配器：$ADAPTER"
    echo "   请先运行训练作业：sbatch slurm/train.slurm"
    exit 1
fi
echo "✅ 适配器目录：$ADAPTER"
echo ""

# ── 显存监控（后台）──
nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv \
    -l 60 > logs/gpu_serve_${SLURM_JOB_ID}.csv &

# ── 健康检查函数（等待服务就绪）──
wait_for_service() {
    local max_wait=120
    local waited=0
    echo "等待服务就绪（最多 ${max_wait}s）..."
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://localhost:${PV_PORT}/health" > /dev/null 2>&1; then
            echo "✅ 服务已就绪！"
            curl -s "http://localhost:${PV_PORT}/health" | python3 -m json.tool
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  ... 已等待 ${waited}s"
    done
    echo "⚠️  服务启动超时，但进程可能仍在加载模型"
    return 1
}

# ── 启动 FastAPI 服务（后台启动，等待就绪后打印信息）──
echo "--- 启动 FastAPI 推理服务 ---"
python scripts/06_serve_api.py &
SERVER_PID=$!

# 等待服务就绪
sleep 10
wait_for_service

echo ""
echo "======================================================"
echo "🚀 PocketVibe V2 推理服务运行中"
echo "   节点内访问：http://localhost:${PV_PORT}"
echo "   API 文档  ：http://localhost:${PV_PORT}/docs"
echo ""
echo "   从本地访问（SSH 端口转发）："
echo "   ssh -L ${PV_PORT}:localhost:${PV_PORT} \\"
echo "       student40@aaillm.eduhk.hk"
echo "   然后浏览器打开：http://localhost:${PV_PORT}/docs"
echo ""
echo "   测试命令："
echo "   curl -X POST http://localhost:${PV_PORT}/generate \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"instruction\":\"做一个倒计时器\"}'"
echo "======================================================"
echo ""

# ── 等待服务进程（保持 SLURM 作业存活）──
wait $SERVER_PID
SERVE_EXIT=$?

echo ""
echo "===== 推理服务结束 $(date) ====="
echo "退出码：$SERVE_EXIT"

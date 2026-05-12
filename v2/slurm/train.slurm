#!/bin/bash
#SBATCH --job-name=pv_v2_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

# =================================================================
# PocketVibe V2 — 训练作业脚本
# 提交方式：sbatch slurm/train.slurm
# 监控方式：squeue -u $USER
#           tail -f logs/*_train.out
# =================================================================

# ── 激活环境（使用 V1 已调通的个人 venv，不是只读共享 venv）──
source ~/venvs/pv-train/bin/activate

# ── 诊断信息 ──
echo "Python interpreter: $(which python)"
echo "Python version: $(python -V)"

# ── 模型缓存（online-first：缺了能自动拉，V1 教训）──
export HF_HOME=/opt/shared/model-cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# ── V2 训练参数（可通过环境变量覆盖）──
# 默认不使用 4-bit 量化（bitsandbytes HPC 后端缺失，V1 已验证）
export PV_USE_4BIT=${PV_USE_4BIT:-0}
# ★ V2 重做关键修复：max_seq_length 从 1024 提升到 4096
#    V2 首轮 bug 复盘：train.jsonl 100% 样本 > 1024 tokens，最长估算 > 5000 tokens
#    3072 会截掉 ~20% 样本；4096 覆盖 ~96%，且配合 02c 预过滤可保证 0 截断
#    4096 也是 2 的整数幂，Flash/SDPA Attention 对齐最佳
#    显存预算：A16 16GB, bf16 LoRA + grad_ckpt, 峰值约 10 GB，余量充裕
export PV_MAX_SEQ_LENGTH=${PV_MAX_SEQ_LENGTH:-4096}
# ★ V2 重做新增：梯度累积 32 步（有效 batch=32），gradient checkpointing 必开
#    序列长度从 1024→4096（4x）导致单步显存占用增加，
#    累积步数加倍可在保持收敛稳定性的同时降低峰值显存
export PV_GRAD_ACC_STEPS=${PV_GRAD_ACC_STEPS:-32}
export PV_USE_GRAD_CHECKPOINTING=${PV_USE_GRAD_CHECKPOINTING:-1}
# Run tag：V2 重做直接覆盖 V2 首轮目录（qlora-v2-run1）
# 如需保留首轮产物，请先在 HPC 侧手动备份：
#   mv ~/PocketVibe/outputs/qlora-v2-run1 ~/PocketVibe/outputs/qlora-v2-run1_old_1024trunc
export PV_RUN_TAG=${PV_RUN_TAG:-qlora-v2-run1}

# ── CUDA 分配器优化（V1 验证有效）──
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── 进入项目目录（复用 V1 目录，venv 和缓存都现成）──
cd ~/PocketVibe
mkdir -p logs

echo "===== 训练开始 $(date) ====="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "节点: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ── 显存监控（后台，每 60 秒记录一次）──
nvidia-smi \
    --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv \
    -l 60 > logs/gpu_mem_${SLURM_JOB_ID}.csv &
GPU_MON_PID=$!

# ── 运行数据流水线（若尚未在本地运行）──
# 如果已在本地完成数据准备并通过 scp 上传，注释掉下面这段
# echo "--- 01a: 多实现扩充 ---"
# python scripts/01a_multi_implementation.py
# echo "--- 01b: 进化指令扩充 ---"
# python scripts/01b_evol_instruct_tools.py
# echo "--- 01c: 跨类组合生成 ---"
# python scripts/01c_cross_category.py
# echo "--- 01d: 合并去重 ---"
# python scripts/01d_merge_and_dedupe.py
# echo "--- 01e: 静态校验 ---"
# python scripts/01e_static_validate.py
# echo "--- 02: 类别切分 ---"
# python scripts/02_category_split.py

# ── 确认训练数据存在 ──
if [ ! -f data/processed/train.jsonl ]; then
    echo "❌ 错误：data/processed/train.jsonl 不存在！"
    echo "   请先在本地运行数据流水线脚本（01a→01e→02），再 scp 上传"
    exit 1
fi

TRAIN_N=$(wc -l < data/processed/train.jsonl)
VAL_N=$(wc -l < data/processed/val.jsonl)
echo "训练集：${TRAIN_N} 条"
echo "验证集：${VAL_N} 条"
echo ""

# ── 启动训练 ──
echo "--- QLoRA 微调开始 ---"
python scripts/03_train_qlora_v2.py

TRAIN_EXIT=$?
echo ""
echo "===== 训练结束 $(date) ====="
echo "退出码：$TRAIN_EXIT"

# ── 停止显存监控 ──
kill $GPU_MON_PID 2>/dev/null

# ── 训练成功后自动打印关键指标 ──
if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "--- 最终 loss 摘要 ---"
    PV_RUN_TAG_VAL=${PV_RUN_TAG} python - <<'EOF'
import json, os
tag = os.environ.get("PV_RUN_TAG_VAL", "qlora-v2-run1")
log = os.path.expanduser(f"~/PocketVibe/logs/train_log_{tag}.json")
if os.path.exists(log):
    with open(log) as f:
        h = json.load(f)
    tl = [x["loss"]      for x in h if "loss"      in x]
    el = [x["eval_loss"] for x in h if "eval_loss" in x]
    if tl: print(f"  最终 train_loss : {tl[-1]:.4f}")
    if el: print(f"  最佳 eval_loss  : {min(el):.4f}  (最后一次: {el[-1]:.4f})")
    print(f"  总训练步数      : {max((x.get('step',0) for x in h), default=0)}")
else:
    print(f"  (日志文件未找到: {log})")
EOF
    echo ""
    ADAPTER=$(ls -d ~/PocketVibe/outputs/${PV_RUN_TAG}/final_adapter 2>/dev/null)
    if [ -n "$ADAPTER" ]; then
        echo "✅ 适配器已保存：$ADAPTER"
        echo "   文件大小："
        du -sh $ADAPTER
    fi
else
    echo "❌ 训练失败，退出码：$TRAIN_EXIT"
    echo "   查看完整错误：cat logs/${SLURM_JOB_ID}_train.err"
fi

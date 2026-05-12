# PocketVibe V2 — HPC 分步执行清单 (student07 专用)

> **一步一步照做，每一步都贴了完整命令，复制粘贴即可。**

## 📌 重要信息速查

| 项目 | 值 |
|---|---|
| HPC 登录地址 | `aaillm.eduhk.hk` |
| 账号 | `student07` |
| 密码 | **你自己记得的那个（老 README 里没有明文）** |
| 服务器项目目录 | `~/PocketVibe`（复用 V1 目录，venv 和缓存都现成） |
| Python venv | `~/venvs/pv-train/`（V1 已验证可用）|
| 本地项目目录 | `C:\Users\Lenovo\Desktop\Enoch - Version2` |

---

## 🎬 Step 1：在本地打包

> ⚠️ **重要**：反引号 `` ` `` 只在 PowerShell 里是续行符，**cmd 不认**。
> 下面给的是**单行命令**，cmd 和 PowerShell 都能跑，**整整一行复制粘贴**。

先进入桌面目录：

```
cd C:\Users\Lenovo\Desktop
```

然后粘贴这一整行（不要换行！）：

```
tar --exclude="Enoch - Version2/outputs" --exclude="Enoch - Version2/logs" --exclude="Enoch - Version2/data/processed/new_tools.shard0.jsonl" --exclude="Enoch - Version2/data/processed/new_tools.shard1.jsonl" --exclude="Enoch - Version2/data/processed/new_tools.shard2.jsonl" --exclude="Enoch - Version2/data/processed/new_tools.shard3.jsonl" --exclude="Enoch - Version2/data/processed/new_tools.jsonl" --exclude="Enoch - Version2/data/processed/augmented_instructions.jsonl" --exclude="Enoch - Version2/data/processed/train_generated.jsonl" --exclude="Enoch - Version2/data/processed/train_clean.jsonl" --exclude="__pycache__" --exclude="*.pyc" -czf pocketvibe_v2.tar.gz "Enoch - Version2"
```

验证打包成功（应该看到一个 5-15 MB 的 .tar.gz 文件）：

```
dir pocketvibe_v2.tar.gz
```

---

## 🚀 Step 2：上传到 HPC

```powershell
scp pocketvibe_v2.tar.gz student07@aaillm.eduhk.hk:~/
```

**会提示输入密码**，输入你的 HPC 密码（密码在输入时不会显示，正常的）。

成功的话会看到进度条跑到 100%。

---

## 🔐 Step 3：SSH 登录 HPC

```powershell
ssh student07@aaillm.eduhk.hk
```

再输一次密码。登录成功后你会看到提示符变成类似：
```
[student07@aaillm-login ~]$
```

> ⚠️ **以下所有 Step 4-9 都在这个 HPC 终端里运行，不要回本地 PowerShell。**

---

## 📂 Step 4：备份旧项目 + 解压新项目

**先备份 V1 的 outputs（防止覆盖）**：

```bash
cd ~
[ -d PocketVibe/outputs ] && cp -r PocketVibe/outputs PocketVibe/outputs.v1.bak.$(date +%Y%m%d)
```

**解压新包**：

```bash
cd ~
tar -xzf pocketvibe_v2.tar.gz
```

**把 V2 的文件覆盖到 V1 目录**（复用 V1 venv 就不用重装依赖）：

```bash
# 复制 V2 的 scripts、slurm、data、requirements 到 PocketVibe
cp -r "Enoch - Version2/scripts/"* ~/PocketVibe/scripts/
cp -r "Enoch - Version2/slurm/"* ~/PocketVibe/slurm/
cp -r "Enoch - Version2/data/"* ~/PocketVibe/data/
[ -f "Enoch - Version2/requirements-train.txt" ] && cp "Enoch - Version2/requirements-train.txt" ~/PocketVibe/

# 删掉解压出来的空目录
rm -rf "Enoch - Version2"
```

---

## ✅ Step 5：验证环境和数据都 OK

```bash
cd ~/PocketVibe
```

**验证数据文件行数（应显示 480 / 70 / 85）**：

```bash
wc -l data/processed/train.jsonl data/processed/val.jsonl data/processed/test.jsonl
```

**验证 venv 存在**：

```bash
ls ~/venvs/pv-train/bin/python && echo "✅ venv OK"
```

**验证 V2 训练脚本已就位**：

```bash
ls -la scripts/03_train_qlora_v2.py slurm/train.slurm && echo "✅ 脚本 OK"
```

**验证 Qwen 模型已缓存**：

```bash
ls /opt/shared/model-cache/hub/ | grep -i qwen2.5-coder && echo "✅ 模型已缓存"
```

> 如果以上任何一步失败，**先别继续**，把错误截图发给我。

---

## 🔬 Step 6：验证 V2 新增依赖（如有需要补装）

V2 用到了 `DataCollatorForCompletionOnlyLM` 和 `EarlyStoppingCallback`，V1 的 venv 里 transformers / trl 版本应该够用，但保险起见检查一下：

```bash
source ~/venvs/pv-train/bin/activate
python -c "from trl import SFTTrainer, DataCollatorForCompletionOnlyLM; from transformers import EarlyStoppingCallback; print('✅ V2 依赖全部就位')"
```

如果报错 `ImportError`，就补装一下（只在报错时运行）：

```bash
pip install --upgrade "trl==0.10.1" "transformers>=4.40.0" "peft>=0.10.0"
```

---

## 🎯 Step 7：提交训练作业

```bash
cd ~/PocketVibe
chmod +x slurm/*.slurm
mkdir -p logs

sbatch slurm/train.slurm
```

成功提交会看到类似输出：
```
Submitted batch job 1456
```

**把这个作业号（比如 1456）记下来**，监控和出错排查都要用到。

---

## 📊 Step 8：监控训练

### 查看作业状态（每隔几分钟跑一次）
```bash
squeue -u $USER
```

状态含义：
- `PD` = Pending（排队等 GPU）
- `R` = Running（正在跑）
- 列表空了 = 作业结束（成功或失败都会消失）

### 实时看训练输出（Ctrl+C 退出）
```bash
tail -f logs/*_train.out
```

### 看错误日志（如果作业突然消失）
```bash
tail -50 logs/*_train.err
```

### 看显存占用
```bash
tail -20 logs/gpu_mem_*.csv
```

**预计耗时：1-2 小时**（A16 单卡 × 480 条 × 2 epoch）

---

## ✅ Step 9：训练成功的标志

日志末尾出现这些，就是成功了：

```
===== 训练结束 ... =====
退出码：0
--- 最终 loss 摘要 ---
  最终 train_loss : 0.xxxx
  最佳 eval_loss  : 0.xxxx
✅ 适配器已保存：~/PocketVibe/outputs/qlora-v2-run1/final_adapter
```

---

## 🆘 常见故障速查

| 症状 | 处理 |
|---|---|
| `CUDA out of memory` | 重新提交时降低序列长度：<br>`PV_MAX_SEQ_LENGTH=768 sbatch slurm/train.slurm` |
| `bitsandbytes` 相关报错 | 默认已关（`PV_USE_4BIT=0`），无需处理 |
| `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM'` | 回到 Step 6 执行 pip 升级命令 |
| 作业排队 > 15 分钟 | GPU 被占用中，耐心等；`squeue -p gpu` 看队列 |
| 作业很快结束（< 5 分钟）且状态变 `CD` 但无 final_adapter | 看 `logs/*_train.err`，大概率是脚本报错 |
| Early Stopping 在 step 100-150 就触发 | **正常现象**，说明过拟合防护起效了，不是 Bug |

---

## ⏭️ 训练完成之后

把 Step 7 记下的**作业号**告诉我（例如："作业号 1456 跑完了"），我会继续给你：

- `04_inference_test.py` — 跑 5 条测试指令看生成效果
- `05_eval_compare.py` — V1 vs V2 定量对比（测试集 85 条）
- `06_serve_api.py` + `07_plot_loss.py` — 部署 API 和画 loss 曲线

---

**✨ 现在从 Step 1 开始，一步一步来。每一步做完如果有问题就停下发给我。**

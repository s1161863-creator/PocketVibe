# V2+ 推理评测 — 手把手操作指南

> 本文档面向"我下一步具体要敲哪条命令"而写。  
> 操作对象：V2+ 100 分制评测（5 条高难度 held-out 题）  
> 前置条件：V2 训练已完成（SLURM job 1442 成功）  
> 预计总耗时：**1.5 小时**（两张 A16 并行）

---

## 🎯 目标

在 HPC 上并行跑完两个评测作业，下载结果到本地，截图填到报告里。

- **Job A**：V2 单模型在 5 条高难度新题上的 100 分制表现（~30 分钟）
- **Job B**：V1 vs V2 在同 5 题上的并排对比（~60 分钟）

---

## 📋 总览：7 个步骤

| 步骤 | 地点 | 耗时 | 做什么 |
|---|---|---|---|
| 第 1 步 | 本地 PowerShell | 30 秒 | 打开终端并切到 Version2 目录 |
| 第 2 步 | 本地 PowerShell | 1 分钟 | 上传 5 个新文件到 HPC |
| 第 3 步 | HPC（SSH） | 30 秒 | 给两个 SLURM 加执行权限 |
| 第 4 步 | HPC（SSH） | 20 秒 | 同时提交两个 SLURM 作业 |
| 第 5 步 | HPC（SSH） | 30~60 分钟 | 监控日志、等跑完 |
| 第 6 步 | 本地 PowerShell | 2 分钟 | 下载 19 份产物到本地 |
| 第 7 步 | 本地浏览器 + 截图 | 30 分钟 | 打开 HTML 截图 + 填到报告 |

---

## 🚀 第 1 步：打开本地终端，切换到项目目录

> ⚠️ **重要**：本指南所有本地命令都**同时兼容 CMD 和 PowerShell**（写成单行，不用换行续行符）。  
> 你当前用的是 **CMD**（命令提示符）就按 CMD 列；想换 PowerShell（开始菜单搜索 "PowerShell"）也可以。

```cmd
cd /d "C:\Users\Lenovo\Desktop\Enoch - Version2"
```

确认 5 个新文件都在（CMD 用 `dir`，PowerShell 也认这个命令）：

```cmd
dir scripts\pv_scoring.py scripts\04_inference_test_v2plus.py scripts\05_eval_compare_v1_vs_v2_v2plus.py slurm\eval_v2_only.slurm slurm\eval_compare.slurm
```

看到 5 个文件全部列出、**没有 "File Not Found"** 就对了。

---

## 📤 第 2 步：上传 5 个新文件到 HPC

> 把下面命令里的 `student07` 换成你自己的账号（如果不是 student07）。  
> ⚠️ **每条命令写成单行，不要断行**。CMD 里反引号 `` ` `` 会报错。

### 方式 A：CMD / PowerShell 通用单行命令（推荐）

**上传 3 个 Python 脚本**（整条复制，一次性粘贴到终端，回车）：

```cmd
scp scripts\pv_scoring.py scripts\04_inference_test_v2plus.py scripts\05_eval_compare_v1_vs_v2_v2plus.py student07@aaillm.eduhk.hk:~/PocketVibe/scripts/
```

**上传 2 个 SLURM 脚本**：

```cmd
scp slurm\eval_v2_only.slurm slurm\eval_compare.slurm student07@aaillm.eduhk.hk:~/PocketVibe/slurm/
```

每条执行后要求你输密码 —— 输入 HPC 密码（不显示字符，瞎按就行），回车。每个文件出现 `100%` 就是成功。

**预期输出**（粗略看到 5 行 `100%` 即可）：

```
pv_scoring.py                   100%   XX KB
04_inference_test_v2plus.py     100%   XX KB
05_eval_compare_v1_vs_v2_v2plus.py   100%   XX KB
（输一次密码 → 上传 3 个脚本）

eval_v2_only.slurm              100%   XX KB
eval_compare.slurm              100%   XX KB
（再输一次密码 → 上传 2 个 SLURM）
```

### 方式 B：如果方式 A 有任何一个文件报错，逐个上传（最稳）

```cmd
scp scripts\pv_scoring.py student07@aaillm.eduhk.hk:~/PocketVibe/scripts/
scp scripts\04_inference_test_v2plus.py student07@aaillm.eduhk.hk:~/PocketVibe/scripts/
scp scripts\05_eval_compare_v1_vs_v2_v2plus.py student07@aaillm.eduhk.hk:~/PocketVibe/scripts/
scp slurm\eval_v2_only.slurm student07@aaillm.eduhk.hk:~/PocketVibe/slurm/
scp slurm\eval_compare.slurm student07@aaillm.eduhk.hk:~/PocketVibe/slurm/
```

每条独立一行，各输一次密码。共 5 次密码输入。

---

## 🔑 第 3 步：SSH 登录 HPC，给 SLURM 脚本加执行权限

```powershell
ssh student07@aaillm.eduhk.hk
```

进入 HPC 后：

```bash
cd ~/PocketVibe
chmod +x slurm/eval_v2_only.slurm slurm/eval_compare.slurm
```

无报错就表示成功。

---

## ▶️ 第 4 步：同时提交两个评测作业

```bash
# 提交 Job A（V2 单跑，约 30 分钟）
sbatch slurm/eval_v2_only.slurm

# 紧接着提交 Job B（V1 vs V2 对比，约 60 分钟）
sbatch slurm/eval_compare.slurm
```

每条命令会回一行类似：

```
Submitted batch job 1455
Submitted batch job 1456
```

**把这两个 job id 记下来**（后面监控日志要用）。

---

## 👀 第 5 步：监控进度

### 5.1 先确认两个作业都在跑

```bash
squeue -u $USER
```

应该看到类似：

```
JOBID  PARTITION  NAME       USER     ST  TIME  NODES NODELIST(REASON)
1455   gpu        pv_v2only  student07 R  0:30  1     node01
1456   gpu        pv_compare student07 R  0:25  1     node02
```

两行都是 `ST=R`（RUNNING）即正常。若显示 `ST=PQ`（PENDING/排队），等几分钟即可。

### 5.2 看 Job A 实时输出（另开一个 SSH 终端）

```bash
tail -f logs/1455_v2only.out
```

### 5.3 看 Job B 实时输出（再开一个 SSH 终端）

```bash
tail -f logs/1456_compare.out
```

> Ctrl+C 退出 `tail -f`，不会影响作业本身。

### 5.4 身份自证（必看）

Job B 的日志开头会打印两段 JSON，类似：

```
--- V1 adapter_config.json ---
{
  "base_model_name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "r": 32,
  "lora_alpha": 64,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  ...
}

--- V2 adapter_config.json ---
{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
  ...
}
```

**这两段是你报告的"V1/V2 身份对齐"铁证**，记得截图。

### 5.5 等结束

Job A ≈ 30 分钟，Job B ≈ 60 分钟。再次运行 `squeue -u $USER` 看不到你的 job，就是都跑完了。

---

## 📥 第 6 步：把产物下载到本地

**新开一个本地 CMD / PowerShell 窗口**（保留 SSH 那个不动），执行：

```cmd
cd /d "C:\Users\Lenovo\Desktop\Enoch - Version2"
```

**下载 Job A 产物**（V2 单跑，5 HTML + 3 报表 = 8 文件，单行命令）：

```cmd
scp student07@aaillm.eduhk.hk:~/PocketVibe/data/eval/v2plus_* data\eval\
```

**下载 Job B 产物**（V1 vs V2 对比，10 HTML + 3 报表 = 13 文件，单行命令）：

```cmd
scp student07@aaillm.eduhk.hk:~/PocketVibe/data/eval/compare_v1v2p_* data\eval\
```

**下载两份 SLURM 日志**（身份自证 + 汇总表，截图用）：

```cmd
scp student07@aaillm.eduhk.hk:~/PocketVibe/logs/*_v2only.out logs\
scp student07@aaillm.eduhk.hk:~/PocketVibe/logs/*_compare.out logs\
```

> 注意：CMD 对路径通配符 `*` 的处理不会本地展开（由 scp 在 HPC 端展开），所以上面命令可以直接跑。  
> 如果 CMD 报"文件名不正确"，把远端路径用双引号包起来：`"student07@aaillm.eduhk.hk:~/PocketVibe/data/eval/v2plus_*"`

下载完后，`data\eval\` 下应该有约 **21 份文件**（19 个产物 + 可能旧的保留）。

验证下载是否齐全：

```cmd
dir data\eval\v2plus_*
dir data\eval\compare_v1v2p_*
```

应看到前者 8 个、后者 13 个。

---

## 🖼 第 7 步：按报告要求截图

### 7.1 先建截图目录（如果还没有）

```powershell
mkdir "C:\Users\Lenovo\Desktop\EdUHK MSC AIEP\部署LLM\作业截图" -ErrorAction SilentlyContinue
```

### 7.2 打开下列 HTML 文件，每个截图保存（Chrome 按 F12 开手机模拟器，iPhone 12 Pro 即可）

| 要开的 HTML | 截图存成（放到作业截图目录） |
|---|---|
| `data/eval/v2plus_C1_depth_stopwatch_lap.html` | `04-C_v2plus_C1_depth_browser.jpg` |
| `data/eval/v2plus_C2_breadth_swim_timer.html` | `04-D_v2plus_C2_breadth_browser.jpg` |
| `data/eval/v2plus_C3_reasoning_calculator.html` | `04-E_v2plus_C3_reasoning_browser.jpg` |
| `data/eval/v2plus_C4_combination_todo_pomodoro.html` | `04-F_v2plus_C4_combination_browser.jpg` |
| `data/eval/v2plus_C5_cross_rps_scoreboard.html` | `04-G_v2plus_C5_cross_browser.jpg` |

### 7.3 并排对比截图（报告最有冲击力的部分）

Chrome 开两个窗口，左右并排，分别打开 V1 和 V2 同一题，合起来截图：

| 左窗口（V1） | 右窗口（V2） | 保存为 |
|---|---|---|
| `compare_v1v2p_C1_depth_stopwatch_lap_v1.html` | `compare_v1v2p_C1_depth_stopwatch_lap_v2.html` | `04-H_compare_C1_v1_vs_v2_sidebyside.jpg` |
| `compare_v1v2p_C3_reasoning_calculator_v1.html` | `compare_v1v2p_C3_reasoning_calculator_v2.html` | `04-I_compare_C3_v1_vs_v2_sidebyside.jpg` |
| `compare_v1v2p_C4_combination_todo_pomodoro_v1.html` | `compare_v1v2p_C4_combination_todo_pomodoro_v2.html` | `04-J_compare_C4_v1_vs_v2_sidebyside.jpg` |

> C4 是组合题（待办+番茄钟），**最能体现 V2 的优势**，必截。

### 7.4 日志/表格类截图（文本内容用 VSCode / 记事本打开后截图）

| 打开什么文件 | 截哪段 | 保存为 |
|---|---|---|
| `logs/*_compare.out` | 开头两段 V1/V2 adapter_config.json | `04-B_v1v2_adapter_identity_proof.jpg` |
| `logs/*_compare.out` | 结尾"V1 vs V2 对比总表"那段控制台输出 | `04-K_compare_summary_table.jpg` |
| `data/eval/v2plus_results.md` | 整个表格内容 | `04-L_v2plus_results_md_content.jpg` |
| HPC 终端 `squeue -u $USER` | 两个 job 都在 RUNNING 那一刻 | `04-A_v2plus_slurm_job_submitted.jpg` |

### 7.5 确认截图齐全

检查 `作业截图/` 目录下 `04-A` 到 `04-L` **共 12 张**全部存在。

---

## ✅ 完成自检清单

跑完上述 7 步后，检查以下条目：

- [ ] `data/eval/v2plus_results.md` 打开后能看到 5 行评分表（每行 5 维度分数）
- [ ] `data/eval/compare_v1v2p_results.md` 打开后能看到 V1/V2 并排对比表
- [ ] `作业截图/` 下有 `04-A` 到 `04-L` 共 12 张截图
- [ ] `logs/*_compare.out` 里能肉眼看到 V1 和 V2 两个 adapter 配置的 JSON（r=32 vs r=16）

全部 ✅ 即 V2+ 评测阶段完整结束。

---

## 🧯 常见故障排查

### Q1：`squeue -u $USER` 显示 `ST=PD`（PENDING）很久不动

**原因**：HPC GPU 队列拥挤。  
**对策**：耐心等，或 `scancel {JOB_ID}` 取消后下午再提交。单提交一个也可以（A 或 B 任选其一，另一个排队时再开）。

### Q2：`tail -f logs/*_compare.out` 看到 Python 报错

**常见原因**：
1. **`ModuleNotFoundError: pv_scoring`** → 第 2 步上传时漏了 `pv_scoring.py`。重新 scp 上传，再重新 sbatch。
2. **`FileNotFoundError: outputs/qlora-v2-run1/final_adapter`** → V2 adapter 路径变了。在 HPC 上用 `ls ~/PocketVibe/outputs/` 确认，必要时改 SLURM 里的路径。
3. **`CUDA out of memory`** → 不应该发生（1.5B × bf16 + LoRA < 8 GB），如真发生先用 `squeue -u $USER` 确认没有其他作业抢显存。

### Q3：生成的 HTML 在浏览器里显示空白

**诊断**：
1. VSCode 打开 HTML 文件看源码是否以 `<!DOCTYPE html>` 开头、以 `</html>` 结尾
2. 如果 `</html>` 缺失 → 输出被截断了，100 分制得分也会低
3. 这种情况**本身就是报告要讨论的数据**（说明该题 V2 也没解好），直接记录下来

### Q4：scp 下载报 `No such file or directory`

**原因**：评测作业还没跑完或失败了。  
**对策**：先 `ssh` 上去 `ls ~/PocketVibe/data/eval/` 看产物是否真的在那里。

---

## 📞 操作到某一步卡住了？

下次对话直接告诉我：
1. 你执行到第几步（1~7）
2. 最后一条命令的输出（复制粘贴整段终端，或截图）
3. `squeue -u $USER` 当前的状态

我会按这些信息精确诊断卡点。

---

*本指南专为 V2+ 评测阶段编写。V2 训练本身已在 SLURM job 1442 完成，本轮不涉及重训练。*

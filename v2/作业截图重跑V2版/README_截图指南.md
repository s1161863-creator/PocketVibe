# 作业截图重跑 V2 版 — 最终精简版截图清单

> **🎯 本次方案的核心决定**
> - V2 单模型 5 张 Demo **不截图**，分数已在 `v2plus_results.md` 报表里
> - 只做 **V1 vs V2 对比**（PowerPoint 拼接图，最能直击评分）
> - 总共 **20 张**截图，覆盖 Deployment 40% + Demo 25% + Doc 20% + Presentation 15%

---

## 📊 截图总览

| 类别 | 数量 | 状态 |
|---|---|---|
| **Part A** V2 训练阶段 | 8 | ✅ 已有 |
| **Part B** HPC 补证 | 5 | 🔴 今晚立刻做（1455 运行中） |
| **Part C** Demo 对比图 | 5 | 🟢 等 1455 跑完 |
| **Part D** Documentation | 2 | 🟢 等 1455 跑完 |
| **Part E** Presentation | 3 | 🔵 最后 5 分钟 |
| **合计** | **20 张** | — |

保存位置：`C:\Users\Lenovo\Desktop\Enoch - Version2\作业截图重跑V2版\`

---

## Part A：V2 训练阶段（✅ 已有 8 张，无需重截）

| 现有文件名 | 评分对应 | 报告引用位置 |
|---|---|---|
| `01_配置摘要_MAX_SEQ_4096.png` | Deployment：V2 超参配置 | §7 超参详解 / §10.1 bug 清单 |
| `02_LoRA参数与数据加载.png` | Deployment：模型 + 数据加载 | §6 执行流程 |
| `02_LoRA可训练参数_7层.png` | Deployment：LoRA 设计（7 层 targets / 1.18% trainable） | §3.1 LoRA 升级 |
| `03_完整loss序列.png` | Documentation：训练收敛曲线 | §10.2 训练结果 |
| `03_训练loss前20步.png` | Documentation：训练启动正常 | §10.2 附图 |
| `04_squeue作业状态.png` | Deployment：HPC 训练 Job 1442 运行中 | §6 执行流程 |
| `04_训练完成最终loss.png` | Documentation：train_loss=0.2480 收敛 | §10.2 |
| `05_final_adapter产物文件.png` | Deployment：86M adapter 产物 | §6 / §10.2 |

---

## Part B：HPC 补证（5 张，今晚 1455 跑着的时候截）

### 🔴 现在立刻做（1455 还在 R 状态）

#### 📸 01-A：`01-A_slurm_queue_new_compare_running.png`
**命令**：
```bash
squeue -u $USER
```
**要求**：画面看到 1455 的 ST=R（running）

---

#### 📸 01-B：`01-B_eval_slurm_scripts_cat.png`
**命令**：
```bash
cat slurm/eval_compare.slurm
```
**要求**：看到 `#SBATCH --gres=gpu:a16:1` 以及 Python 启动行

---

#### 📸 01-D：`01-D_hpc_hostname_env.png`
**命令**：
```bash
hostname && whoami && nvidia-smi -L
```
**要求**：看到 `aaillm`、`student07`、4 张 A16 GPU 列表

---

#### 📸 02-A：`02-A_adapter_identity_v1_v2.png`
**命令**：
```bash
echo "=========== V1 adapter_config ==========="
cat outputs/qlora-run1/final_adapter/adapter_config.json
echo ""
echo "=========== V2 adapter_config ==========="
cat outputs/qlora-v2-run1/final_adapter/adapter_config.json
```
**要求**：画面能看到两段对比，V1 是 `r=8, target_modules=[q_proj,k_proj,v_proj,o_proj]`（或 rank=32 的原始值），V2 是 `r=16, target_modules=7 层`

---

### 🟠 3 分钟后做

#### 📸 01-C：`01-C_nvidia_smi_during_eval.png`
**命令**：
```bash
nvidia-smi
```
**要求**：画面能看到 python 进程占用 ~3-4GB 显存

---

## Part C：V1 vs V2 对比 Demo（5 张，等 1455 跑完后做）

### 前置：下载产物到本地
```powershell
cd "C:\Users\Lenovo\Desktop\Enoch - Version2\data\eval"
scp "student07@aaillm.eduhk.hk:~/PocketVibe/data/eval/compare_v1v2p_*" .
```

### 拼接流程（每题约 5 分钟）

**步骤**：
1. Chrome 打开 `compare_v1v2p_CN_*_v1.html`
2. F12 → Ctrl+Shift+M → 顶部选 **iPhone 12 Pro**
3. 点点按钮让界面活起来（秒表按开始 / 待办加 2 条 / 石剪布玩 2 局）
4. Ctrl+Shift+P → 输入 `screenshot` → 选 **Capture full size screenshot**
5. 暂存为 `tmp_CN_v1.png`
6. 同样流程截 v2.html → `tmp_CN_v2.png`
7. **PowerPoint 拼接**：
   - 新建 16:9 空白幻灯片
   - 左半贴 v1 截图
   - 右半贴 v2 截图
   - 下方插入文本框：**左：V1（基线） 右：V2（Version2 新训练）**
   - 右键幻灯片 → 另存为图片（PNG）
8. 最终命名按下表

### 5 张对比图清单

| 最终截图 | V1 HTML | V2 HTML |
|---|---|---|
| `03-A_compare_C1_stopwatch.png` | `compare_v1v2p_C1_*_v1.html` | `compare_v1v2p_C1_*_v2.html` |
| `03-B_compare_C2_swim.png` | `compare_v1v2p_C2_*_v1.html` | `compare_v1v2p_C2_*_v2.html` |
| `03-C_compare_C3_calc.png` | `compare_v1v2p_C3_*_v1.html` | `compare_v1v2p_C3_*_v2.html` |
| `03-D_compare_C4_todo.png` | `compare_v1v2p_C4_*_v1.html` | `compare_v1v2p_C4_*_v2.html` |
| `03-E_compare_C5_rps.png` | `compare_v1v2p_C5_*_v1.html` | `compare_v1v2p_C5_*_v2.html` |

---

## Part D：Documentation（2 张）

### 📸 04-A：`04-A_compare_v1v2_results_md.png`
**操作**：VSCode 打开 `data/eval/compare_v1v2p_results.md` → 按 `Ctrl+Shift+V` 打开 Markdown Preview → 截打分表部分

### 📸 04-D：`04-D_console_summary_table.png`
**操作**（在 HPC 终端 1455 跑完后）：
```bash
tail -40 $(ls -t logs/*compare*.out | head -1)
```
截画面里的最终 V1 vs V2 打分汇总表。

---

## Part E：Presentation（3 张）

### 📸 05-A：`05-A_project_tree_vscode.png`
VSCode 打开 `Enoch - Version2` 根目录 → 左侧 Explorer 展开 `scripts/`、`slurm/`、`data/eval/`、`作业截图重跑V2版/` → 截图

### 📸 05-B：`05-B_readme_toc_rendered.png`
VSCode 打开 `README.md` → `Ctrl+Shift+V` → 截目录部分（能看到 §10 开发记录 6 个子章节）

### 📸 05-C：`05-C_data_eval_folder_listing.png`
Windows 资源管理器打开 `Enoch - Version2\data\eval\` → 切换到"详细信息"视图 → 截图

---

## 📌 报告截图引用映射表

| 报告章节 | 必用截图 |
|---|---|
| §3 V2 方法论 | `02_LoRA可训练参数_7层.png` |
| §6 执行流程 | `01-B`、`02_LoRA参数与数据加载.png`、`04_squeue作业状态.png`、`05_final_adapter产物文件.png` |
| §7 超参配置 | `01_配置摘要_MAX_SEQ_4096.png` |
| §10.1 Bug 清单 | `01_配置摘要_MAX_SEQ_4096.png` |
| §10.2 训练结果 | `03_完整loss序列.png`、`04_训练完成最终loss.png`、`05_final_adapter产物文件.png` |
| §10.5 V2+ 评测方法论 | `02-A`（adapter 身份）、`04-A`（V1vsV2 打分表）、`04-D`（控制台） |
| §11 Demo 章节 | `03-A ~ 03-E`（5 张 V1vsV2 对比图） |
| §12 结论 | `05-B`（README 目录） |
| 附录 A 可复现性 | `01-D`（HPC 环境）、`05-A`（项目树）、`05-C`（eval 文件清单） |
| 附录 B GPU 证据 | `01-A`（squeue）、`01-C`（nvidia-smi） |

---

## 🗓 今晚执行时间轴（配合 `今晚执行清单.md` 勾选）

**阶段 1 — 现在（1455 在 R 状态）**（5 分钟）
1. 截 01-A（squeue）
2. 截 01-B（slurm 脚本）
3. 截 01-D（hostname/GPU）
4. 截 02-A（adapter 身份）

**阶段 2 — 提交后 3 分钟**（30 秒）
5. 截 01-C（nvidia-smi）

**阶段 3 — 等待 1455（约 25 分钟）**
- 可以做别的事 / 休息

**阶段 4 — 1455 跑完**（10 分钟）
6. HPC 截 04-D（控制台汇总）
7. 本地 scp 下载 compare_v1v2p_* 到 data/eval/

**阶段 5 — 本地截对比图**（25 分钟）
8~12. 截 03-A ~ 03-E（PowerPoint 拼接 5 张）

**阶段 6 — 扫尾**（5 分钟）
13. 截 04-A（compare md 渲染）
14. 截 05-A（项目树）
15. 截 05-B（README 目录）
16. 截 05-C（eval 文件夹）

---

*本清单最终版。最后更新：2026-05-12 19:40*

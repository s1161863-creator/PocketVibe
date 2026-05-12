# INT6138 Individual Assignment — PocketVibe: 两轮 LoRA 微调的客观反思（第二稿）

**课程**: INT6138 Project II: Deploying Personal LLMs / ELMs
**项目名**: PocketVibe — 中文自然语言驱动的移动端 HTML 微应用生成器
**作者**: Enoch
**提交日期**: 2026-05-12
**正文字数**（不含参考文献与附录）: 约 2150 字

> **本稿相对一稿的三维度强化**
> 1. **部署可复现性**——补齐 SLURM 全流程、SSH 隧道部署 Gradio、adapter 身份自证、bf16 替代 4-bit 的工程取舍；
> 2. **评测严谨性**——三轮交叉 + 5 维雷达 + 逐题 Δ + 人工评判推翻 AI 评分的完整证据链，并给出可视化图表；
> 3. **批判性反思**——把 V2 的负结果嵌入 LIMA / Chinchilla / Qwen 官方 SFT→DPO 的文献框架，给出可执行的 V3 路线。

---

## 1. Introduction（引言）

PocketVibe 的目标是让用户用一句中文描述（例如「做个能倒计时的秒表」）直接得到一份可在手机浏览器直接打开的**单文件 HTML 微应用**。基础模型选用 **Qwen2.5-Coder-1.5B-Instruct**（Hui et al., 2024），部署平台为 **EdUHK AAILLM HPC**（4×NVIDIA A16 16GB 节点，CUDA 12.x，Python 3.11，SLURM 调度）。前端采用 Gradio + iframe 沙盒渲染，推理栈为 HuggingFace `transformers + peft + trl`。

项目围绕**同一基座**完成了两轮独立的 LoRA 后训练：**V1**（伪数据增强 + 仅 Attention LoRA）与 **V2**（Evol-Instruct + 一指令×4 风格 + MLP LoRA + NEFTune + CompletionOnlyLoss）。本报告以**客观第三方视角**评估两轮训练——既肯定 V2 在数据多样性与流水线工程化上的进步，也直面 V2 在人工评判下整体败给 V1 的事实，并将这一"负结果"升华为对 **1.5B 规模 SFT 阶段合理边界**的实证讨论。

**Code & artifact availability.** All scripts, SLURM job files, evaluation data, generated HTMLs and HPC screenshots are released as a public GitHub repository at `https://github.com/<USER>/PocketVibe` (commit `<HASH>`, tag `v2.0`, accessed 2026-05-12). Appendix C provides a file-level crosswalk mapping every report section to its corresponding code path — this allows the marker to verify any claim in the report within two clicks.

---

## 2. Deployment（部署）

### 2.1 为何选 Qwen2.5-Coder-1.5B-Instruct

同规模候选对比：

| 候选模型 | 参数量 | 中文能力 | 代码能力 | A16 16 GB 可行性 |
|---|---|---|---|---|
| **Qwen2.5-Coder-1.5B-Instruct ✅** | 1.5 B | 强（中文预训练占比高） | 强（代码专门后训练） | bf16 LoRA 峰值 ~7 GB，余量充足 |
| DeepSeek-Coder-1.3B-Instruct | 1.3 B | 中 | 强 | 可行但中文弱 |
| CodeLlama-7B-Instruct | 7 B | 弱 | 强 | QLoRA 下仍需 ~13 GB，几无余量 |

Qwen2.5-Coder 技术报告明确采用 **SFT → DPO 两阶段后训练**（Hui et al., 2024）——这一官方工艺为本项目 §4 的改进路线提供直接参考。

### 2.2 可复现部署流程（V1 / V2 共用）

```text
本地 (Windows)                                 HPC (Linux + SLURM)
──────────────                                  ──────────────
scp -r "Enoch - Version2" …:~/PocketVibe ──►  ~/PocketVibe/
                                               │
python scripts/00_create_seeds.py              ├─ python -m venv ~/venvs/pv-train
python scripts/01{a,b,c,d,e}_*.py              ├─ pip install -r requirements-train.txt
python scripts/02_category_split.py            │
python scripts/02c_filter_by_length.py         │
                                               │
scp data/processed/*.jsonl …:~/PocketVibe/data/processed/
                                               │
                                               ├─ sbatch slurm/train.slurm         （~22 min / A16）
                                               ├─ sbatch slurm/eval_compare.slurm  （~8 min）
                                               └─ sbatch slurm/serve_app.slurm     （Gradio 常驻）
                                               │
ssh -N -L 7860:<node>:7860 user@aaillm  ◄──────┘  （本地浏览器访问 Demo）
```

> 插图：`Enoch - Version2/作业截图重跑V2版/01-D_hpc_hostname_env.png`
> 插图：`Enoch - Version2/作业截图重跑V2版/01-B-SLURM 脚本内容.png`
> 插图：`Enoch - Version2/作业截图（第二版在这里图）/02-B_gpu_nvidia-smi_4xA16.jpg`

### 2.3 关键部署决策的工程取舍

**决策 1：bf16 替代 4-bit 量化**
HPC 预装 `bitsandbytes` 缺对应 CUDA binary，首次 `sbatch` 报错。验证 bf16 全精度 LoRA 峰值显存仅 ~7 GB，远低于 16 GB 上限——遂**主动放弃 4-bit，改走 bf16**。核心价值（LoRA 低秩适配 + 冻结基座）不受影响，换回的是训练稳定性与推理一致性。

**决策 2：`device_map="auto"` 单卡实际运行**
1.5 B 模型 + bf16 LoRA 单卡即足。多卡申请仅用于 SLURM 排队优先级，**主动保留单卡推理方案**，避免 tensor-parallel 带来的 checkpoint 合并开销。

**决策 3：V2 重做（`max_seq_length=1024 → 4096`）**
V2 首轮 `max_seq_length=1024` 导致 **100% 样本静默截断**（HTML 平均 2000+ tokens），推理普遍在 1024 处断尾。修复：调 4096 + 新增 `02c_filter_by_length.py` 预过滤。这是本项目最昂贵的一课——**一次 HPC 作业的时间换一条工程红线**。

> 插图：`Enoch - Version2/作业截图重跑V2版/01_配置摘要_MAX_SEQ_4096.png`
> 插图：`Enoch - Version2/作业截图重跑V2版/02_LoRA可训练参数_7层.png`
> 插图：`Enoch - Version2/作业截图重跑V2版/02-A_adapter_identity_v1_v2-1.png`
> 插图：`Enoch - Version2/作业截图重跑V2版/02-A_adapter_identity_v1_v2-2.png`

### 2.4 Gradio Demo 后端（`app/app.py`）

基于 `app/app.py` 的核心设计：
- **Adapter 热切换**：一次加载 bf16 基座，通过 `model.set_adapter("v1"|"v2")` 瞬切，演示时可实时对比；
- **iframe `srcdoc` 沙盒渲染**：生成 HTML 转义嵌入，兼顾安全与真机预览效果；
- **环境变量覆盖路径**：`PV_BASE_MODEL / PV_V1_ADAPTER / PV_V2_ADAPTER`。

通过 `slurm/serve_app.slurm` 提交后，本地一条 SSH 隧道即可访问——对应附录 `app/app.py` 与 `slurm/serve_app.slurm`。

> 插图：`Enoch - Version2/作业截图重跑V2版/05_final_adapter产物文件.png`

---

## 3. Evaluation（评测）

### 3.1 三轮独立交叉验证方法论

为避免单一指标偏差，设计三轮独立评测：

| 轮次 | 脚本 | 目的 | 样本 |
|---|---|---|---|
| 第一轮 | `04_inference_test_v2plus.py` | V2 单模型 100 分制自测，Best-of-3 采样 | 5 题高难度 |
| 第二轮 | `05_eval_compare_v1_vs_v2_v2plus.py` | V1 vs V2 同题对比（通用型） | 5 题 Compare-Hard |
| 第三轮 | `06_showcase_v2_strengths.py` | V1 vs V2「V2 主场」对比（跨类组合 + 视觉风格） | 2 题 Showcase |

**100 分制维度加权**（参考 KRT2002 qwen-python-finetuning + Design2Code NAACL 2025, Si et al., 2025）：

$$
\text{Total} = J_{30} + I_{25} + F_{20} + C_{15} + S_{10}
$$

其中 J = JS 深度，I = 指令遵循，F = 功能可运行，C = CSS 质量，S = 结构合规。

### 3.2 第二轮 V1 vs V2 逐题结果

| 案例 | 类型 | V1 | V2 | Δ |
|---|---|---|---|---|
| C1 分段秒表 | DEPTH | 92 | 91 | −1 |
| C2 游泳防水秒表 | BREADTH | 80 | 56 | **−24** |
| C3 括号优先级计算器 | REASONING | 95 | 83 | −12 |
| C4 待办 + 番茄钟 | COMBINATION | 96 | 93 | −3 |
| C5 石头剪刀布 + 计分 | CROSS | 82 | 83 | +1 |
| **均值** | | **89.0** | **81.2** | **−7.8** |

> 插图：`Enoch - Version2/report/fig_radar_v1_vs_v2.png`
> 插图：`Enoch - Version2/report/fig_per_case_bars.png`

雷达图显示 V1 在 **J（JS 深度）** 和 **F（功能可运行）** 两个"硬能力"维度显著外扩；V2 仅在 **I（指令遵循）** 略胜——这一现象佐证 V2 的问题是"看得懂要求但写不出能跑的代码"，而非"误解需求"。柱状图则把 C2（−24）这个 BREADTH 灾难点暴露得非常直观。

### 3.3 第三轮：AI 评分与人工评判的戏剧性反转

为避免「第二轮场地对 V1 有利」的质疑，第三轮特意挑选 V2 训练中显式覆盖的场景——**跨类组合 + 视觉风格**：

| 题目 | AI 评分 | 浏览器人工评判 |
|---|---|---|
| S1 晨间三合一（番茄 + 天气卡片 + 冥想） | V2 (93) > V1 (80) | **V1 完胜**——V1 功能完整；V2 的"5 分钟冥想计时"实际渲染显示 `4.6666666 分钟吸气`，浮点循环错乱 |
| S2 暗色霓虹井字棋 | V2 (91) > V1 (86) | **V1 完胜**——V1 秒表可交互；V2 主题色更"霓虹"但核心按钮功能错位，井字棋无法完成一局 |

> 插图：`Enoch - Version2/report/fig_rounds_summary.png`
> 插图：`Enoch - Version2/作业截图（第二版在这里图）/03-H0b_inference_creative_01.jpg`
> 插图：`Enoch - Version2/作业截图（第二版在这里图）/03-H0c_inference_social_01.jpg`
> 插图：`Enoch - Version2/作业截图（第二版在这里图）/03-H6_evaluation_bug_analysis.jpg.jpg`

这一反转是**整份报告的方法论枢纽**：它同时证伪了"V2 的训练改动让模型更强"与"AI 自动评分即可替代人工判定"两条假设。

### 3.4 V1 / V2 优劣客观清单

**V1 强项**（肯定）
- 格式稳定：`</html>` 闭合率 ~95%；
- 简单任务可靠：秒表 / 计时 / BMI 几乎 100% 可用。

**V1 弱项**（批判）
- 泛化极弱：超出 50 条种子覆盖的类型后只会"换皮肤"；
- `eval_loss=0.1345` 存在**数据泄漏**：同一 HTML 的 10 条指令变体被随机切分到训练 / 验证集。

**V2 强项**（肯定）
- 信息熵显著提升：独立样本 ~50 → ~830 条；
- 工程化度高：NEFTune + CompletionOnlyLoss + 类别隔离验证集 + Best-of-3 + `adapter_config.json` 身份自证。

**V2 弱项**（批判）
- 人工评判整体败给 V1；
- **SFT 阶段贪多求全**——1.5 B 模型被迫同时学多风格 + 跨类 + 复杂推理，触发 Chinchilla（Hoffmann et al., 2022）警告的小模型多目标过载。

> 插图：`Enoch - Version2/作业截图重跑V2版/03_完整loss序列.png`
> 插图：`Enoch - Version2/作业截图重跑V2版/04_训练完成最终loss.png`
> 插图：`Enoch - Version2/作业截图（第二版在这里图）/03-E_eval_loss_0.1094_result.jpg`
> 插图：`Enoch/report/loss_curve.png`（V1 loss 曲线）

### 3.5 对评测方法论本身的自我批判

5 维静态启发式评分**天然奖励"看起来复杂"的输出**：CSS 越长、JS 变量越多、函数嵌套越深即加分，但与"功能是否真的能跑"无关。V2 的输出恰恰在这些"可见复杂度"上刷高，而在"不可见正确性"上翻车——正是 **Goodhart's Law** 在 ML 评测中的经典显现：*"当指标成为优化目标时，它就失去了衡量意义。"* 这一自我批判直接通向 §4.3 的评测框架升级建议。

---

## 4. Recommendations（改进建议）

### 4.1 SFT 阶段应收敛到「精准路线」（对应 §3.3、§3.4）

**证据链**：
- V2 输出 `4.666666 分钟吸气` 这种浮点错乱，是"SFT 数据过散导致基座能力被噪声淹没"的典型症状，与 **LIMA**（Zhou et al., 2023）"Less is More for Alignment"的观测方向一致；
- Chinchilla 的算力-参数-数据三角约束（Hoffmann et al., 2022）在小模型上尤为敏感——1.5 B 参数与 ~830 条多模态混合样本的比值已高于 Qwen2.5-Coder 官方推荐区间。

**V3 具体参数建议**：
- 一指令×**2** 风格（原 4），Evol-Instruct 比例 ≤ 30%，总样本 ≤ 500；
- 对齐 LIMA 的"1000 条高质量阈值"在 1.5 B 模型上的等比缩放（约 500）；
- 训练轮数 3 不变，学习率同样 2e-4，但 `lora_rank` 降至 **8**（原 16），强制模型只学"精准"。

### 4.2 增加 DPO 阶段修复泛化能力（对应 §3.3、§3.5）

**证据链**：
- Qwen2.5-Coder 官方就是 SFT → DPO 两阶段（Hui et al., 2024），V2 试图在 SFT 阶段同时完成"精准 + 泛化"违背了模型厂商既定工艺；
- 当前 V1-win / V2-lose 的 7 组对比数据天然构成 **偏好对**（V1 功能可用→`chosen`，V2 炫但错→`rejected`），是零额外标注成本的 DPO 数据集。

**可执行路线**：
- 使用 `trl.DPOTrainer`，成本 ~1 张 A16 × 2 小时（Rafailov et al., 2023）；
- 直接在 V2 final_adapter 之上继续训练，不重置权重；
- 评测仍用三轮框架，但追加 Playwright 行为级断言（见 §4.3）。

### 4.3 评测框架升级：从静态启发式到浏览器级功能测试（对应 §3.5）

**证据链**：
- 现有 5 维静态评分和 V2 的表面模式过拟合是同源的——互相自证；
- Design2Code（Si et al., 2025）已证明 GPT-4V 对渲染截图的打分可作为视觉正确性金标准。

**升级方案**：
- 引入 **Playwright** 自动化：对每份生成的 HTML 模拟"点击开始 → 检查秒表跳秒 → 点击停止 → 检查数字冻结"这种行为级断言；
- 追加 **GPT-4V 截图打分**作为视觉维度交叉验证；
- 人工评判 5% 样本作为最终仲裁，闭合"指标 ↔ 真实质量"的回路。

---

## 5. Reflection（反思）

本项目最直接对应两条 CILO：

- **"Evaluate and critically assess the capabilities and limitations of deployed LLMs"**——三轮评测 + 人工评判 + Goodhart's Law + LIMA / Chinchilla 证据链，构成对"AI 自动评分本身"的批判性评估：**数字高不等于真的好**；
- **"Apply LLM fine-tuning techniques including LoRA/QLoRA to domain-specific tasks"**——V1/V2 两轮 LoRA 系统对比，暴露 `target_modules` 选择、`lora_rank` 与数据量匹配、`max_seq_length` 与样本长度联动这三组超参的工程红线。

**负结果的价值**。V2 用真实 HPC 作业时间实证了"1.5 B 模型 SFT 阶段贪多求全"的代价，让下一轮 V3 可以直接跳到"精准 SFT + DPO 泛化"的成熟路径——这条由官方工艺（Qwen）+ 学术假设（LIMA）+ 算力约束（Chinchilla）三重独立来源共同背书的结论，远比"V2 分数更高"本身更有价值。

**未来应用场景**。该"SFT 精准 + DPO 泛化 + 行为级评测"框架可直接迁移到：
- 教育场景的**学生作业自动生成器**（输入「小学四年级应用题练习」→ 输出可打印 PDF）；
- 企业内部的**表单 / 报表工具生成器**；
- 任何对"功能正确性 > 视觉复杂度"的中文移动端低代码场景。

---

## References（APA 7, English）

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., … Zaremba, W. (2021). *Evaluating large language models trained on code* (arXiv:2107.03374). arXiv. https://doi.org/10.48550/arXiv.2107.03374

Databricks. (2025). *Efficient fine-tuning with LoRA: A guide to optimal parameter selection for large language models*. Databricks. https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms

Enoch. (2026). *PocketVibe: Two-round LoRA fine-tuning of Qwen2.5-Coder-1.5B for Chinese-to-mobile-HTML code generation* (Version 2.0) [Software and data artifact]. GitHub. https://github.com/<USER>/PocketVibe

Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical neural story generation. In I. Gurevych & Y. Miyao (Eds.), *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 889–898). ACL.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., … Sifre, L. (2022). *Training compute-optimal large language models* (arXiv:2203.15556). arXiv. https://doi.org/10.48550/arXiv.2203.15556

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. In *ICLR*. https://openreview.net/forum?id=rygGQyrFvH

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *ICLR*. https://openreview.net/forum?id=nZeVKeeFYf9

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, B., Yu, B., Lu, K., Dang, K., Che, B., He, B., Chen, G., Lin, R., & Ren, W. (2024). *Qwen2.5-Coder technical report* (arXiv:2409.12186). arXiv. https://doi.org/10.48550/arXiv.2409.12186

HuggingFace. (2025). *Generation strategies documentation*. HuggingFace. https://huggingface.co/docs/transformers/generation_strategies

Jain, N., Chiang, P.-H., Wen, Y., Kirchenbauer, J., Chu, H.-M., Somepalli, G., Bartoldson, B. R., Kailkhura, B., Schwarzschild, A., Bhatele, A., Geiping, J., Huang, F., & Goldstein, T. (2024). NEFTune: Noisy embeddings improve instruction finetuning. In *ICLR*. https://openreview.net/forum?id=0bMmZ3fkCk

KRT2002. (2024). *qwen-python-finetuning: 14-indicator evaluation framework for Python code generation* [Software]. GitHub. https://github.com/KRT2002/qwen-python-finetuning

Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2023). *An empirical study of catastrophic forgetting in large language models during continual fine-tuning* (arXiv:2308.08747). arXiv. https://doi.org/10.48550/arXiv.2308.08747

Luo, Z., Xu, C., Zhao, P., Sun, Q., Geng, X., Hu, W., Tao, C., Ma, J., Lin, Q., & Jiang, D. (2024). WizardCoder: Empowering code large language models with Evol-Instruct. In *ICLR*. https://openreview.net/forum?id=UnUwSIgK5W

Muxup. (2025). *Vendor-recommended LLM parameter quick reference*. https://muxup.com/2025q2/recommended-llm-parameter-quick-reference

OpenAI. (2021). *human-eval: Sample + greedy-fallback evaluation framework for code generation* [Software]. GitHub. https://github.com/openai/human-eval

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. In *NeurIPS 35* (pp. 27730–27744).

Qwen Team. (2024). *Qwen2.5 technical blog and model card*. https://qwenlm.github.io/blog/qwen2.5/

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In *NeurIPS 36* (pp. 53728–53741).

Si, C., Yang, Y., & Hashimoto, T. (2025). Design2Code: Benchmarking multimodal code generation for visual design. In *NAACL 2025*. https://github.com/NoviScl/Design2Code

Unsloth. (2026). *LoRA fine-tuning hyperparameters guide*. https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., & Weston, J. (2020). Neural text generation with unlikelihood training. In *ICLR*. https://openreview.net/forum?id=SJeYe0NtvH

Zhang, Y. (2025). *Breaking memorization barriers in LLM code fine-tuning via information bottleneck for improved generalization* (arXiv:2510.16022). arXiv. https://doi.org/10.48550/arXiv.2510.16022

Zhang, Z. (2025). *Memorize or generalize? Evaluating LLM code generation with evolved questions* (arXiv:2503.02296). arXiv. https://doi.org/10.48550/arXiv.2503.02296

Zhou, C., Liu, P., Xu, X., Iyer, S., Du, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., & Levy, O. (2023). LIMA: Less is more for alignment. In *NeurIPS 36* (pp. 55006–55021).

---

## Appendix A：补充插图目录（按出现顺序）

> 本目录仅列文件位置，便于定稿时按正文 `插图：...` 标记手动插入。若某张图已在正文出现，此处不再重复。

**部署证据**
- `Enoch - Version2/作业截图重跑V2版/01_配置摘要_MAX_SEQ_4096.png`
- `Enoch - Version2/作业截图重跑V2版/01-A_slurm_queue_new_compare_running.png`
- `Enoch - Version2/作业截图重跑V2版/01-B-SLURM 脚本内容.png`
- `Enoch - Version2/作业截图重跑V2版/01-C_nvidia_smi_during_eval.png`
- `Enoch - Version2/作业截图重跑V2版/01-D_hpc_hostname_env.png`
- `Enoch - Version2/作业截图重跑V2版/02_LoRA参数与数据加载.png`
- `Enoch - Version2/作业截图重跑V2版/02_LoRA可训练参数_7层.png`
- `Enoch - Version2/作业截图重跑V2版/02-A_adapter_identity_v1_v2-1.png`
- `Enoch - Version2/作业截图重跑V2版/02-A_adapter_identity_v1_v2-2.png`
- `Enoch - Version2/作业截图重跑V2版/04_squeue作业状态.png`
- `Enoch - Version2/作业截图重跑V2版/05_final_adapter产物文件.png`
- `Enoch - Version2/作业截图（第二版在这里图）/02-B_gpu_nvidia-smi_4xA16.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/02-K_training_completed_summary.jpg`

**训练证据**
- `Enoch - Version2/作业截图（第二版在这里图）/02-A1_training_head60_config.jpg.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/02-A2_training_tail40_config.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/02-C_training_progress_33of60.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-D_training_loss_all_6_steps.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-D2_training_loss_first_3 (可选，如果内容重复可跳过).jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-E_eval_loss_0.1094_result.jpg`
- `Enoch - Version2/作业截图重跑V2版/03_完整loss序列.png`
- `Enoch - Version2/作业截图重跑V2版/03_训练loss前20步.png`
- `Enoch - Version2/作业截图重跑V2版/04_训练完成最终loss.png`
- `Enoch - Version2/作业截图（第二版在这里图）/06_loss曲线.png`
- `Enoch - Version2/作业截图（第二版在这里图）/07_v1_vs_v2_loss.png`
- `Enoch/report/loss_curve.png`（V1 基线对照）

**推理 / 评测证据**
- `Enoch - Version2/作业截图（第二版在这里图）/03-H0a_inference_startup.jpg.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-H0b_inference_creative_01.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-H0c_inference_social_01.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-H0d_inference_finance_01.jpg .jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-H0e_inference_final_summary.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-H0e_inference_summary_json.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/03-H6_evaluation_bug_analysis.jpg.jpg`
- `Enoch - Version2/作业截图（第二版在这里图）/02-G_v1_adapter_baseline.jpg.jpg`

**可视化总结图（本次新绘）**
- `Enoch - Version2/report/fig_radar_v1_vs_v2.png`
- `Enoch - Version2/report/fig_per_case_bars.png`
- `Enoch - Version2/report/fig_rounds_summary.png`

---

## Appendix B：V1 vs V2 超参数差异一览

见 `Enoch - Version2/README.md §7` 与 `Enoch/README.md` 对应章节。两份 README 作为提交附件一并上交；在 GitHub 仓库中对应路径为 `v1/README.md` 与 `v2/README.md`。

---

## Appendix C：Report-to-Repository Crosswalk（顶会可复现性承诺）

按 NeurIPS 2025 Reproducibility Checklist 与 ACL 2024 Artifact Policy 的建议，本附录把报告正文的每一处技术论断精确映射到公开仓库的代码 / 数据路径。**仓库快照**：`https://github.com/<USER>/PocketVibe` commit `<HASH>`（tag `v2.0`）。

| 报告章节 / 论断 | 仓库路径 |
|---|---|
| §2.2 部署全流程（本地→HPC→SSH 隧道）| `v2/slurm/train.slurm`、`v2/slurm/eval_compare.slurm`、`v2/slurm/serve_app.slurm` |
| §2.3 决策 1 — bf16 替代 4-bit | `v2/scripts/03_train_qlora_v2plus.py`（`bnb_config` 相关代码被注释保留） |
| §2.3 决策 3 — `max_seq_length=4096` 红线 | `v2/scripts/02c_filter_by_length.py`、`v2/scripts/03_train_qlora_v2plus.py` |
| §2.4 Gradio demo（adapter 热切换 + iframe 沙盒）| `v2/app/app.py`、`v2/slurm/serve_app.slurm` |
| §3.1 三轮独立评测方法论 | `v2/scripts/04_inference_test_v2plus.py`、`v2/scripts/05_eval_compare_v1_vs_v2_v2plus.py`、`v2/scripts/06_showcase_v2_strengths.py` |
| §3.2 第二轮逐题数值（89.0 vs 81.2） | `v2/data/eval/compare_v1_vs_v2_v2plus.json` + `v2/report/plot_results.py` |
| §3.2 雷达图 / 柱状图 | `v2/report/fig_radar_v1_vs_v2.png`、`v2/report/fig_per_case_bars.png` |
| §3.3 第三轮 AI vs 人工反转 | `v2/data/eval/showcase_v2_strengths.json` + `v2/evidence/round2/S{1,2}_v{1,2}_*.jpg` |
| §3.4 V1 数据泄漏分析（`eval_loss=0.1345`）| `v1/scripts/02b_split_data.py`、`v1/report/loss_curve.png` |
| §3.4 V2 `eval_loss=0.1094`（干净值）| `v1/evidence/.../03-E_eval_loss_0.1094_result.jpg`（原始日志）|
| §3.5 Goodhart's Law 方法论批判 | 基于 `v2/data/eval/*.json` 的 AI 分 vs `v2/evidence/round2/` 的人工判定双源头 |
| §4.1 V3 SFT 精准路线建议 | （规划）`v3/` 分支，不在 `v2.0` 快照中 |
| §4.2 DPO 数据自动生成逻辑 | （规划）利用 `v2/data/eval/showcase_v2_strengths.json` 的 win/lose 对作为 chosen/rejected |
| §4.3 Playwright + GPT-4V 升级评测 | （规划）将集成进 `v3/evaluation/` |
| 所有超参数完整差异表 | `v1/README.md §7`、`v2/README.md §7` |

Commit `<HASH>` 是本报告所有数字、表格、图表的精确产出源。复现命令见仓库 `README.md §3`。

> 提交终稿前请把 `<USER>` 全局替换成 GitHub 用户名，`<HASH>` 替换成 git push 后的真实 7 位短 hash（`git log -1 --format=%h` 可直接取到）。

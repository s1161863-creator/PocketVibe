# INT6138 Individual Assignment — PocketVibe: 两轮 LoRA 微调的客观反思

**课程**: INT6138 Project II: Deploying Personal LLMs / ELMs
**项目名**: PocketVibe — 中文自然语言驱动的移动端 HTML 微应用生成器
**作者**: Enoch
**提交日期**: 2026-05-12
**正文字数**（不含参考文献与附录）: 约 1680 字

---

## 1. Introduction（引言，约 160 字）

PocketVibe 的目标是让用户用一句中文描述（如"做个能倒计时的秒表"）直接得到一份可在手机浏览器打开的**单文件 HTML 微应用**。基础模型选用 **Qwen2.5-Coder-1.5B-Instruct**（阿里通义实验室, Hui et al., 2024），部署平台为 **EdUHK AAILLM HPC**（4×NVIDIA A16 16GB 节点，CUDA 12.x，Python 3.11, SLURM 调度），前端拟用 Gradio，推理框架为 HuggingFace `transformers + peft + trl`。项目围绕**同一基座** 做了两轮独立的 LoRA 后训练：**V1（初版，伪数据增强 + 仅 Attention LoRA）** 与 **V2（改进版，Evol-Instruct + 多风格 + MLP LoRA + NEFTune）**。本报告以**客观第三方视角**评估两轮训练，既肯定 V2 在数据多样性与训练流水线工程化上的进步，也直面 V2 在人工评判下整体败给 V1 的事实，最终收敛到一个被 LIMA / Qwen 官方 / InstructGPT / DPO 等多条独立文献链支撑的结论：**1.5B 参数规模的 LoRA 后训练，SFT 阶段应专攻"精准"，"泛化"应留给后续 DPO/RLHF 阶段**。

## 2. Deployment（部署，约 340 字）

### 2.1 为何选 Qwen2.5-Coder-1.5B-Instruct

同规模候选模型比较：

| 候选 | 参数量 | 中文能力 | 代码能力 | A16 16GB 可行性 |
|---|---|---|---|---|
| **Qwen2.5-Coder-1.5B-Instruct** ✅ | 1.5B | 强（中文预训练占比高） | 强（专门代码后训练） | bf16 LoRA 峰值 ~7GB，宽裕 |
| DeepSeek-Coder-1.3B-Instruct | 1.3B | 中 | 强 | 可行但中文弱 |
| CodeLlama-7B-Instruct | 7B | 弱 | 强 | QLoRA 下也要 ~13GB，无余量 |

Qwen2.5-Coder 的官方技术报告明确提到**两阶段后训练流程（SFT → DPO）**（Hui et al., 2024），这对本项目后续路线有直接参考价值。

### 2.2 可复现部署流程（V1 / V2 共用）

1. `scp` 本地项目目录至 HPC：`scp -r "Enoch - Version2" student07@aaillm.eduhk.hk:~/PocketVibe`
2. 远程创建虚拟环境：`python -m venv ~/venvs/pv-train && pip install -r requirements-train.txt`
3. 数据流水线（本地）：`00_create_seeds.py → 01a/01b/01c → 01d merge → 01e validate → 02 category_split.py → 02c filter_by_length.py`
4. HPC SLURM 提交：`sbatch slurm/train.slurm`（1 张 A16, ~22 min）
5. 推理评测：`sbatch slurm/eval_compare.slurm`（V1 vs V2 对比）

### 2.3 工程踩坑三例（节选）

- **Bug A**: HPC bitsandbytes 缺 CUDA binary → 放弃 4-bit 改走 **bf16 全精度 LoRA**（显存仍足，不影响核心价值）
- **Bug F（V2 重做起因）**: V2 首轮 `max_seq_length=1024` 导致 **100% 样本被静默截断**，推理输出普遍在 1024 tokens 处断尾。修复：改为 4096 + 新增 `02c_filter_by_length.py` 预过滤（见附录图 A3、A5）
- **Bug**: `device_map="auto"` 对 1.5B 模型实际只用 1 张卡，多卡申请仅用于 HPC 排队优先级，**主动保留单卡方案**

📸 **截图位置**：
- `作业截图/02-B_gpu_nvidia-smi_4xA16.jpg` — HPC 硬件环境
- `作业截图/02-K_training_completed_summary.jpg` — V2 训练完成总览
- `Enoch/report/loss_curve.png` — V1 训练 loss 曲线
- `Enoch - Version2/作业截图/03-D_training_loss_all_6_steps.jpg` — V2 训练 loss 曲线

## 3. Evaluation（评测，约 680 字）

### 3.1 评测方法论

为避免单一指标偏差，我们设计了**三轮独立交叉验证**：

- **第一轮**：V2 单模型 100 分制 5 题高难度测试（`04_inference_test_v2plus.py`，Best-of-3 采样）
- **第二轮**：V1 vs V2 同题对比（`05_eval_compare_v1_vs_v2_v2plus.py`，5 题 Compare-Hard）
- **第三轮**：V1 vs V2 "V2 主场"对比（`06_showcase_v2_strengths.py`，2 题专挑跨类组合 + 视觉风格）

100 分制加权：**J（JS 深度）30 + I（指令遵循）25 + F（功能可运行）20 + C（CSS 质量）15 + S（结构合规）10**（方法论参考 KRT2002 qwen-python-finetuning + Design2Code NAACL 2025, Si et al., 2025）。

### 3.2 三轮结果汇总

**第二轮（通用型对比）**：

| 案例 | V1 | V2 | Δ |
|---|---|---|---|
| C1 分段秒表（DEPTH） | 92 | 91 | -1 |
| C2 游泳防水秒表（BREADTH） | 80 | 56 | **-24** |
| C3 括号优先级计算器（REASONING） | 95 | 83 | -12 |
| C4 待办+番茄钟（COMBINATION） | 96 | 93 | -3 |
| C5 石头剪刀布+计分（CROSS） | 82 | 83 | +1 |
| **均值** | **89.0** | **81.2** | **-7.8** |

**第三轮（V2 强项主场）** AI 评分 V2 胜 2/2（S1: 93 vs 80, S2: 91 vs 86），但——

### 3.3 决定性环节：人工评判推翻 AI 评分

逐题打开 HTML 在 Chrome 渲染：

| 题目 | AI 评分 | 人工评判 |
|---|---|---|
| S1 晨间三合一 | V2 (93) > V1 (80) | **V1 完胜**——V1 功能完整；V2 的"5 分钟冥想计时"打开后显示 **4.6666666 分钟吸气** 这种浮点循环错乱 |
| S2 暗色霓虹井字棋 | V2 (91) > V1 (86) | **V1 完胜**——V1 秒表功能可用；V2 主题色更"霓虹"但核心按钮功能错位，无法完成井字棋交互 |

📸 **截图位置**：
- `Enoch - Version2/作业截图/S1_v1_morning_routine.jpg` 与 `S1_v2_morning_routine.jpg`（V1/V2 同题并排）
- `Enoch - Version2/作业截图/S2_v1_tictactoe.jpg` 与 `S2_v2_tictactoe.jpg`
- `Enoch - Version2/作业截图/04-K_compare_summary_table.jpg` — 100 分制汇总表

### 3.4 V1 两个强项（肯定）

- **格式稳定**: V1 训练数据虽"伪"，但模板固定，输出 `</html>` 闭合率 ~95%
- **简单任务可靠**: 秒表/计时/BMI 计算类 V1 几乎 100% 可用

### 3.5 V1 两个弱项（批判）

- **泛化极弱**: 超出种子覆盖的 90 类工具后，V1 只会"换皮肤"不会"换逻辑"
- **validation loss 被数据泄漏污染**: V1 的 `eval_loss=0.1345` 是因为同一份 HTML 的 10 个指令变体被随机切进了训练/验证集（§V1 README §2.2 问题 4）

### 3.6 V2 两个强项（肯定）

- **数据信息熵显著提升**: 一指令×4风格 + Evol-Instruct 四方向，真实样本量从 V1 的 ~50 条独立样本升到 ~830 条
- **训练工程化程度高**: NEFTune + CompletionOnlyLoss + 类别隔离验证集 + Best-of-3 采样 + adapter_config 身份自证，流程可复现度远高于 V1

### 3.7 V2 两个弱项（批判）

- **人工评判整体败于 V1**: 第二轮 4/5 输 + 第三轮 AI 赢但人工全输
- **SFT 阶段贪多求全**: 1.5B 模型被迫同时学"多视觉风格 + 跨类组合 + 复杂推理"，触发 Chinchilla (Hoffmann et al., 2022) 所警告的小模型多目标过载

### 3.8 评测方法论本身的局限

5 维静态启发式评分**天然奖励"看起来复杂"的输出**——CSS 更长、JS 变量更多即加分，但和"功能真的能跑"无关。这正是 Goodhart's Law 在 ML 评测中的经典显现：**当指标成为优化目标时，它就失去了衡量意义**。

## 4. Recommendations（改进建议，约 320 字）

### 4.1 改进 1：SFT 阶段应收敛到"精准路线"（对应 §3.3、§3.7）

**证据**：V2 输出"4.666666 分钟吸气"这种浮点错乱，在 LIMA 假设下（Zhou et al., 2023）是典型的"SFT 数据过散导致基座能力被噪声淹没"。**建议**：下一轮（V3）SFT 仅保留一指令×2风格（不再 4 风格），Evol-Instruct 比例降到 30% 以下，总样本量控制在 500 条以内，对齐 LIMA 的"1000 条高质量阈值"在 1.5B 模型上的等比缩放。

### 4.2 改进 2：增加 DPO 阶段修复泛化能力（对应 §3.3、§3.6）

**证据**：Qwen2.5-Coder 官方本身就是 SFT → DPO 两阶段（Hui et al., 2024），V2 试图在 SFT 阶段同时做"精准 + 泛化"违背了模型厂商的既定工艺。**建议**：利用 V1-win / V2-lose 的对比数据天然形成**偏好对**（V1 可用 → chosen，V2 炫但错 → rejected），用 DPO（Rafailov et al., 2023）训一轮，成本约 1 张 A16×2 小时，直接修复 V2 退化。

### 4.3 改进 3：评测框架升级为浏览器级功能测试（对应 §3.8）

**证据**：现有 5 维静态启发式评分和 V2 的表面模式过拟合是同源的——互相自证。**建议**：引入 **Playwright 自动化**模拟真实用户点击，验证"点开始 → 计时器真的跳秒、点停止 → 计时器真的停"这种行为级功能；同时用 **GPT-4V 打分渲染后的截图**（Design2Code, Si et al., 2025 的做法）做视觉正确性交叉验证。

## 5. Reflection（反思，约 170 字）

本项目最直接对应两条 CILO：

- **CILO "Evaluate and critically assess the capabilities and limitations of deployed LLMs"**：三轮评测 + 人工评判 + Goodhart's Law 分析，构成了对"AI 自动评分"本身的批判性评估——数字高不等于真的好。
- **CILO "Apply LLM fine-tuning techniques including LoRA/QLoRA to domain-specific tasks"**：V1/V2 两轮 LoRA 系统性对比，暴露了"target_modules 选择、rank 与数据量匹配、max_seq_length 与数据特性联动"三组超参数的工程红线。

**未来应用**：该"SFT 精准 + DPO 泛化"框架可直接迁移到教育场景的**学生作业自动生成器**（输入"做一份小学四年级应用题练习"→ 输出可打印 PDF）、**企业内部表单工具生成器**等中文移动端低代码场景。V2 的"用昂贵 HPC 时间换一条工程红线实证"的负结果，对同参数量（1-3B）的中文 Coder 微调项目都有直接警示价值。

---

## References（APA 7, English）

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., … Zaremba, W. (2021). *Evaluating large language models trained on code* (arXiv:2107.03374). arXiv. https://doi.org/10.48550/arXiv.2107.03374

Databricks. (2025). *Efficient fine-tuning with LoRA: A guide to optimal parameter selection for large language models*. Databricks. https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms

Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical neural story generation. In I. Gurevych & Y. Miyao (Eds.), *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 889–898). Association for Computational Linguistics.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., … Sifre, L. (2022). *Training compute-optimal large language models* (arXiv:2203.15556). arXiv. https://doi.org/10.48550/arXiv.2203.15556

Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. In *Proceedings of the International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=rygGQyrFvH

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *Proceedings of the International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=nZeVKeeFYf9

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, B., Yu, B., Lu, K., Dang, K., Che, B., He, B., Chen, G., Lin, R., & Ren, W. (2024). *Qwen2.5-Coder technical report* (arXiv:2409.12186). arXiv. https://doi.org/10.48550/arXiv.2409.12186

HuggingFace. (2025). *Generation strategies documentation*. HuggingFace. https://huggingface.co/docs/transformers/generation_strategies

Jain, N., Chiang, P.-H., Wen, Y., Kirchenbauer, J., Chu, H.-M., Somepalli, G., Bartoldson, B. R., Kailkhura, B., Schwarzschild, A., Bhatele, A., Geiping, J., Huang, F., & Goldstein, T. (2024). NEFTune: Noisy embeddings improve instruction finetuning. In *Proceedings of the International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=0bMmZ3fkCk

KRT2002. (2024). *qwen-python-finetuning: 14-indicator evaluation framework for Python code generation* [Software]. GitHub. https://github.com/KRT2002/qwen-python-finetuning

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., Hubert, T., Choy, P., de Masson d'Autume, C., Babuschkin, I., Chen, X., Huang, P.-S., Welbl, J., Gowal, S., Cherepanov, A., … Vinyals, O. (2022). Competition-level code generation with AlphaCode. *Science*, *378*(6624), 1092–1097. https://doi.org/10.1126/science.abq1158

Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2023). *An empirical study of catastrophic forgetting in large language models during continual fine-tuning* (arXiv:2308.08747). arXiv. https://doi.org/10.48550/arXiv.2308.08747

Luo, Z., Xu, C., Zhao, P., Sun, Q., Geng, X., Hu, W., Tao, C., Ma, J., Lin, Q., & Jiang, D. (2024). WizardCoder: Empowering code large language models with Evol-Instruct. In *Proceedings of the International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=UnUwSIgK5W

Muxup. (2025). *Vendor-recommended LLM parameter quick reference*. Muxup. https://muxup.com/2025q2/recommended-llm-parameter-quick-reference

OpenAI. (2021). *human-eval: Sample + greedy-fallback evaluation framework for code generation* [Software]. GitHub. https://github.com/openai/human-eval

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh (Eds.), *Advances in Neural Information Processing Systems* (Vol. 35, pp. 27730–27744). Curran Associates.

Qwen Team. (2024). *Qwen2.5 technical blog and model card*. Qwen. https://qwenlm.github.io/blog/qwen2.5/

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, & S. Levine (Eds.), *Advances in Neural Information Processing Systems* (Vol. 36, pp. 53728–53741). Curran Associates.

Si, C., Yang, Y., & Hashimoto, T. (2025). Design2Code: Benchmarking multimodal code generation for visual design. In *Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*. https://github.com/NoviScl/Design2Code

Unsloth. (2026). *LoRA fine-tuning hyperparameters guide*. Unsloth. https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., & Weston, J. (2020). Neural text generation with unlikelihood training. In *Proceedings of the International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=SJeYe0NtvH

Zhang, Y. (2025). *Breaking memorization barriers in LLM code fine-tuning via information bottleneck for improved generalization* (arXiv:2510.16022). arXiv. https://doi.org/10.48550/arXiv.2510.16022

Zhang, Z. (2025). *Memorize or generalize? Evaluating LLM code generation with evolved questions* (arXiv:2503.02296). arXiv. https://doi.org/10.48550/arXiv.2503.02296

Zhou, C., Liu, P., Xu, X., Iyer, S., Du, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., & Levy, O. (2023). LIMA: Less is more for alignment. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, & S. Levine (Eds.), *Advances in Neural Information Processing Systems* (Vol. 36, pp. 55006–55021). Curran Associates.

---

## Appendix A: Screenshots Index（截图索引，占位待补）

| 编号 | 文件名 | 位置 | 用途 |
|---|---|---|---|
| A1 | `02-A1_training_head60_config.jpg` | `Enoch - Version2/作业截图/` | V2 训练超参配置 |
| A2 | `02-B_gpu_nvidia-smi_4xA16.jpg` | `Enoch - Version2/作业截图/` | HPC 硬件环境 |
| A3 | `03-D_training_loss_all_6_steps.jpg` | `Enoch - Version2/作业截图/` | V2 train_loss 曲线 |
| A4 | `03-E_eval_loss_0.1094_result.jpg` | `Enoch - Version2/作业截图/` | V2 eval_loss |
| A5 | `loss_curve.png` | `Enoch/report/` | V1 train/val loss 曲线 |
| A6 | `S1_v1_morning_routine.jpg` + `S1_v2_morning_routine.jpg` | `Enoch - Version2/作业截图/` | 人工评判 S1 并排 |
| A7 | `S2_v1_tictactoe.jpg` + `S2_v2_tictactoe.jpg` | `Enoch - Version2/作业截图/` | 人工评判 S2 并排 |
| A8 | `04-K_compare_summary_table.jpg` | `Enoch - Version2/作业截图/` | 100 分制汇总 |
| A9 | `04-B_v1v2_adapter_identity_proof.jpg` | `Enoch - Version2/作业截图/` | V1/V2 身份自证 |

> 正式提交时把以上截图按编号插入正文对应 📸 标记处，并加图注（中文，格式: "图 A3. V2 训练 loss 曲线（来源: HPC SLURM job 1442, 2026-05）"）。

## Appendix B: V1 vs V2 configuration diff（关键超参对比）

见 `Enoch - Version2/README.md §7 超参数配置一览` 与 `Enoch/README.md` 对应章节（两份 README 提交时作为附件一并附上）。

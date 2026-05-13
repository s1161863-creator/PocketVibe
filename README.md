# PocketVibe
Two-round LoRA fine-tuning of Qwen2.5-Coder-1.5B for Chinese-to-mobile-HTML code generation (INT6138 Project II, EdUHK).
#进去看分别的V1和V2就可以，读取各自文件夹的readme,所有的数据和修改的log都总结在那上面
#！！！！！！！！重点看V2
V3我已经有了完整的思路，我和Chenbo做两版完全不同的思路，但由于时间问题已经不属于作业的范畴，不过做好实验找到创新点，可以作为博士论文先发
V3的具体思路:
Chenbo:你去完成我论文结尾提到的V3的全部技术方案。（另外V2的推理脚本里的指令我人工查了，有些写的有问题，要手写），我们的卡能带动一个轻量的DPO算法去强化学习，配 Trainer 的所有超参（学习率、epoch、batch size、gradient_accumulation）你来按我论文里建议的方向定或者问问Enoch
FangZheng:我去做一个完全不训练不调参，只推理的版本：目前确定的是做TTS试试，并行比如：同一个 prompt 生成 N 个候选，用 verifier 选最好的，还有串行比如让模型一步步精修自己的输出，这两个混着来，然后在前端浏览器里，我们模拟真实用户query:
for candidate in candidates:
    # 用 Playwright 在 headless Chrome 里渲染
    page.goto(f"file://{candidate}.html")
    
    # 收集"执行反馈"
    feedback = {
        "console_errors": page.console_errors(),
        "has_h1": page.locator("h1").count() > 0,
        "button_clickable": test_click("button#start"),
        "timer_ticks": test_timer_behavior(),  # 点击后秒表数字真的在跳吗？
        "viewport_ok": page.viewport_size == mobile_size,
    }
    
    if feedback.has_errors:
        # 把错误回灌给模型让它修
        refined = model.generate(
            prompt=f"{original_prompt}\n\n之前的输出:\n{candidate}\n\n问题:\n{feedback}\n\n请修复"
        )

但是不调参的意义在于，我只在推理里激发了Qwen1.5B原生能力让它output适应手机端HTML。

我们可以把两轮V3，你强化学习，我TTS，的结果放在一起对比，然后做一个V4，V4可以用TTRL(推理过程当中调参），但目前问题是小模型做TTRL容易崩，这个到时候再研究

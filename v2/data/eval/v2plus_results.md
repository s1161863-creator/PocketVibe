# PocketVibe V2+ — V2 单模型细粒度推理结果

**基座**: Qwen/Qwen2.5-Coder-1.5B-Instruct  |  **LoRA**: qlora-v2-run1

**推理**: Best-of-3, temp=0.7, top_p=0.8, top_k=20

**平均分**: 88.2/100

| 用例 | 类别 | 总分 | J(30) | I(25) | F(20) | C(15) | S(10) | 长度 | I命中率 |
|------|------|------|-------|-------|-------|-------|-------|------|---------|
| C1_depth_stopwatch_lap | DEPTH | **98** | 30 | 25 | 20 | 13 | 10 | 9876 | 8/8 |
| C2_breadth_swim_timer | BREADTH | **83** | 18 | 21 | 20 | 14 | 10 | 4656 | 6/7 |
| C3_reasoning_calc_paren | REASONING | **82** | 17 | 22 | 20 | 13 | 10 | 4928 | 7/8 |
| C4_combination_todo_pomodoro | COMBINATION | **93** | 24 | 25 | 20 | 14 | 10 | 6773 | 9/9 |
| C5_cross_rps_scoreboard | CROSS_CATEGORY | **85** | 16 | 25 | 20 | 14 | 10 | 6976 | 9/9 |

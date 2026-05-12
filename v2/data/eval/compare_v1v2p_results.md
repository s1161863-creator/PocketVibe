# PocketVibe — V1 vs V2 Fine-grained Comparison (100-pt)

- **Base**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **V1**: `outputs/qlora-run1/final_adapter` (original / aligned with local Enoch repo)
- **V2**: `outputs/qlora-v2-run1/final_adapter` (Version2 new training, SLURM 1442)
- **Inference**: Best-of-3, temp=0.7, top_p=0.8, top_k=20, max_new_tokens=4096
- **Note**: Job 1448/1455 was interrupted by a runtime bitsandbytes CUDA fault after generating V2 for cases C1/C2/C3. C4_v2 / C5_v2 are marked `--` (MISSING). Averages and win counts are computed over **valid cases only**.

**Avg (n=5 valid cases)**: V1 = 89.0/100  |  V2 = 81.2/100  |  delta = -7.8

**Wins (valid only)**: V2=1 | tie=0 | V1=4

## Total score comparison

| Case | Category | V1 | V2 | delta |
|------|----------|----|----|----|
| C1_depth_stopwatch_lap | DEPTH | 92/100 | **91/100** | -1 |
| C2_breadth_swim_timer | BREADTH | 80/100 | **56/100** | -24 |
| C3_reasoning_calc_paren | REASONING | 95/100 | **83/100** | -12 |
| C4_combination_todo_pomodoro | COMBINATION | 96/100 | **93/100** | -3 |
| C5_cross_rps_scoreboard | CROSS_CATEGORY | 82/100 | **83/100** | +1 |

## Dimension breakdown (V1 -> V2)

| Case | J(30) | I(25) | F(20) | C(15) | S(10) | length | I hit-rate |
|------|-------|-------|-------|-------|-------|--------|------------|
| C1_depth_stopwatch_lap | 26->22 | 22->25 | 20->20 | 14->14 | 10->10 | 14155->9082 | 7/8 -> 8/8 |
| C2_breadth_swim_timer | 15->0 | 21->25 | 20->7 | 14->14 | 10->10 | 5743->13141 | 6/7 -> 7/7 |
| C3_reasoning_calc_paren | 26->14 | 25->25 | 20->20 | 14->14 | 10->10 | 13255->4367 | 8/8 -> 8/8 |
| C4_combination_todo_pomodoro | 27->24 | 25->25 | 20->20 | 14->14 | 10->10 | 8164->8665 | 9/9 -> 9/9 |
| C5_cross_rps_scoreboard | 16->14 | 22->25 | 20->20 | 14->14 | 10->10 | 7588->8416 | 8/9 -> 9/9 |

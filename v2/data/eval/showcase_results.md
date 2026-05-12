# V2 强项展示对比（Showcase）

> 两条题目均针对 V2 训练目标（跨类组合 + 长HTML + 视觉风格 + 复杂状态）设计。

## 分数对比（100 分制）

| 题号 | 场景 | V1 总分 | V2 总分 | Δ (V2-V1) |
|------|------|---------|---------|-----------|
| S1_morning_routine | 晨间例行三合一工具 | 95 | 87 | **-8** |
| S2_tictactoe_neon | 暗色霓虹风井字棋 + 计分板 | 95 | 93 | **-2** |

## 维度细分 J/I/F/C/S

| 题号 | 模型 | J(30) | I(25) | F(20) | C(15) | S(10) | 总分 |
|------|------|-------|-------|-------|-------|-------|------|
| S1_morning_routine | **V1** | 28 | 23 | 20 | 14 | 10 | **95** |
| S2_tictactoe_neon | **V1** | 26 | 25 | 20 | 14 | 10 | **95** |
| S1_morning_routine | **V2** | 20 | 23 | 20 | 14 | 10 | **87** |
| S2_tictactoe_neon | **V2** | 26 | 25 | 20 | 12 | 10 | **93** |

## HTML 长度（反映输出完整度）

| 题号 | V1 字符数 | V2 字符数 | V2/V1 |
|------|-----------|-----------|-------|
| S1_morning_routine | 12375 | 6222 | 0.50x |
| S2_tictactoe_neon | 11608 | 6584 | 0.57x |

## 产物清单（data/eval/ 下）

- `showcase_S1_morning_routine_V1.html` — V1 / S1_morning_routine / 95分
- `showcase_S2_tictactoe_neon_V1.html` — V1 / S2_tictactoe_neon / 95分
- `showcase_S1_morning_routine_V2.html` — V2 / S1_morning_routine / 87分
- `showcase_S2_tictactoe_neon_V2.html` — V2 / S2_tictactoe_neon / 93分
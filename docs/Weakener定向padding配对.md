# Weakener 定向 padding 三臂配对结果

三臂在相同 Full New5 事件序列上逐 flow 匹配实际真实包 padding 字节。
Random-all 使用全部合法真实上行事件。Random-candidate 只使用 warmup 候选桶并随机排序。
Targeted-candidate 使用同一候选集合并按稳定辨识分排序。

## unsw8

| 视图 | 攻击 | Random-all | Random-candidate | Targeted-candidate | T−RA | T−RC |
|---|---|---:|---:|---:|---:|---:|
| C | LR cumulative | 0.9509 | 0.9496 | 0.9536 | +0.0027 | +0.0040 |
| C | LR sliding | 0.8883 | 0.8918 | 0.8917 | +0.0033 | -0.0001 |
| C | RF cumulative | 0.9690 | 0.9686 | 0.9751 | +0.0061 | +0.0065 |
| C | RF sliding | 0.8537 | 0.8481 | 0.8548 | +0.0011 | +0.0067 |
| D | LR cumulative | 0.7683 | 0.7760 | 0.7800 | +0.0117 | +0.0040 |
| D | LR sliding | 0.7395 | 0.7420 | 0.7455 | +0.0060 | +0.0035 |
| D | RF cumulative | 0.8899 | 0.8928 | 0.8972 | +0.0073 | +0.0044 |
| D | RF sliding | 0.7529 | 0.7558 | 0.7574 | +0.0045 | +0.0016 |
| real_uplink | LR cumulative | 0.7504 | 0.6870 | 0.7366 | -0.0138 | +0.0496 |
| real_uplink | LR sliding | 0.7585 | 0.7274 | 0.7855 | +0.0270 | +0.0581 |
| real_uplink | RF cumulative | 0.9062 | 0.9136 | 0.9466 | +0.0404 | +0.0329 |
| real_uplink | RF sliding | 0.7824 | 0.7902 | 0.8014 | +0.0190 | +0.0112 |

## deakin10

| 视图 | 攻击 | Random-all | Random-candidate | Targeted-candidate | T−RA | T−RC |
|---|---|---:|---:|---:|---:|---:|
| C | LR cumulative | 0.4504 | 0.4520 | 0.4556 | +0.0052 | +0.0036 |
| C | LR sliding | 0.4554 | 0.4545 | 0.4661 | +0.0106 | +0.0115 |
| C | RF cumulative | 0.5734 | 0.5719 | 0.5974 | +0.0241 | +0.0255 |
| C | RF sliding | 0.5115 | 0.5103 | 0.5224 | +0.0109 | +0.0121 |
| D | LR cumulative | 0.3208 | 0.3218 | 0.3217 | +0.0009 | -0.0001 |
| D | LR sliding | 0.4088 | 0.4090 | 0.4091 | +0.0003 | +0.0001 |
| D | RF cumulative | 0.5861 | 0.5827 | 0.5973 | +0.0112 | +0.0146 |
| D | RF sliding | 0.5129 | 0.5133 | 0.5187 | +0.0057 | +0.0054 |
| real_uplink | LR cumulative | 0.3897 | 0.3919 | 0.3873 | -0.0023 | -0.0046 |
| real_uplink | LR sliding | 0.4466 | 0.4456 | 0.4486 | +0.0021 | +0.0030 |
| real_uplink | RF cumulative | 0.5345 | 0.5352 | 0.5330 | -0.0014 | -0.0021 |
| real_uplink | RF sliding | 0.5284 | 0.5316 | 0.5359 | +0.0075 | +0.0043 |

迁移张量、设备对 JS、在线混淆矩阵和逐设备 recall 见 `docs/Weakener定向padding机制诊断.md`。
原始逐日结果与逐流审计见 `artifacts/es/weakener_padding_{family}_s{seed}.{json,npz}`。

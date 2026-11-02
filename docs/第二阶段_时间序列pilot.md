# Rotation 时间序列 pilot：Fixed / Periodic / Per-flow

## 1. 协议

- 数据：UNSW，按日期排序，每 5 天一个 block，共 34 个 block；
- 攻击者：LR，累计训练（block < t 的全部 defended 数据），from-scratch；
- 当前 block 只用于测试，标签不参与训练；
- 防御：v2 position 配置，W=3；
- 映射：
  - Fixed：所有 block 用同一 mapping；
  - Periodic：每 2 个 block 轮换 mapping，循环；
  - Per-flow：每条 flow 在 mapping 集合中独立随机；
- 指标：BA 曲线、`J(H)=mean BA`、`T_τ`（绝对阈值 0.5/0.8，连续 2 个 block 达标；未达标标记 censored）。

## 2. 结果

| policy | J(H) | T_first_0.5 | T_first_0.8 | T_sustained_0.5 | T_sustained_0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | **0.965** | 0 | 1 | 0 | 1 |
| Periodic | **0.828** | 0 | 2 | 2 | 2 |
| Per-flow | **0.897** | 0 | 1 | 0 | 1 |

注意：之前报告的 Periodic `T_0.5=2` 实际是 **T_sustained**（连续 2 个 block 达标）；首次到达 `T_first_0.5=0`（第一个 block BA 0.783 已 ≥0.5）。两个指标必须分开命名，不能混用。

BA 曲线（前 12 个测试 block）：

| policy | BA_1..BA_12 |
| --- | --- |
| Fixed | 0.783, 1.000, 1.000, 1.000, 0.996, 1.000, 0.998, 0.996, 0.750, 0.870, 0.987, 0.822 |
| Periodic | 0.783, 0.231, 0.923, 0.813, 0.965, 0.826, 0.995, 0.837, 0.744, 0.597, 0.960, 0.803 |
| Per-flow | 0.588, 0.993, 0.861, 0.967, 0.991, 0.997, 0.993, 0.969, 0.746, 0.755, 0.951, 0.799 |

## 3. 解释

1. **Fixed 最高（J=0.965）**：
   - 攻击者可以稳定学习 code/position→类别映射；
   - T_0.8=1，几乎立即达到高 BA。

2. **Periodic 最低（J=0.828）**：
   - 每次切换后 BA 明显下降（如 block2 0.231、block10 0.597）；
   - 随后随累计数据重训练恢复；
   - 说明“定期 rotation”确实造成周期性的适应成本。

3. **Per-flow 居中（J=0.897）**：
   - 每条 flow 随机 mapping 让 code/position 不再是一致信号；
   - 但同时也移除了 train/test 的**系统性映射错位**，攻击者退回到自然流量特征；
   - 因此 BA 没有像 Fixed 那样高，也没有像预期那样接近 chance；
   - 这提示：rotation 的效果依赖“训练/测试之间的系统性错位”，而不只是“随机性”。

4. **T_τ 的局限**：
   - Fixed/Periodic 在 block1-2 就达到 0.8，阈值太容易；
   - 需要相对阈值（clean BA 的 50%/80%）和更长的初始窗口；
   - T_τ 目前是 pilot 级，不足以区分策略。

## 4. 重要 caveat

- 只做了 UNSW、LR、累计、from-scratch；
- 没有 sliding window、warm-start、CIC、相对阈值、置信区间；
- Per-flow 的 mapping 是**每 flow 随机**，但仍然共享同一 code/position 结构与自然流量；
- `T_τ` 用连续 2 block；censored 目前未出现；
- block 边界按 5 天，可能掩盖日内轮换。

## 5. 下一步完整矩阵

| 维度 | 取值 |
| --- | --- |
| policy | Fixed / Periodic(P=1,2,5) / Per-flow |
| training | cumulative / sliding W=2,5,10 |
| init | from-scratch / warm-start |
| attacker | LR / RF / MLP |
| threshold | absolute 0.5/0.8；relative 50%/80% clean BA |
| dataset | UNSW（128 天）；CIC（22 天，block 2 天） |
| metrics | BA_t, J(H), T_τ + censoring, position-only BA, displacement |

实现量：Fixed/Periodic 已有；Per-flow 需要 per-flow code/position（当前用分组 apply 实现，可扩展）；warm-start 用 SGD/MLP warm_start 或增量 fit；预计 1–2 天代码 + 3–5 小时计算。

## 6. 结论（pilot）

- **Fixed 可被学会**；
- **Periodic 造成周期性下降，但可恢复**；
- **Per-flow 不等于最强防御**：随机化 mapping 会让攻击者退回自然流量特征；
- 因此 rotation 的研究重点不是“越随机越好”，而是**训练/测试之间的系统性映射错位 + 可恢复成本**。

## 7. 产物

- 脚本：`scripts/rotation_timeseries.py`
- 结果：`artifacts/es/rotation_timeseries_unsw8.json`
- 分支：`stage2-coordination`

# Position Placebo：类—位置关联是否必要？

## 1. 设计

四臂，全部 W=3、weakener 347 B / 1.05 s、同一批 8k train/test、同一位置池：

| arm | trigger 事件 | 长度 code | 位置 |
| --- | ---: | --- | --- |
| A `placebo_code` | 2 | 所有类相同 | 原类相关位置 |
| B `placebo_position` | 2 | 所有类相同 | **per-flow 随机位置** |
| C `T0_matched` | 0 | 无 | 无 |
| D `real` | 2 | 类特定 | 原类相关位置 |

B 的位置策略：

- 从原配置的**所有类位置对**构成池；
- 对每条 flow 独立抽样，池按 flow 数重复后整体 shuffle；
- 不读取类别标签；
- 因此位置的边际分布、合法范围、两次事件间距分布与原方案一致；
- A/B 只差“位置是否与类别关联”。

## 2. UNSW 结果

Retrained BA：

| arm | LR | RF | MLP |
| --- | ---: | ---: | ---: |
| A placebo_code | 0.263 | 0.405 | 0.376 |
| B placebo_position | **0.933** | **0.999** | **0.976** |
| C T0_matched | 0.985 | 0.999 | 0.991 |
| D real | 0.261 | 0.352 | 0.342 |

配对 contrast（retrained，负 = 左臂 BA 更低/防御更好）：

| contrast | LR [CI] | RF [CI] | MLP [CI] |
| --- | --- | --- | --- |
| A − B | **−0.670 [−0.693,−0.550]** | **−0.594 [−0.609,−0.491]** | **−0.600 [−0.619,−0.489]** |
| B − C | −0.052 [−0.087,−0.032] | 0.000 [−0.001,0.001] | −0.016 [−0.031,−0.002] |
| A − D | 0.002 [−0.018,0.018] | 0.053 [0.026,0.098] | 0.034 [0.016,0.054] |

## 3. CIC 结果

Retrained BA：

| arm | LR | RF | MLP |
| --- | ---: | ---: | ---: |
| A placebo_code | 0.138 | 0.057 | 0.210 |
| B placebo_position | **0.656** | **0.830** | **0.773** |
| C T0_matched | 0.723 | 0.856 | 0.805 |
| D real | 0.138 | 0.051 | 0.204 |

配对 contrast（retrained）：

| contrast | LR [CI] | RF [CI] | MLP [CI] |
| --- | --- | --- | --- |
| A − B | **−0.517 [−0.553,−0.486]** | **−0.773 [−0.813,−0.760]** | **−0.563 [−0.599,−0.537]** |
| B − C | −0.067 [−0.084,−0.055] | −0.026 [−0.032,−0.012] | −0.032 [−0.035,−0.026] |
| A − D | 0.000 [−0.006,0.007] | 0.007 [0.001,0.013] | 0.006 [−0.003,0.021] |

## 4. 位移与窗口控制

A/B/D 的成本与窗口统计完全相同：

| dataset | arm | added B | padding B | dummy B | trigger in window | weakener in window | real in window | dummy fraction | displaced mean/median/p95 | displaced any |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| UNSW | A/B/D | 630 | 41 | 588 | 2 | 3 | 21.21 | 0.20 | 0.20 / 0 / 0 | 5% |
| UNSW | C | 347 | 41 | 305 | 0 | 3 | 21.30 | 0.14 | 0.11 / 0 / 0 | 4% |
| CIC | A/B/D | 630 | 42 | 588 | 2 | 3 | 23.51 | 0.19 | 2.39 / 2 / 5 | 54% |
| CIC | C | 347 | 42 | 305 | 0 | 3 | 24.56 | 0.12 | 1.34 / 0 / 3 | 47% |

A 与 B 的位移分布、trigger/weakener 数量、真实/dummy 比例完全一致：

- UNSW A/B：displaced 0.20、any 5%、dummy fraction 0.20；
- CIC A/B：displaced 2.39、median 2、p95 5、any 54%、dummy fraction 0.19。

因此 A−B 的 BA 差异**不能**用位移或窗口组成解释。

## 5. 结论

1. **类—位置关联是必要的。**
   - 去掉它（B），UNSW retrained 从 0.26–0.41 回到 0.93–1.00；CIC 从 0.06–0.21 回到 0.66–0.83；
   - B ≈ C（T0_matched），说明“只有两个随机位置的事件”几乎不构成防御。

2. **类—位置关联也几乎是充分的。**
   - A 与 D（真实 code）差异 ≤0.05；
   - code/mapping rotation 不是主因，这一点与 Stage A.1 一致。

3. **事件插入/预算本身不够。**
   - B 与 A 有相同的 2 个 trigger、3 个 weakener、630 B、相同位移；
   - 但 B 没有防御效果；
   - 所以“插入事件”必须与**类相关的位置结构**结合。

4. **核心机制应改写为：class-specific trigger placement。**
   - 不是“双包编码”，不是“mapping rotation”，也不是“weakener 剂量”；
   - 是“把两个事件放在类别相关的观察位置”，从而改变自适应攻击者可稳定学习的序列表示。

## 6. 限制

- Position placebo 的位置池来自原配置；边际分布与间距匹配，但 per-flow 随机化后仍可能产生不同的局部上下文；
- 仍只使用 LR 搜索目标；RF/MLP 为事后评估；
- 尚未做“位置强度/位置数量”的剂量实验；
- Held-out 复用，不是盲测。

## 7. 对 Stage B 的影响

Stage B 的原条件是“类特定 code + mapping rotation 有明确贡献”；A.1/A.2/Position placebo 均不支持。

因此：

- **不应优先做 T2_W1**；
- 若继续，应该研究 **position encoding 的强度与结构**（位置数量、位置池、是否与类别映射耦合），而不是增加 trigger pair 数量；
- 研究问题更新为：

> 在固定开销下，类特定的 trigger placement 如何改变自适应攻击者能够稳定学习的序列表示？其收益是否随位置数量/结构饱和或反转？

## 8. 产物

- 脚本：`scripts/position_placebo.py`
- method33 新增可选 `trigger_position_override`（默认行为不变，14 项测试通过）
- 结果：`artifacts/es/position_placebo_unsw8.json`、`position_placebo_cic9.json`
- 分支：`stage2-coordination`

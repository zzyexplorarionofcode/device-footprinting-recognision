# Stage A.1 / A.2：Real vs Placebo Trigger、固定五事件对照

## 1. 目的

Stage A pilot 显示 trigger on/off 是主效应，但 T0/T1 同时改变了预算分配、事件数、类特定 code 与 mapping rotation。A.1/A.2 用于拆开：

- **A.1**：固定 W=3，比较
  - Real Trigger（类特定双包 code + mapping）；
  - Placebo Trigger（同样 2 个事件、同样位置/预算，但所有类共享同一 code，无 mapping rotation）；
  - Padding-matched T0（无 trigger，weakener 347 B/W3，padding 与 T1 相同）。
- **A.2**：固定 5 个 dummy 事件，比较 T1_W3 与 T0_W5。

所有对比使用同一批 8k train/test、同一位置候选、同一分类器与按天配对 bootstrap（300 次）。

## 2. 臂与成本

| arm | trigger | W | weakener cap | total added B | padding B | dummy B | dummy events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real_T1_W3 | 类特定 code | 3 | 347 | 630 | 41 | 588 | 5 |
| Placebo_T1_W3 | 全类同 code | 3 | 347 | 630 | 41 | 588 | 5 |
| T0_matched_W3 | 无 | 3 | 347 | 347 | 41 | 305 | 3 |
| T0_full_W3 | 无 | 3 | 630 | 623 | 318 | 305 | 3 |
| T0_W5 | 无 | 5 | 630 | 628 | 121 | 508 | 5 |
| T1_W3 | 类特定 code | 3 | 347 | 630 | 41 | 588 | 5 |

Placebo 保留真实 trigger 的位置（验证器要求位置唯一），只移除**类特定长度 code 与 mapping rotation**。因此 Real−Placebo 直接隔离 code/rotation；Placebo−T0 仍包含“两个事件 + 位置 + 预算”。

## 3. A.1 结果：Real ≈ Placebo

Retrained BA：

| dataset | arm | LR | RF | MLP |
| --- | --- | ---: | ---: | ---: |
| UNSW | Real | 0.261 | 0.352 | 0.342 |
| UNSW | Placebo | 0.263 | 0.405 | 0.376 |
| UNSW | T0_matched | 0.985 | 0.999 | 0.991 |
| CIC | Real | 0.138 | 0.051 | 0.204 |
| CIC | Placebo | 0.138 | 0.057 | 0.210 |
| CIC | T0_matched | 0.723 | 0.856 | 0.805 |

配对 contrast（retrained，负 = 左臂 BA 更低/防御更好）：

| contrast | dataset | LR [CI] | RF [CI] | MLP [CI] |
| --- | --- | --- | --- | --- |
| Real − Placebo | UNSW | −0.002 [−0.018,0.018] | **−0.053 [−0.098,−0.026]** | **−0.034 [−0.054,−0.016]** |
| Real − Placebo | CIC | 0.000 [−0.007,0.006] | **−0.007 [−0.013,−0.001]** | −0.006 [−0.021,0.003] |
| Placebo − T0_matched | UNSW | **−0.722 [−0.745,−0.603]** | **−0.594 [−0.609,−0.492]** | **−0.615 [−0.638,−0.502]** |
| Placebo − T0_matched | CIC | **−0.585 [−0.626,−0.551]** | **−0.799 [−0.826,−0.784]** | **−0.595 [−0.627,−0.571]** |

**判读：**

- Real vs Placebo 的差异最多约 0.05（UNSW RF），CIC ≤0.007；**类特定长度 code + mapping rotation 不是主因**。
- Placebo vs T0 的差异 0.59–0.80；**两个事件 + 位置 + 预算分配才是主效应**。
- Placebo 仍保留类特定 trigger 位置，所以“位置编码”尚未被移除；这是下一步需要单独控制的层。

## 4. Padding 的作用

T0_matched（padding 41 B）vs T0_full（padding 318 B），同样 W=3、无 trigger：

| dataset | model | T0_matched retr | T0_full retr | Δ |
| --- | --- | ---: | ---: | ---: |
| UNSW | LR | 0.985 | 0.971 | −0.014 |
| UNSW | RF | 0.999 | 0.999 | 0.000 |
| UNSW | MLP | 0.991 | 0.991 | 0.000 |
| CIC | LR | 0.723 | 0.715 | −0.008 |
| CIC | RF | 0.856 | 0.856 | 0.000 |
| CIC | MLP | 0.805 | 0.800 | −0.005 |

Padding 对 retrained aligned BA 的影响很小；它主要影响 clean→defended（zero-shot）和长度边际。

## 5. A.2：固定五事件，T1_W3 vs T0_W5

| dataset | model | T1_W3 retr | T0_W5 retr | Δ [CI] |
| --- | --- | ---: | ---: | --- |
| UNSW | LR | 0.261 | 0.967 | **−0.706 [−0.737,−0.589]** |
| UNSW | RF | 0.352 | 0.999 | **−0.647 [−0.677,−0.571]** |
| UNSW | MLP | 0.342 | 0.995 | **−0.653 [−0.678,−0.541]** |
| CIC | LR | 0.138 | 0.722 | **−0.584 [−0.624,−0.557]** |
| CIC | RF | 0.051 | 0.851 | **−0.800 [−0.841,−0.779]** |
| CIC | MLP | 0.204 | 0.795 | **−0.591 [−0.621,−0.573]** |

同样 5 个事件、同样 ~630 B：

- T1_W3 用 **2 个 trigger 事件 + 3 个 weakener**，dummy 588 B、padding 41 B；
- T0_W5 用 **5 个 weakener**，dummy 508 B、padding 121 B；
- T1_W3 的 retrained BA 低 0.58–0.80。

结合 A.1：这不是 code/rotation 的贡献，而是 **两个事件的插入位置/结构 + 预算从 weakener 转向 trigger 层**。

## 6. 结论

1. **Mapping rotation / 类特定长度 code 不是主要防御机制。**
   - Real vs Placebo 差异 ≤0.05，CIC ≤0.007；
   - 这直接削弱了“诱导包编码是核心”的假设。

2. **Trigger 层的效果主要来自“插入两个事件 + 位置 + 资源分配”。**
   - Placebo（无类 code）仍然把 retrained BA 从 0.99/0.86 压到 0.38/0.21；
   - 位置仍是类特定的，因此不能把全部效应归因于“事件数量”本身；下一步应做位置 placebo。

3. **在固定五事件下，牺牲 2 个 weakener 换 1 个 trigger pair 是值得的（aligned BA）。**
   - T1_W3 vs T0_W5：Δ −0.58~−0.80；
   - 但收益不是来自 code，而是 trigger 事件的位置/结构。

4. **Stage B 的启动条件未满足。**
   - 你设定的条件是“A.1 证明类特定 Trigger code + mapping rotation 有明确贡献”；
   - 结果相反：贡献很小；
   - 因此**不应**现在重构 T2_W1 的多 trigger pair。

## 7. 下一步（不扩 Stage B）

要真正分离“事件/位置”与“code”，只需一个补充控制：

- **Position placebo**：保持 2 个 trigger 事件与预算，但将 trigger 位置在类间随机化/平衡（理想是 per-flow 随机位置），使位置不携带类信息；
- 若 Position placebo 仍显著优于 T0，则“插入事件本身/预算”是主因；
- 若 Position placebo 退回 T0 附近，则“类特定位置编码”是主因。

这比增加 trigger pair 数量更直接回答当前问题。

## 8. 产物

- 脚本：`scripts/stage2_contrasts.py`
- 结果：`artifacts/es/stage2_contrasts_unsw8.json`、`stage2_contrasts_cic9.json`
- 分支：`stage2-coordination`

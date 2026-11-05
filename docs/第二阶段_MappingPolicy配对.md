# UNSW 三种 Mapping Policy 配对协议

## 样本、状态与权限

- UNSW8 预处理数据，FADE UNSW8 pos v2 配置，32 events，硬上限 630 B / 1.5 s / 6 dummy。执行器逐流验证预算，日志记录 Blind 实际成本均值和最大值。
- 日期按 capture group 排序。前 20 个观测日保留为共享历史参考，从中均匀抽取至多 12,000 条原始流。此后每 20 个观测日为一块，前半块抽至多 1,024 条 Calibration 流，后半块抽至多 1,000 条 Blind 流。最后不足 20 天的块仍按同样比例切分。抽样不读取类别标签。
- 三种策略用同一组流索引、设备标签和映射候选。Fixed 全程一个未知 State，Periodic 每两块共享一个 State，按六状态顺序轮换并从独立随机相位开始。攻击者知道策略和轮换边界，选择器不使用真实相位或状态。Per-flow 对原始行索引预生成独立均匀 State，先于抽样且使用独立 RNG。校准的样本数和顺序均不会改变同一行的 State。
- 共享历史权限包括原始参考流及设备标签、FADE 配置、六个候选映射。用候选 0 合成固定参考流量，再以同一个 trace-free 变换生成训练特征，拟合一次 StandardScaler + LR。候选 0 是攻击者自选的合成参考映射。攻击者获得真实策略流量的设备标签仅限 Calibration，整批标签在当块 Blind 开始前可用。历史 Calibration 标签可保留。Blind 标签仅用于事后 BA，真实 State 仅用于仿真、Oracle 和事后指标。Oracle 与 Adaptive 使用相同的候选重构，Oracle 额外获得逐流真实 State。Oracle BA 仍受 detector、逆变换和设备分类器限制。
- No-canon 直接使用冻结 LR。Fixed 用累计 Calibration 设备对数似然选择一个共享 State。Periodic 仅汇总同一个轮换周期内可用的 Calibration 标签。Per-flow 用累计 Calibration 设备标签优化六候选预测的混合似然，以逐流线性门控输出权重并边缘化。三者均无需真实 State 监督。门控 L2 系数固定为 0.05，未依据 Blind 调参。Per-flow 旧 A–F 的真实 State 监督结果单独归档。
- 训练窗口缺少部分设备类别。所有臂的主 BA 都对同一块 Blind 中实际出现的类别等权，包括历史未见类的零召回。补充 BA 仅对同一块 Blind 中历史已覆盖的类等权。标签预算同时记录当块收到、可用、累积参与适应三种计数。宏平均先对每块算 BA，再对块等权。恢复比为宏平均 BA 的增量之比，Oracle 未高于 No-canon 时留空。它允许负值，也不将 Oracle 当作理论上界。

## 机制证据

**A · 流量构造。** 第一阶段冻结 v2 的机制分解将 dummy 混入、窗口挤占和真实包表示变化分开，dummy 混入解释了主要的 clean→defended 降幅，窗口位移随数据集变化。Stage A 的 padding 对照与固定五事件对照提供补充证据：增加 padding 对 aligned BA 的影响小于以 trigger pair 替代两个 weakener 的组合效应。后者同时改变位置、包长结构和 padding 资源，不能拆成各因素的独立贡献。位置 placebo 和位置稳定性消融补充了类内位置可学习性的证据。上述结果来自各自的实验协议。

**B · State 共享。** 本次配对冻结 FADE 构造，仅更换 State 共享策略及相应的无 State 标签适应方法。策略间的结果是这组策略和攻击协议的联合比较。攻击选择和有限历史类别覆盖会影响差值。已有位置与 Oracle 消融提供机制解释，不能据此将配对 BA 差值全部归因于 State 推断误差。

## 复现

```powershell
python -X utf8 -B -m scripts.paired_mapping_policies --dataset data/processed/unsw_main8_v1/flows_unsw_8class.npz --config configs/methods/fade_unsw8_pos_v2.json --output-log artifacts/es/paired_mapping_policies.json
```

## Results (seed 0)

| Policy | No-canon | Adaptive | Oracle | Gain | Gap | Recovery | Common No / Adaptive / Oracle |
|---|---:|---:|---:|---:|---:|---:|---|
| Fixed | 0.178 | 0.517 | 0.517 | 0.339 | 0.000 | 100.0% | 0.264 / 0.837 / 0.837 |
| Periodic | 0.291 | 0.550 | 0.550 | 0.259 | 0.000 | 100.0% | 0.456 / 0.873 / 0.873 |
| Per-flow | 0.312 | 0.442 | 0.539 | 0.130 | 0.096 | 57.5% | 0.502 / 0.683 / 0.850 |

Common BA excludes blocks with no shared historical and Blind classes.

| Block | Cal / Blind classes | Labels received | Fixed BA | Periodic BA | Per-flow BA | Per-flow State Top-1 / Top-2 |
|---:|---|---:|---|---|---|---|
| 1 | 3 / 3 | 1024 | 0.380 / 0.816 / 0.816 | 0.518 / 0.947 / 0.947 | 0.527 / 0.854 / 0.934 | 0.471 / 0.596 |
| 2 | 4 / 5 | 1024 | 0.309 / 0.601 / 0.601 | 0.359 / 0.636 / 0.636 | 0.509 / 0.630 / 0.718 | 0.395 / 0.472 |
| 3 | 7 / 3 | 1024 | 0.088 / 0.546 / 0.546 | 0.270 / 0.663 / 0.663 | 0.206 / 0.479 / 0.567 | 0.291 / 0.457 |
| 4 | 3 / 4 | 1024 | 0.291 / 0.714 / 0.714 | 0.454 / 0.750 / 0.750 | 0.314 / 0.578 / 0.695 | 0.261 / 0.364 |
| 5 | 4 / 2 | 1024 | 0.000 / 0.500 / 0.500 | 0.167 / 0.333 / 0.333 | 0.333 / 0.167 / 0.333 | 0.260 / 0.368 |
| 6 | 4 / 6 | 1024 | 0.178 / 0.472 / 0.472 | 0.318 / 0.498 / 0.498 | 0.328 / 0.397 / 0.560 | 0.210 / 0.349 |
| 7 | 5 / 5 | 1024 | 0.176 / 0.486 / 0.486 | 0.242 / 0.576 / 0.576 | 0.277 / 0.432 / 0.501 | 0.200 / 0.369 |
| 8 | 3 / 1 | 1024 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.190 / 0.341 |

Per-block BA order: No-canon / Adaptive / Oracle.

Blind flows: 8000. Historical class support: {'1': 31, '2': 1588, '3': 3523, '5': 49, '7': 132}. Maximum cross-policy differences in block mean cost (B, s, dummy count): [0.0, 1.1102230246251565e-16, 0.0].

The audit checked disjoint row IDs, chronological separation, per-row State replay, State-label permissions and all budget ceilings. Fixed and Periodic recovered their Oracle BA in all eight blocks. Per-flow recovered 57.5% of the Oracle gain, with a 0.096 BA gap. These are paired results for one seed and one frozen reference model.

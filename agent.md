# AGENT.md — FADE 设备指纹防御项目交接

> 给后续 agent 的最短路径。仓库根目录：`code打包/`；分支：`stage2-coordination`。

## 1. 一句话

FADE 在固定 32-event 观察窗口内，用**真实包 padding、dummy 插入、位置选择、插入后截断**改变流量表示；当前研究主线已从“mapping rotation 能否破坏身份”转为“攻击者要多少标签、多少时间、多少历史知识才能完成在线适应”。

## 2. 代码怎么看

按这个顺序读，不要从 scripts 开始：

1. `methods/method33.py`
   - `Method33Config`：预算、trigger code/positions、mapping、weakener 配置。
   - `resolve_trigger_positions()`：位置唯一性、1-based、递增。
   - `apply_method33_base()`：真正的 FADE 执行流程。
     - 真实包 padding；
     - weakener dummy 插入；
     - trigger 两包插入（`source_index+1`，即真实包之后）；
     - **插入后截断到 32 events**；
     - stats/audit。
   - `apply_method33()`：jitter 包装；per-flow RNG 由内容 token 决定，chunk/order 不变。
2. `methods/method33_tools.py`
   - 配置加载/校验、CLI、预算检查、mapping 是 permutation/derangement。
3. `configs/methods/*.json`
   - `fade_unsw8_pos_v2.json`、`fade_cic9_pos_v2.json` 是当前主配置。
4. `scripts/`
   - 每个实验一个 runner；结果写 `artifacts/es/*.json`。
5. `docs/`
   - 结论与设计；优先看 `第一阶段总结_v2.md`、`第二阶段_FeatureCanonical.md`。

### 关键不变量

- 窗口固定 32 events；截断发生在所有插入之后。
- 等成本上限：`630 B / 1.5 s / ≤6 dummies`。
- 当前分解：weakener `347 B / 1.05 s / 3` + trigger `283 B / 0.45 s / 2`。
- trigger 恰好 2 包、一对/类；位置由 `trigger_positions[code_index]` 决定。
- `mapping` 是 class→code index 的置换；`next_mapping` 必须是 derangement。
- `max_frame_length=1514`；原始数据可能有 1518，实验脚本会 clip。
- 实验性 per-flow 位置用 `trigger_position_override`，默认行为不变。

### 最容易踩的坑

- `trace`（`complete_event_trace`）只应用于 **oracle/known-mapping** 分析；adaptive 不能读 trace，否则泄漏。
- `features()` 的 gap 是事件 outgoing gap，**第 0 个 gap 强制为 0**；不要用 trace 的 `gap_seconds` 直接重建。
- 短流 trigger 会 fallback 到末尾；重插入时必须在“最后一个 real 之后、weakener 之前”。
- UNSW 按类别排序明显：按流顺序取“前 50%”常是单类；calibration 必须按时间切 block 后再分层。
- UNSW 20-day block 的 calibration pool 也**不覆盖全部 8 类**；class 4 常缺席。
- `n=64` 在 20-day block 常只覆盖 1 类，不能当 selector 失败。

## 3. 实验时间线

| 阶段 | 做了什么 | 主要文档 |
| --- | --- | --- |
| Phase 1 | 等成本位置/长度/gap 搜索 + 消融归因；v2 固定 RNG | `docs/第一阶段总结_v2.md` |
| Phase 1 机制 | CC→CD_complete→CD_visible→CD_merged；dummy mixing / displacement / representation | `docs/第一阶段实验结论.md` |
| Stage A.1/A.2 | Real vs Placebo trigger；padding-matched T0；固定 5 事件 T1W3 vs T0W5 | `docs/第二阶段_StageA1_A2.md` |
| Position placebo | 类位置 vs per-flow 随机位置；A/B/C/D | `docs/第二阶段_PositionPlacebo.md` |
| 失败诊断 | per-class recall、confusion、validation permutation、rotation α | `docs/第二阶段_失败诊断与位置剂量.md` |
| K 稳定性 | K=1/2/4/8 类内位置池；位移分层；position-only 对齐 | `docs/第二阶段_位置稳定性K.md` |
| 时间序列 pilot | Fixed/Periodic/Per-flow；J、T_first/sustained、T_recovery | `docs/第二阶段_时间序列pilot.md` |
| 校准曲线 | permutation / probe / finetune；n=64…1024 | `docs/第二阶段_校准曲线.md` |
| 在线适应 | v2 共享状态模型、位置似然；Adaptive pilot 失败 | `docs/第二阶段_在线适应v2.md` |
| 特征逆变换 | target-level、oracle feature、observed inverse、序列重构 | `docs/第二阶段_FeatureCanonical.md` |
| 跨数据集/模型 | CIC、RF、MLP；multi-phase/seed；selector | `docs/第二阶段_FeatureCanonical.md` |
| 类别适应 | Block6/7 负迁移、Common-class BA、support audit | `docs/第二阶段_FeatureCanonical.md` |

## 4. 已经比较确定的结论

1. **位置/序列表示是 mapping rotation 的主要可逆损失来源。**
   - Oracle canonicalization：Fixed/Periodic/Per-flow 都 ≈ **0.884**。
   - Position-only inverse：约 **0.877**；Combined：**0.884**。
   - Code-only：约 **0.711**，贡献小。

2. **Rotation 不是不可逆信息破坏。**
   - Known mapping + 正确序列重构可回到 Fixed 水平；
   - Target-level 逆映射不够，必须 feature-level。

3. **Learned position mapping 是关键。**
   - K=1 固定类位置最强；K 增大单调削弱；
   - 位移标准化后仍成立；position-only validation-aligned K=1..8 接近 1.0；
   - 类内位置稳定是防御资产。

4. **Detector 是工程变量，不是机制结论。**
   - naive code detector：39/300 错误；
   - joint code+position detector：1/300；
   - 之前的 0.748 主要是 naive detector 产物。

5. **跨数据集/模型总体一致，但 RF 有交互。**
   - CIC+LR：unaware 0.585、position 0.915、combined 0.922、oracle 0.923；
   - UNSW+RF：0.868 / 0.910 / 0.962 / 0.960；
   - CIC+RF：0.694 / 0.970 / 0.991 / 0.990；
   - UNSW+MLP：0.838 / 0.957 / 0.958 / 0.958。
   - RF position-only 恢复率随数据集/phase 变化（UNSW 46–68%，CIC 88–93%），不能写成 RF 固有属性。

6. **在线适应：selector 可行，但受类别覆盖限制。**
   - 2 phase×3 seed：selector **65/66 Top-1**；
   - n=256：18/18；n=1024：47/48；n=64 不可评估；
   - n=256：Probe 0.577 / Finetune 0.882；
   - n=1024：Probe 0.801 / Finetune 0.916；
   - Finetune 平均优于 Probe，但有局部负迁移。

7. **Block6/7 负迁移主要是类别覆盖。**
   - Block6 cal 缺 class 2/5/7，blind 有 253/476/234；
   - Full-class BA 0.916 → Common-class 0.988；
   - Block7 common 0.955–0.963，高于 before 0.887；
   - 不是单纯 Finetune 过拟合。

## 5. 图表入口

### Phase 1 图

![位置/长度/gap 搜索](docs/figures/phase1/fig1_search.png)

![held-out CI](docs/figures/phase1/fig2_heldout_ci.png)

![histogram](docs/figures/phase1/fig3_histogram.png)

![机制分解](docs/figures/phase1/fig4_mechanism.png)

![窗口位移](docs/figures/phase1/fig5_window.png)

![特征视图](docs/figures/phase1/fig6_views.png)

### 表

- `docs/v2_tables.md`：Phase 1 v2 主表。
- `artifacts/es/*.json`：每个实验的原始结果。

## 6. 脚本索引

| 目的 | 脚本 |
| --- | --- |
| 核心 FADE | `methods/method33.py` |
| 配置/校验 | `methods/method33_tools.py` |
| 位置/长度/gap 搜索 | `scripts/codebook_experiments.py` |
| 机制分解 | `scripts/mechanism_decomp.py` |
| 表示审计 | `scripts/representation_audit.py` |
| Stage A.1/A.2 | `scripts/stage2_contrasts.py` |
| Position placebo | `scripts/position_placebo.py` |
| 失败诊断/rotation α | `scripts/classifier_failure.py`、`scripts/position_dose.py` |
| K 稳定性 | `scripts/position_stability.py` |
| 时间序列 | `scripts/rotation_timeseries.py` |
| 校准曲线 | `scripts/calibration_replicate.py` |
| 在线矩阵 | `scripts/online_adaptation.py`、`online_adaptation_v2.py` |
| 特征逆变换 | `scripts/observed_inverse.py`、`state_pairwise_continuous.py`、`adaptive_variants.py` |
| 内存流式 | `scripts/streaming_arms.py` |
| Selector | `scripts/adaptive_selector.py`、`adaptive_selector_v2.py` |
| Support audit | `scripts/blind_support.py` |

## 7. 复现命令（主配置）

```powershell
python -X utf8 -B -m scripts.stage2_contrasts `
  --dataset data/processed/unsw_main8_v1/flows_unsw_8class.npz `
  --config configs/methods/fade_unsw8_pos_v2.json `
  --split-indices artifacts/runs/fade/unsw_8class_main/20261028T090552_310384Z_p_vjmhf_/split_indices.npz `
  --output-log artifacts/es/repro.json `
  --dataset-version unsw8_v1
```

## 8. 当前未完成 / 下一步

1. Adaptive 6 组 variant 队列（Full/Common、early/L2/balanced）汇总；
2. Per-flow 的标签单位与 state selector；
3. CIC/RF/MLP 的 adaptive 复验；
4. 时间序列下 canonicalization attacker vs adaptive attacker；
5. Block coverage 更好的划分（20 天、≥5 类）；
6. 负迁移的正则化/早停最终结论。

> 先读 `docs/第二阶段_FeatureCanonical.md` 的最新章节；它是当前研究主线的总账。

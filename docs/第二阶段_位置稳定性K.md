# 类别内位置稳定性：K=1/2/4/8

## 1. 设计

- Trigger 长度 code 全类相同（placebo code），W=3，630 B cap；
- 每类位置池嵌套：K=1 用原 v2 类位置；K=2/4/8 在同一类池中加入随机合法 pair；
- 每条 flow 在 `mapping` 后的 code index 对应的类池中均匀抽一个 pair；
- train 用 old mapping，test 用 next mapping（保留 rotation）；
- 同一批 8k train/test，LR/RF/MLP，按天配对 bootstrap 300。

K 的混淆：K 增大会改变位置边际分布与间距；本实验记录 displacement、position-only BA 和 pair counts，但不宣称边际完全匹配。

## 2. CIC 结果

| K | position-only | LR [CI] | RF [CI] | MLP [CI] | displaced mean |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 0.000 | 0.138 [0.128,0.151] | 0.057 [0.046,0.063] | 0.210 [0.201,0.223] | 2.39 |
| 2 | 0.000 | 0.245 [0.229,0.258] | 0.368 [0.357,0.377] | 0.315 [0.301,0.325] | 2.39 |
| 4 | 0.000 | 0.363 [0.346,0.379] | 0.592 [0.580,0.620] | 0.470 [0.459,0.483] | 2.39 |
| 8 | 0.000 | **0.528 [0.518,0.543]** | **0.771 [0.755,0.806]** | **0.658 [0.641,0.680]** | 2.39 |

## 3. UNSW 结果

| K | position-only | LR [CI] | RF [CI] | MLP [CI] | displaced mean |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 0.000 | 0.263 [0.232,0.288] | 0.405 [0.366,0.425] | 0.376 [0.339,0.408] | 0.20 |
| 2 | 0.000 | 0.492 [0.404,0.520] | 0.889 [0.757,0.915] | 0.594 [0.497,0.629] | 0.20 |
| 4 | 0.034 | 0.785 [0.648,0.828] | 0.954 [0.824,0.976] | 0.772 [0.639,0.799] | 0.20 |
| 8 | 0.009 | **0.817 [0.672,0.844]** | **0.989 [0.855,0.998]** | **0.898 [0.758,0.924]** | 0.20 |

## 4. 结论

1. **K=1 固定类位置是最强配置**：
   - CIC K=1 RF 0.057；K=8 RF 0.771；
   - UNSW K=1 RF 0.405；K=8 RF 0.989。

2. **K 单调削弱防御**：
   - 两个数据集、三个模型方向一致；
   - 不是位移造成的：CIC displacement 始终 2.39，UNSW 始终 0.20；
   - 也不是标签置换造成的：position-only BA 在 K=1..8 都接近 0（K=4/8 为 0.034/0.009，基本 chance）。

3. **攻击者可以利用类内位置分布，即使 mapping 在 rotation**：
   - K=1 时，类→单一位置在部署时被旋转到另一类位置，精确映射失效；
   - K>1 时，类有一个位置**分布**，rotation 只是把分布整体置换；攻击者仍可学习“分布形状/统计签名”，而不需要恢复精确映射；
   - 因此类内变化越大，攻击者越容易获得稳定的类别统计特征。

4. **回答“关系 vs 类内变化”**：
   - 关系本身（rotation）是必要条件；
   - **类内稳定性是防御资产**：同一类内部位置越固定，rotation 越有效；位置越随机，防御越弱。

## 5. 限制

- K>1 的位置池是随机加入的，边际分布/间距随 K 变化；尚未做边际匹配；
- K=1 用原类位置，K>1 是“原位置 + 随机”，因此 K 同时改变位置分布的熵；
- 只做了 UNSW/CIC；code 固定全同；
- position-only 的 K=4/8 非零但接近 chance，说明仍有少量位置信息；
- 未做类别间 overlap 控制。

## 6. 优先级 2：Rotation 时间稳定性协议

### 策略

| 策略 | 定义 |
| --- | --- |
| Fixed | 全部 block 使用同一 mapping |
| Periodic | 每 P 个 block 轮换一次 mapping，循环 |
| Per-flow | 每条 flow 独立抽 mapping（需要 per-flow code/position 支持） |

### 在线自适应评估

把测试流按时间/日期切成 B 个 block（例如每天一个 block，或每 500 flow 一个 block）：

1. 初始攻击者在 block 0 的 defended 数据上训练；
2. 对 t=1..B：
   - 用当前累计/滑窗 defended 数据重训练；
   - 在 block t 上评估 unaligned BA_t；
   - 不把 block t 标签用于训练；
3. 记录：

\[
T_\tau = \min\{t: BA_t \ge \tau\}
\]

\[
J(H) = \frac{1}{H}\sum_{t=1}^{H} BA_t
\]

4. 同时报告 permutation-aligned BA_t 作为诊断（不用于主结论）；
5. 阈值 τ 建议取 clean BA 的 50%/80%，或固定 0.5/0.8。

### 控制

- 同一 flow 顺序、同一 code/position 策略（建议 K=1 固定类位置 + rotation）；
- Fixed/Periodic/Per-flow 的位置边际尽量匹配；
- 记录 displacement、position-only BA、每 block 的样本量；
- 攻击者重训练方式：from-scratch vs warm-start 分开报告；
- 至少 2 个数据集；UNSW 有 128 天，适合多 block。

### 预期

- Fixed：BA_t 快速升到高位，T_τ 小；
- Periodic：切换后 BA_t 下降，然后随重训练恢复；T_τ 与周期 P 相关；
- Per-flow：若 attacker 无法恢复 flow-level mapping，BA_t 可能持续低；但若位置/长度分布仍可学，J(H) 不会为 0。

## 7. 优先级 3：位置池大小与 overlap

在 K 稳定性之后：

- 固定 K（如 4），改变类别间 pool overlap β=0/0.5/1；
- 固定 overlap，改变 K；
- 控制位置边际/间距与 displacement；
- 指标：BA、position-only BA、逐类 recall、T_τ/J(H)；
- 目标：区分“位置空间规模”与“类别可分性”的贡献。

## 8. 建议顺序

1. 先做 K 的边际匹配复验（确认 K 单调不是分布熵的伪影）；
2. 再做 Fixed/Periodic/Per-flow 在线评估（T_τ/J(H)）；
3. 最后做 overlap/池大小。

K 的结果已经表明：**固定类位置 + rotation 是强防御；类内位置随机化会削弱它。** 这为时间稳定性实验提供了明确的对照。

## 9. 产物

- 脚本：`scripts/position_stability.py`
- 结果：`artifacts/es/position_stability_cic9.json`、`position_stability_unsw8.json`
- 分支：`stage2-coordination`

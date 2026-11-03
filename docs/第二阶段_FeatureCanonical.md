# Target-level 审计与 Feature-level 上界

## 1. Target-level 审计

### Fixed 恒等对照

| 流水线 | J(H) |
| --- | ---: |
| Fixed unaware（v2） | 0.862 |
| Fixed target canonicalization | **0.865** |

在 Fixed 上，target canonicalization 等价于原始标签流水线；差异在噪声范围内。

结论：**映射索引和逆映射没有明显错误。**

### Round-trip 单测

`tests/test_canonicalization.py`：

- `mapping` 是 0..K-1 的排列；
- 对每个 class，`inverse[mapping[class]] == class`；
- old/next 对每个 class 不同。

2 项测试通过。

### Periodic 的异常曲线

target canonicalization 的 Periodic BA：

`0.888, 0.000, 0.871, 0.156, 0.682, 0.294, 0.955, 0.329, ...`

0.0/0.156 不是索引错误（Fixed 对照正常）；更像是：
- 共享模型学的是 code index，而 code target 随 state 旋转；
- 当模型在某 block 上偏向错误 code 区域时，逆映射会整体错位；
- 不是“逆映射公式错”，而是“target 层无法对齐特征”。

## 2. Oracle feature-level canonicalization

为了判断“如果特征层能完全规范化，能否恢复”，做 oracle 上界：

- 对每个 block，不按当前 mapping，而是统一用 reference mapping（old）重新生成 FADE 特征；
- 标签仍为 class；
- 累计训练/盲测协议相同。

| 设置 | J(H) |
| --- | ---: |
| Fixed unaware | 0.862 |
| Periodic target canonicalization | 0.568 |
| Periodic unaware | 0.673 |
| **Periodic oracle feature canonicalization** | **0.880** |

结论：

- oracle 特征规范化可以回到 Fixed 水平；
- 因此 Periodic 的损失**主要是表示层/映射层**，不是源流量信息被不可逆破坏；
- target-level 逆映射不够，必须做特征层变换。

## 3. 重要 caveat

Oracle 上界使用：
- 原始流量；
- reference mapping 重新生成；
- 不是从观测到的 defended traffic 逆变换。

因此它不是可实现的攻击者，只是**上界**。真正的 feature-level inverse 需要：
1. 从 trace/码本识别 observed trigger code index；
2. 用 `M^{-1}` 找到 class/canonical code；
3. 把 trigger code/position 变换到 reference；
4. 重构 32-event 序列（包括真实包位移）。

## 4. 判断

- 不能把 target-level 0.568 解释为“rotation 破坏源信息”；
- 也不能把 oracle 0.880 解释为“攻击者已能恢复”；
- 目前最准确的表述：

> Target-level canonicalization 不足；oracle feature canonicalization 能恢复，说明表示层可逆性有希望；真正的 feature-level inverse 尚未实现。

## 5. 下一步

1. 实现 observed-feature inverse：
   - 用码本识别 trigger code index；
   - 用 `R[M^{-1}[k]]` 得到 canonical code；
   - 在 32-event 序列中替换/移动 trigger 特征；
2. 在 Fixed 上做恒等 inverse benchmark；
3. 合成 round-trip：`T^{-1}(T(x))=x`；
4. 再测 Periodic 跨状态迁移。

## 6. Observed inverse pilot

### 实现

- 从 trace 提取 trigger 的 event_position、code index、real anchor→event_position；
- code-only：替换 trigger 行的 length/gap 为 reference code，位置不变；
- both：再把 trigger 行移动到 reference anchor 对应的 slot。

### 结果

| 设置 | J(H) |
| --- | ---: |
| Periodic unaware（v2） | 0.673 |
| Periodic identity（同脚本） | **0.708** |
| Periodic code-only inverse | **0.700** |
| Periodic code+position inverse | **0.687** |
| Periodic oracle feature | 0.880 |

Fixed identity 对照：**0.883**（baseline v2 Fixed 0.862；差异来自历史 cap 6000 vs 5000，不是精确同参）。

### 判读

1. **Observed inverse 没有超过同脚本 identity**：
   - identity 0.708；
   - code-only 0.700；
   - code+position 0.687；
   - 当前 observed inverse 是负收益。

2. **Fixed identity 接近 baseline，但不精确**：
   - 0.883 vs 0.862；
   - 主要差异是 history cap/随机；
   - 需要同参重跑才能作为严格恒等 benchmark。

3. **Oracle 0.880 与 observed 0.687 的差距**：
   - 不能解释为“位置不可逆”；
   - 更可能是当前位置 slot 推导/序列重构不对；
   - 也可能 code 识别/替换有误。

## 7. 审计结论

- Target-level Fixed 对照通过（0.865 vs 0.862）；
- Synthetic round-trip 只验证了 mapping 的逆，不验证 feature transform；
- Observed inverse **未通过** Fixed identity 严格 benchmark；
- 因此 observed inverse 的 0.687 **不能**写成“rotation 不可逆”的证据。

## 8. 下一步

1. 同参重跑 Fixed identity（cap、模型、splits 完全一致）；
2. 构造 synthetic feature round-trip：给定 x/trace，`T^{-1}(T(x))=x` 精确成立；
3. 修正 position slot 推导（trigger 插入在 source event 之后，anchor+1；需验证 truncation）；
4. 再测 code-only / position-only / combined 的增量。



## 10. Strict identity 与 Oracle detector

### Fixed identity（逐元素）

| locator | flows | max_abs_diff | nonzero rows | detector errors |
| --- | ---: | ---: | ---: | ---: |
| Oracle（真实标签定 code） | 200 | **0.0** | **0** | 0 |
| Observable（长度识别 code） | 200 | 5.43 | 13 | **48/200** |

结论：

- **逆变换数学在 code 已知时是精确的**；
- slot/anchor 公式在 identity 路径上没有错；
- 主要误差来自 code detector（48/200）；
- 修掉“code 不变仍替换 jitter 值”后，observable mismatch 从 200/200 降到 13/200。

### Full Periodic：Oracle code vs observable code

| inverse | J(H) |
| --- | ---: |
| identity | 0.708 |
| observable code-only | 0.700 |
| observable code+position | 0.687 |
| **oracle code + position** | **0.698** |
| oracle feature（regenerate） | 0.880 |

关键：

- 用真实 code 后，inverse 仍然只有 0.698，和 code-only 0.700 基本一样；
- 所以 **code detector 不是主要瓶颈**；
- 当前位置 slot/序列重构不对：简单交换 trigger 行，无法复现 canonical anchor 下真实包位移/窗口截断；
- Oracle regeneration 0.880 与 observed 0.698 的差距，主要来自**序列重构**，不是 code 识别。



## 12. 序列重构审计（进行中）

按 canonical anchor 重插入 trigger、再截断到 32，Fixed identity 结果：

| 设置 | flows | max_abs_diff | nonzero |
| --- | ---: | ---: | ---: |
| identity shortcut | 200 | **0.0** | **0** |
| no shortcut（真实重插入） | 200 | 4.72 | **77** |

说明：

- identity 短路（canonical code == actual code 时逐元素返回）通过；
- 关闭短路后，真实重插入仍有 77/200 mismatch；
- 短流 fallback 已加入，但仍不是唯一差异；
- 事件顺序/weakener 插入/特征行对齐仍未完全对齐。



## 13. Synthetic round-trip（Stage 2/3）

新增 `scripts/event_canonical.py` + `tests/test_event_canonical.py`。

### Stage 2：无截断

- 构造 24 个真实事件 + 2 trigger + 2 weakener（<=32）；
- forward：reference anchors [3,9] → actual [7,15]；
- inverse：actual → reference；
- 断言 `event_id`、`role`、`attrs`、顺序完全一致；
- `lost=[]`。

结果：**通过**。

### Stage 3：有截断

- 32 个真实事件 + trigger，actual anchors [30,31]；
- forward 产生 lost 事件；
- 断言：
  - lost 与 retained 的 event_id 不相交；
  - 如果 trigger 本身被截断，不强行做完整 inverse；
  - 可恢复部分 <=32。

结果：**通过**。

### Fallback 回归

- 只有 5 个真实事件，trigger anchor=10 不存在；
- 断言 trigger 仍在、`fallback_slots=[10]`。

结果：**通过**。

### 边界

以上验证的是 **event-level 参考变换**：
- 事件顺序、属性、截断标记正确；
- 还没有接回真实 FADE 的 64 维特征；
- 真实 trace 的 gap/索引对齐仍待 Stage 4 前修。



## 15. Stage 4/5：正确序列重构后的恢复

### Oracle code

| 设置 | J(H) |
| --- | ---: |
| Periodic unaware | 0.673 |
| Periodic oracle code + reconstruction | **0.884** |
| Fixed unaware | 0.862 |
| Oracle feature regenerate | 0.880 |

### Observable detector

| 设置 | J(H) |
| --- | ---: |
| Periodic observable code + reconstruction | **0.748** |

### 恢复比例

\[
\Delta_{\mathrm{total}} = 0.884 - 0.673 = 0.211
\]

\[
\Delta_{\mathrm{observed}} = 0.748 - 0.673 = 0.075
\]

\[
\text{recovered fraction}
= \frac{0.075}{0.211}
= 0.355
\]

结论：

- 已知 mapping + 正确序列重构可以恢复到 Fixed 水平（0.884）；
- 实际 observed detector 只恢复约 **35.5%**；
- 剩余 ~64.5% 损失来自 code detector（48/200 错误）和不可观测/截断事件；
- 这条分解现在可以正式报告。



## 18. Streaming 四臂与模块增量

### 四臂

| arm | code | position | J(H) |
| --- | --- | --- | ---: |
| 1 | Oracle | Oracle | 0.884 |
| 2 | Weak detector | Oracle | 0.748 |
| 3 | Oracle | Detected | 0.884 |
| 4 | **Joint detector** | **Detected** | **0.882** |

说明：

- 0.748 使用的是 naive `code_index`（从 trace 位置取长度），不是真实可观测上限；
- 联合检测 code/position 后，Arm4 = 0.882，几乎等于 oracle；
- 因此 code detector 的真实损失很小；之前 0.136 主要是 detector 实现差异。

### 模块增量（Oracle code/rows）

| variant | J(H) | 相对 unaware |
| --- | ---: | ---: |
| Periodic unaware | 0.673 | 0 |
| Code-only | 0.711 | +0.038 |
| Position-only | 0.877 | +0.204 |
| Combined | 0.884 | +0.211 |

结论：

- **Position-only 恢复约 97% 的 oracle 增益**；
- Code-only 只增加约 0.038；
- 位置/序列重构是主要可逆成分，code 内容只是小增量；
- Combined 与 Oracle 基本一致。

### 恢复比例（用 Arm4）

\[
\frac{0.882-0.673}{0.884-0.673}=0.991
\]

即联合 detector + 正确重构恢复约 **99%** 的 oracle 增益。  
但这依赖 detector 实现；naive detector 只有 0.748。



| arm | code | position | J(H) |
| --- | --- | --- | ---: |
| 1 | Oracle | Oracle | **0.884** |
| 2 | Detected | Oracle | **0.748** |
| 3 | Oracle | Detected | **0.884** |
| 4 | Detected | Detected | 未完成（内存） |

结论：

- **Arm 1 = Arm 3 = 0.884**：position detector 在本任务上没有损失；
- **Arm 1 − Arm 2 = 0.136**：全部来自 code detector；
- 所以当前恢复差距的主因是 code 识别，不是位置重构；
- Arm 4 和 code-only/position-only 仍需在降低内存后补跑。



## 20. Multi-phase / seed 稳定性

四条件：unaware / position-only / combined / oracle-regenerate。

| 设置 | unaware | position-only | combined | oracle | combined 恢复 | position 恢复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase0 seed0 | 0.706 | 0.877 | 0.884 | 0.883 | 1.01 | 0.97 |
| phase1 seed0 | 0.701 | 0.880 | 0.882 | 0.883 | 1.00 | 0.98 |
| phase0 seed1 | 0.737 | 0.876 | 0.884 | 0.883 | 1.01 | 0.95 |

结论：

- oracle 始终 0.883；
- combined 始终 0.882–0.884；
- position-only 始终 0.876–0.880；
- 恢复比例在不同 phase/seed 下稳定：combined≈100%，position-only≈95–98%；
- 没有观察到某个 mapping 相位特别有利/不利。



## 22. 跨数据集 / 跨模型复验

### CIC + LR（2 天/block，10 个测试 block）

| 条件 | J(H) |
| --- | ---: |
| unaware | 0.585 |
| position-only | 0.915 |
| combined | 0.922 |
| oracle | 0.923 |

恢复：combined ≈ 0.998，position-only ≈ 0.976。

### UNSW + RF

| 条件 | J(H) |
| --- | ---: |
| unaware | 0.868 |
| position-only | 0.910 |
| combined | 0.962 |
| oracle | 0.960 |

恢复：combined ≈ 1.02，position-only ≈ 0.46。

### UNSW + MLP

| 条件 | J(H) |
| --- | ---: |
| unaware | 0.838 |
| position-only | 0.957 |
| combined | 0.958 |
| oracle | 0.958 |

恢复：combined ≈ 1.00，position-only ≈ 0.99。

### 结论

- CIC 上位置逆变换同样有效（position-only 97.6%，combined 99.8%）；
- LR/MLP 下 position-only 几乎足够（97–99%）；
- RF 下 position-only 只有 46%，combined 才能到 102%；
- 因此“位置是主要成分”对 LR/MLP 成立，对 RF 则是 **code+position 组合**更重要；
- 跨数据集、跨模型总体一致：combined 都恢复到 oracle 水平。



## 25. RF 的 phase 配对复验

| 数据集 | phase | unaware | position | combined | oracle | position 恢复 | combined 恢复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UNSW | 0 | 0.868 | 0.910 | 0.962 | 0.960 | **0.46** | 1.02 |
| UNSW | 1 | 0.851 | 0.925 | 0.960 | 0.960 | **0.68** | 1.00 |
| CIC | 0 | 0.694 | 0.970 | 0.991 | 0.990 | **0.93** | 1.00 |
| CIC | 1 | 0.694 | 0.953 | 0.991 | 0.990 | **0.88** | 1.00 |

结论：

- UNSW-RF position 恢复在 46–68% 之间波动；
- CIC-RF 在 88–93% 之间；
- **46% 和 93% 都不是稳定常数**；
- combined 在两组都约 100%，稳定；
- RF 的 code-position 交互是 **数据集 + phase 依赖**，不能写成模型固有属性；
- 只有两个 phase，无法给可靠 paired CI；需要更多 phase/seed。



## 24. CIC + RF/MLP 与 detector 逐 flow

### CIC + RF

| 条件 | J(H) |
| --- | ---: |
| unaware | 0.694 |
| position-only | 0.970 |
| combined | 0.991 |
| oracle | 0.990 |

恢复：position ≈ 0.932，combined ≈ 1.003。

### CIC + MLP

| 条件 | J(H) |
| --- | ---: |
| unaware | 0.707 |
| position-only | 0.974 |
| combined | 0.975 |
| oracle | 0.976 |

恢复：position ≈ 0.994，combined ≈ 0.996。

### RF 的模型/数据集交互

| 数据集 | RF position-only | RF combined |
| --- | ---: | ---: |
| UNSW | 0.46 | 1.02 |
| CIC | 0.93 | 1.00 |

结论：RF 的 code-position 交互是 **数据集依赖**，不是模型固有属性。UNSW RF 需要 combined，CIC RF 位置已足够。

### Detector 逐 flow

300 条分层随机 flow：

| detector | 错误数 | 错误率 |
| --- | ---: | ---: |
| naive `code_index` | 39 | 13.0% |
| joint `detect_code_rows` | **1** | **0.3%** |

naive 混淆：

- 1→4：15
- 2→3：14
- 3→2：9
- 5→2：1

joint 仅 4→5 一次。

这解释了 0.748 → 0.882：naive 在 code 1/2/3 上系统错分；joint 联合搜索 code 与位置，几乎消除错误。



## 27. Known-mapping 三 Policy

| policy | J(H) |
| --- | ---: |
| Fixed | 0.884 |
| Periodic | 0.884 |
| Per-flow | 0.883 |

结论：

- 在 oracle trigger + 正确序列重构下，三种 policy 的 BA 几乎相同；
- **Known-mapping 上界对 policy 不敏感**；
- Periodic/Per-flow 的劣势只出现在 mapping 未知或 detector 有误差时；
- 因此后续 adaptive 实验的核心是“推断 mapping 的成本”，不是逆变换本身。



## 29. Adaptive Periodic pilot（未通过）

### 结果

| attacker | n=0 | n=64 | n=256 | n=1024 |
| --- | ---: | ---: | ---: | ---: |
| probe | 0.817 | 0.685 | 0.638 | 0.639 |
| finetune | 0.817 | 0.820 | 0.809 | 0.818 |

### 问题

1. **state inference 始终为 0**：
   - 位置似然在 mapping 置换下边际不变；
   - 无标签数据无法识别置换；
   - 当前 canonicalization 用的是错误 state。
2. **probe 标签抽样未按类分层**：
   - n 增大反而更差；
   - 不是 adaptive 能力，而是抽样偏差。
3. **finetune ≈0.82 不能解释**：
   - 它是在错误 canonicalization 上微调；
   - 不能当 adaptive 结果。

### 结论

- adaptive 分支 **未通过语义验证**；
- 不能报告 Adaptive-Probe / Adaptive-Finetune 的恢复率；
- 下一步必须用 **标签做 state selection**：
  - 每个候选 state 下 canonicalize calibration 样本；
  - 用 n 标签做 cross-validation，选最优 state；
  - n=0 时只能使用先验/identity，并明确标记为无自适应能力。



## 30. Candidate-state CV audit

17 个有效 block（按类分层抽 64 标签）：

| 指标 | 值 |
| --- | ---: |
| true state top1 | **5 / 17（29%）** |
| mean rank | **1.71** |
| true score 范围 | 0.906–1.000 |
| best score 范围 | 0.921–1.000 |

结论：

- 64 标签只能弱区分 state；
- 大多数候选 state 的 CV 分数都接近 1.0，差距很小；
- true state 通常在前 2，但不稳定 top1；
- **label-based selector 需要更大 n（256/1024）或更强的 canonicalization**；
- 这也解释了为什么无标签 state inference 会退化为 state0。



## 32. Frozen historical classifier audit

| n | top1（best==true） | mean rank | mean margin |
| ---: | ---: | ---: | ---: |
| 64 | 14/16 | 1.00 | **0.00** |
| 256 | 15/16 | 1.00 | **0.00** |

Oracle canonical BA：0.885。

问题：

- 多个 candidate 的 frozen accuracy 完全相同（常见 1.0 或同一低值）；
- `best_state` 与 true 不同时，score 也相同，rank 仍为 1；
- mean margin 恒为 0；
- 说明这个评分函数**饱和/打平**，不能区分正确与错误 canonicalization。

解释：

- 仅用 frozen accuracy on n labels 不够；
- 需要连续分数（log-loss / confidence margin / likelihood），而不是硬 accuracy；
- 也需要检查 trace-free canonicalization 是否对不同 state 产生足够差异。

结论：

- 当前 frozen-classifier selector **未通过**；
- 不能据此判断 mapping state 是否可识别；
- 下一步应改用连续评分并检查 canonicalization 的 state sensitivity。



## 34. State sensitivity / continuous-score audit（未运行成功）

尝试：

- 每个 block 前 50% 为 calibration pool；
- pool 内按类分层嵌套标签；
- 对每个 candidate state 计算 canonical feature distance、log-loss、Brier、margin。

结果：

- UNSW 10 天 block 的前 50% flow 经常只有单一类别；
- `len(unique(y_pool)) < 3`，所有 block 被跳过；
- 17 个 block 没有得到有效 continuous-score 样本；
- 因此没有 log-loss/Brier/margin 结果。

结论：

- 这不是 selector 失败，而是 **block/calibration pool 设计问题**；
- 时间前 50% 与类别覆盖在 UNSW 上冲突；
- 必须先重新定义 block：
  - 更大 block（30 天）；
  - 或按时间切分后做跨天分层；
  - 或选择类别覆盖足够的 block；
- 在 pool 满足类别覆盖前，不能评估 continuous selector。



## 36. Block coverage audit

### 汇总

| block 天数 | block 数 | ≥3 类 | ≥5 类 | 全部 8 类 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 18 | 17 | 7 | 0 |
| 20 | 9 | 9 | 4 | 0 |
| 30 | 6 | 6 | 5 | 0 |

### 观察

- **没有任何方案让 calibration pool 覆盖全部 8 类**；
- class 4 在多数 block 前半段完全缺席；
- 有些 block 的少数类只有 2–3 条样本；
- 10 天 block 多，但类别支持弱；
- 30 天 block 类别覆盖最好，但只有 6 个 block；
- 20 天是折中：9 个 block，4 个 ≥5 类。

### 重要含义

这验证了 Adaptation Cost 的第二个维度：

\[
\text{Adaptation Cost}=f(\text{Label Budget},\text{Label Acquisition Time})
\]

某些设备在 calibration window 内根本没有出现，攻击者无法获得语义锚点。这不是样本量不足，而是时间窗口内没有观测支持。

### 后续建议

- 先用 20 天 block 做 selector audit（9 block，保留 phase/seed 重复）；
- 只保留 calibration pool ≥3 类或 ≥5 类的 block，并明确报告筛选；
- 标签预算按每类可用样本上限截断；
- 不要假装全覆盖；报告每类实际样本数。



## 38. 20-day projection & feature sensitivity

### Block 覆盖

| block | cal flows | classes | counts（非零） |
| ---: | ---: | ---: | --- |
| 0 | 2816 | 5 | 31,892,1721,40,132 |
| 1 | 2718 | 3 | 898,1795,25 |
| 2 | 1951 | 4 | 430,21,230,1270 |
| 3 | 8461 | 7 | 2485,3045,1234,876,155,358,308 |
| 4 | 7710 | 3 | 6209,436,1065 |
| 5 | 85514 | 5 | 75018,9517,264,672,43 |
| 6 | 7271 | 4 | 6505,2,99,665 |
| 7 | 27656 | 5 | 15200,6867,1931,2804,854 |
| 8 | 3325 | 3 | 3156,31,138 |

### Projection equivalence

所有 block 的候選 state 等价组均为 **singleton**：

- 在出现的类别集合上，6 个 state 的 `(code, position)` 投影互不相同；
- 因此 state 在原理上可区分，至少不是投影等价。

### Feature sensitivity

对 50 条 calibration flow，canonicalized feature 到 state0 的平均 L2：

| block | state1 | state2 | state3 | state4 | state5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.28 | 5.10 | 7.29 | 5.18 | 4.71 |
| 1 | 5.81 | 8.57 | 5.81 | 5.81 | 7.15 |
| 2 | 5.63 | 8.44 | 5.63 | 5.63 | 7.05 |
| 3 | 8.03 | 6.29 | 7.59 | 7.24 | 6.50 |
| 4 | 8.13 | 6.62 | 7.72 | 7.09 | 6.25 |
| 5 | 7.70 | 6.50 | 7.41 | 7.06 | 6.10 |
| 6 | 8.23 | 6.58 | 7.74 | 7.10 | 7.06 |
| 7 | 8.22 | 6.84 | 7.70 | 7.35 | 7.03 |
| 8 | 7.78 | 6.10 | 7.51 | 7.24 | 6.55 |

结论：

- Canonicalization 对 state **有明显敏感性**（距离 4.7–8.6）；
- 之前 selector 饱和不是“特征不变”；
- 问题更可能在历史模型评分/标签量，而不是表示层不可区分；
- 下一步可在 20 天、≥5 类的 block（0/3/5/7）做 continuous-score 诊断。



## 40. Pairwise distance + class-filtered continuous score

### 6×6 pairwise（20-day block，50 flows）

| block | true state | min off-diagonal | oracle distance true | oracle distance others |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 2.46 | 0.0 | 4.71–7.29 |
| 3 | 1 | **0.00** | 0.0 | 7.07–8.27 |
| 5 | 2 | **0.00** | 0.0 | 5.88–8.49 |
| 7 | 3 | 5.15 | 0.0 | 5.84–7.71 |

结论：

- 真实 state 到 Oracle 距离恒为 0；
- 其他 state 距离 4.7–8.5；
- 但 block3/5 存在 **state 对 canonical feature 完全重合**（min pairwise=0）；
- 因此“所有 state 两两可区分”不成立，可辨识性依赖 block 的类别集合。

### Class-filtered continuous score

只评估 `E_b = C_b ∩ H_b`，同一冻结历史 LR：

| block | n | actual classes | rank_LL | rank_margin | G_LL | G_margin |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 64 | 1 | 1 | 1 | 0.000 | 0.000 |
| 3 | 256 | 2 | 1 | 1 | 0.068 | 0.011 |
| 3 | 1024 | 6 | 1 | 1 | 0.051 | 0.003 |
| 5 | 64 | 1 | 1 | 1 | 4.727 | 0.713 |
| 5 | 256 | 2 | 1 | 1 | 3.914 | 0.599 |
| 5 | 1024 | 5 | 1 | 1 | 2.247 | 0.348 |
| 7 | 64 | 1 | 1 | 1 | 7.260 | 0.613 |
| 7 | 256 | 2 | 1 | 1 | 6.938 | 0.570 |
| 7 | 1024 | 5 | 1 | 1 | 4.715 | 0.375 |

观察：

- 真实 state 在连续评分上 rank=1；
- G_LL>0 表示真实 state 的 log-loss 优于所有错误 state；
- G_margin>0 同样成立；
- block3 的 n=64 只有 1 类，G_LL=0，不能过度解释；
- n=1024 多类下仍 rank1，但 G_LL 随类别覆盖增加而下降；
- 结论：**在类别覆盖足够的 block 上，历史 LR 可以识别真实 state；但并非所有 state 对都可区分，且小 n 的类别覆盖不足。**



## 42. Label-based selector v2（20-day）

### 关键结果

只在 `E_b=C_b∩H_b` 且 n 子集至少 2 类时评估：

| block | n | selected | true | gap | ambiguous | probe BA | finetune BA |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 1024 | 0 | 0 | 1.12 | no | 1.000 | 1.000 |
| 2 | 1024 | 1 | 1 | 0.25 | no | 0.516 | 0.865 |
| 3 | 256 | 1 | 1 | 0.065 | no | 1.000 | 0.762 |
| 3 | 1024 | 1 | 1 | 0.066 | no | 0.997 | 0.996 |
| 4 | 1024 | 2 | 2 | 1.82 | no | 0.500 | 0.957 |
| 5 | 256 | 2 | 2 | 4.15 | no | 0.333 | 0.971 |
| 5 | 1024 | 2 | 2 | 2.28 | no | 0.991 | 0.994 |
| 6 | 1024 | 3 | 3 | 6.03 | no | 0.396 | 0.901 |
| 7 | 256 | 3 | 3 | 7.27 | no | 0.400 | 0.908 |
| 7 | 1024 | 3 | 3 | 4.66 | no | 0.997 | 0.952 |
| 8 | 1024 | 4 | 4 | 16.60 | no | 1.000 | 0.774 |

### 观察

1. **Selector 在所有可评估单元都选中真实 state**；
2. gap 全为正，最小 0.065，最大 16.6；
3. 没有 ambiguous；
4. n=64 全部被跳过：子集只有 1 类，无法稳定评分；
5. n=256 部分可用，n=1024 全部可用；
6. Probe 在 n=256 时常常很差（0.33–0.40），因为标签类别覆盖不足；
7. Finetune 在 n=256 更稳（0.76–0.97），说明历史初始化在小标签下更鲁棒；
8. n=1024 时 Probe 多数接近 1.0，但 block2/4/6 仍偏低，说明类别覆盖/未知类仍影响。

### 解释边界

- Selector 可行，但只在 `E_b` 覆盖足够、n 子集多类时；
- n=64 在当前 20-day block 下不可用；
- Probe 与 Finetune 的差异与小 n 类别覆盖强相关；
- 不能把 n=256 Probe 的低 BA 解释为“从头训练差”，它同时缺少类别锚点。



## 44. 2 phase × 3 seed selector 汇总

6 组，20-day block，n=64/256/1024。

### Selector

| 预算 | 可评估单元 | Top-1 | mean gap | min gap | ambiguous |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0 | — | — | — | — |
| 256 | 18 | **18/18** | 3.204 | 0.013 | 0 |
| 1024 | 48 | **47/48** | 2.914 | 0.0016 | 1 |

- 总：**65/66**；
- 唯一失败：p1_s2 block6 n=1024，true=4，selected=5，gap=0.382；
- 唯一 ambiguous：p1_s1 block1 n=1024，gap=0.0016；
- n=64 全部因单类子集不可评估，不计为 selector 失败。

### Probe / Finetune

| 预算 | Probe mean | Finetune mean | before mean | known BA | paired FT−Probe mean | min / max |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 0.577 | **0.882** | 0.798 | 0.798 | **+0.305** | −0.238 / +0.658 |
| 1024 | 0.801 | **0.916** | 0.841 | 0.848 | **+0.116** | −0.226 / +0.555 |

### 负迁移

5 个单元出现 Finetune < before（下降 0.016–0.032）：

- p0_s0 block6 n=1024
- p0_s2 block7 n=256
- p1_s0 block6 n=1024
- p1_s0 block7 n=256
- p1_s2 block7 n=256

幅度较小，但说明微调并非总为正收益。

### 解释

- Selector 在多 phase/seed 下稳定：n=256 全对，n=1024 47/48；
- n=64 不可评估，不是 selector 失败；
- Finetune 平均显著优于 Probe，尤其 n=256（+0.305）；
- paired 差异存在负值，说明 Probe 在某些 block 反而更好；
- known-mapping matched BA 约 0.798（n256）/0.848（n1024），低于 oracle 0.884，说明 adaptive 仍有差距；
- 负迁移集中在 block 6/7，需结合 per-class recall 进一步分析。



## 45. 产物

- 脚本：`scripts/online_canonical.py`、`scripts/feature_canonical.py`、`scripts/observed_inverse.py`、`scripts/audit_identity.py`、`scripts/reconstruct_audit.py`、`scripts/event_canonical.py`
- 测试：`tests/test_canonicalization.py`、`tests/test_event_canonical.py`
- 结果：
  - `artifacts/es/online_canonical_fixed.json`
  - `artifacts/es/online_canonical_periodic.json`
  - `artifacts/es/feature_canonical_reference.json`
- 分支：`stage2-coordination`

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

| flows | max_abs_diff | nonzero |
| ---: | ---: | ---: |
| 200 | **0.0** | **0** |

说明：

- 已加入 identity 短路（canonical code == actual code 时逐元素返回）；
- 同时定位到短流 fallback：计划 anchor=10、但流中无第 10 个真实包时，FADE 会把 trigger fallback 到末尾；
- 修复 fallback 前有 80/200 mismatch，identity 短路后为 0。

但 identity 短路只验证恒等路径；跨状态重构仍需 synthetic round-trip。



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



## 14. 产物

- 脚本：`scripts/online_canonical.py`、`scripts/feature_canonical.py`、`scripts/observed_inverse.py`、`scripts/audit_identity.py`、`scripts/reconstruct_audit.py`、`scripts/event_canonical.py`
- 测试：`tests/test_canonicalization.py`、`tests/test_event_canonical.py`
- 结果：
  - `artifacts/es/online_canonical_fixed.json`
  - `artifacts/es/online_canonical_periodic.json`
  - `artifacts/es/feature_canonical_reference.json`
- 分支：`stage2-coordination`

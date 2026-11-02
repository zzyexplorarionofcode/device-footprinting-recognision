# 在线适应 pilot：n 前缀标签、盲测、mapping-aware

## 1. 协议

- UNSW，30 天/block，共 5 个 block；
- 训练：只用过去 block；
- 当前 block：前 n 条标签给攻击者，剩余流量盲测；
- n=0/64/256/1024；
- 更新方式：permutation / probe / fine-tune；
- 攻击者：
  - Fixed/Periodic 的 mapping-unaware；
  - Periodic 另加 schedule-known、state-inferred；
- 历史最多 3000 条，SGD log-loss，3 次/方法。

## 2. J(H,n)

| policy | attacker | method | n=0 | n=64 | n=256 | n=1024 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Fixed | unaware | permutation | 0.757 | 0.677 | 0.677 | 0.677 |
| Fixed | unaware | probe | 0.740 | 0.645 | 0.646 | 0.646 |
| Fixed | unaware | finetune | 0.743 | 0.747 | 0.751 | 0.754 |
| Periodic | unaware | permutation | 0.551 | 0.531 | 0.531 | 0.531 |
| Periodic | unaware | probe | 0.549 | 0.455 | 0.457 | 0.457 |
| Periodic | unaware | finetune | 0.521 | 0.529 | 0.538 | 0.542 |
| Periodic | schedule-known | permutation | 0.544 | 0.534 | 0.534 | 0.531 |
| Periodic | schedule-known | probe | 0.583 | 0.486 | 0.488 | 0.488 |
| Periodic | schedule-known | finetune | 0.583 | 0.593 | 0.592 | 0.596 |
| Periodic | state-inferred | permutation | 0.545 | 0.568 | 0.568 | 0.564 |
| Periodic | state-inferred | probe | 0.548 | 0.488 | 0.489 | 0.489 |
| Periodic | state-inferred | finetune | 0.539 | 0.533 | 0.537 | 0.540 |

## 3. 观察

1. **Periodic 仍然低于 Fixed**：
   - Fixed permutation J≈0.68–0.76；
   - Periodic J≈0.53–0.58；
   - 5 个 block 上，Periodic 的阶段性劣势没有消失。

2. **Mapping-aware 只有小幅帮助**：
   - schedule-known finetune J≈0.596，比 unaware finetune 0.542 高约 0.05；
   - state-inferred permutation J≈0.568，比 unaware 0.531 高约 0.04；
   - 没有把 Periodic 拉回 Fixed 水平。

3. **Probe 在 n=64 时可能伤害第一个 block**：
   - Fixed probe block1 从 0.666（n=0）降到 0.193；
   - 少量标签从头训练不稳定；
   - fine-tune 更稳，J 基本不降。

4. **n=1024 相比 64 提升有限**：
   - 多数曲线在 n=64 后饱和；
   - 可能因为 block 内类别覆盖已足够，或 SGD/特征限制；
   - 需要更大 block 数和更细 n 网格确认。

5. **N_τ 不可用**：
   - 多数 block 在 n=0 已达到 0.5，N_τ 中位数=0；
   - 2 个 block 到 n=1024 仍未达到（censored）；
   - 5 个 block 不足以估 N_τ。

## 4. BA 曲线

| policy | attacker | n | BA_1..BA_5 |
| --- | --- | ---: | --- |
| Fixed | unaware | 0 | 0.666,0.792,0.970,0.558,0.799 |
| Fixed | unaware | 64 | 0.533,0.526,0.970,0.558,0.799 |
| Periodic | unaware | 0 | 0.669,0.311,0.905,0.190,0.682 |
| Periodic | unaware | 64 | 0.538,0.340,0.905,0.190,0.682 |
| Periodic | schedule-known | 0 | 0.670,0.252,0.900,0.130,0.768 |
| Periodic | schedule-known | 64 | 0.539,0.333,0.900,0.130,0.768 |
| Periodic | state-inferred | 0 | 0.671,0.262,0.904,0.173,0.715 |
| Periodic | state-inferred | 64 | 0.539,0.341,0.923,0.184,0.851 |

## 5. 限制

- 只有 5 个 block，统计功效低；
- 历史最多 3000 条，SGD，未用 RF/MLP；
- 没有 per-flow；
- 没有 T_recovery、多相位、多 seed；
- probe/fine-tune 只做 LR/SGD；
- n 的曲线在 64 后饱和，可能是 block 大小/模型容量造成；
- CIC 未跑。

## 6. 结论（pilot）

- **Periodic 的优势在在线设置下仍然存在**；
- **mapping-aware 只解释了很小一部分**，没有消除 Periodic 劣势；
- 目前不能说“Periodic 优势完全来自攻击者未利用状态”；
- 也不能说“mapping-aware 一定能恢复”，因为 pilot 的 block 数和模型容量有限。

## 7. 下一步

1. 扩到 10 天/block（约 13–17 blocks），重跑同样矩阵；
2. 加入 per-flow；
3. 加 T_recovery、多相位/seed；
4. 再扩 CIC、RF/MLP；
5. 用完整 34 block 版本估 N_τ。

## 8. 产物

- 脚本：`scripts/online_adaptation.py`
- 结果：`artifacts/es/online_adaptation_unsw8.json`
- 分支：`stage2-coordination`

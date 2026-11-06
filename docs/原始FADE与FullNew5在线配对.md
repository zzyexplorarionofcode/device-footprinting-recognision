---
layout: null
---

# 原始 FADE 与 Full New5：统一在线协议配对回放

## 问题与边界

比较同一在线回放下 `J_FullNew5 − J_OriginalFADE`，BA 越低防御越强。本实验是论文机制的在线实例化，不是论文 Table I 的复现：论文按防御样本替换比例再训练，未指定以观测日计的映射轮换周期。

## 冻结协议

- 数据：UNSW8 开发回放，20 日热身、随后 150 个观测日。三条 Full New5 预存随机序列，日历、4000 热身流、每日 8 个评分后才释放的设备标签、同日 Blind 均复用已冻结清单。共 17,259 Blind 流；每条序列回放 18,459 条训练与 Blind 流。Blind 标签不进入训练。
- 原始 FADE 臂：使用初始 `configs/methods/fade_unsw8.json`。固定 weakener 与诱导码本；每 25 个观测日轮换一次共享的类别→码映射。首映射和首轮更新来自原配置，后续采用确定的循环置换，确保所有类别在每次更新时换码。热身流使用初始映射。
- Full New5 臂：复用已冻结的每 5 日完整新码本、每流均匀 IID 六个 Mapping State 结果。热身流按原 Full New5 协议每流 IID。两臂热身 *索引与标签* 一样，防御后的训练流表示随各自策略改变。
- 攻击者：相同标准化、LR / 64 树 RF、累计历史或最近 14 日历史，每日先预测 Blind，后释放当日设备标签。视图 D 为原 64 维序列特征；视图 C 为防御后可观察的无序统计，不读取真实包标记或 State。
- 同一真实流、日历、标签账本与 Blind 配对；原始 FADE 的共享 State 与 Full New5 的逐流 State 按策略各自生成，不强制相同。
- 原文还允许触发式轮换及可选码本刷新；本实验只实例化其固定码本加映射轮换主机制，不能代表所有原论文可选实现。原始配置与后期配置的 Trigger 码值、Weakener 和硬预算相同，位置码本不同。

## 日均 Post BA（3 条预存序列等权平均）

| 视图 | 攻击者 | 原始 FADE | Full New5 | Full − 原始 | 逐序列差值 |
| --- | --- | ---: | ---: | ---: | --- |
| D | 累计 LR | 0.7943 | 0.7683 | −0.0260 | −0.0197 / −0.0212 / −0.0371 |
| D | 14 日 LR | 0.7975 | 0.7405 | −0.0570 | −0.0673 / −0.0524 / −0.0512 |
| D | 累计 RF | 0.8845 | 0.8818 | −0.0027 | +0.0048 / −0.0065 / −0.0063 |
| D | 14 日 RF | 0.7878 | 0.7543 | −0.0334 | −0.0301 / −0.0368 / −0.0334 |
| C | 累计 LR | 0.9544 | 0.9544 | 0 | 0 / 0 / 0 |
| C | 14 日 LR | 0.8944 | 0.8944 | 0 | 0 / 0 / 0 |
| C | 累计 RF | 0.9647 | 0.9647 | 0 | 0 / 0 / 0 |
| C | 14 日 RF | 0.8524 | 0.8524 | 0 | 0 / 0 / 0 |

上述原始臂由固定映射序列决定，三次重复使用同一原始臂；随机重复来自 Full New5，不能把三次当成三个独立的时间捕获。

## 配对审计与解释

- 结果文件逐日核对日期、Blind 类别支持、累计可用标签数和各模型实际训练标签数。
- 两臂各 18,459 条回放流的实际新增字节、时延、事件数逐流一致：平均约 629.458 B、0.5503 s、5 个事件；最大 630 B、0.8307 s、5 个事件。该审计只针对回放流，热身原始臂另行记录成本。
- 两臂的视图 C 向量在全部 18,459 条回放流及 4,000 条热身流上逐流相同，因此 C 攻击结果相同不是分类器偶然收敛；这仍不表示所有可观察统计都不变。
- 64 维 LR 和滑动 RF 对 Full New5 有条件性收益；累计 RF 基本持平、逐序列变号。视图 C 没有收益且 BA 很高，因此这项方法增量不能解释成设备身份被充分隐藏。
- 本结果仅适用于 UNSW8 已见开发时间段、指定 25 日原始轮换实例、当前攻击特征和模型。没有跨独立捕获的显著性或泛化结论。

## 复现

```powershell
python -X utf8 -B -m scripts.original_fade_online_compare --seed 0
python -X utf8 -B -m scripts.original_fade_online_compare --seed 1
python -X utf8 -B -m scripts.original_fade_online_compare --seed 2
python -X utf8 -B -m scripts.audit_original_fade_cost
```

代码：`scripts/original_fade_online_compare.py`、`scripts/audit_original_fade_cost.py`。逐日结果：`artifacts/es/original_fade_online_compare_s{0,1,2}.json`；逐流开销与视图一致性计数：`artifacts/es/original_fade_cost_audit_s{0,1,2}.json`。Full New5 的原冻结结果由 `artifacts/es/residual_views_unsw8_s{0,1,2}.json` 复用。

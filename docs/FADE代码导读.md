# FADE 代码导读

FADE 对每条输入会话的包长、方向和间隔进行变换。它先随机弱化设备原有特征，再插入与设备类别关联的诱导码；攻击者在旧映射下学习这些关联后，防御方切换设备到诱导码的映射。

## 输入与输出

核心文件是 `methods/method33.py`，公开入口为 `apply_method33`。四个流量数组 `lengths`、`gaps`、`directions`、`mask` 的形状是 `[N,32]`，另有 `[N]` 的 `labels`。方向 `+1` 表示上行，`-1` 表示下行。有效包必须是连续前缀，首包 gap 为零。

输出七元组依次为长度、间隔、方向、掩码、逐流成本、元数据和逐流审计。开启 `include_event_trace` 可查看完整事件来源。固定观察窗口只显示前 32 个发送事件，完整轨迹保留被挤出观察窗口的真实包和 dummy，用于核算完整开销。

## 沿调用链阅读

1. `method33_tools.load_config` 读取 JSON，校验设备顺序、预算、码字、映射及抖动参数；`apply_configured_method` 负责把数据与配置传入核心。
2. `apply_method33` 是正式变换入口，复用 `apply_method33_base` 的基础阶段，再完成诱导码内部的零和抖动。
3. `apply_method33_base` 为每条流建立随机状态，先抽取弱化 dummy 计划并预留预算，再依次处理真实包。
4. 真实上行包按概率进行 padding 和 delay；弱化 dummy 从配置给出的上行位置中抽样插入。弱化器的随机规则不读取类别标签。
5. `labels` 与所选 `mapping` 决定诱导码。诱导码包含两个 dummy 的长度与 gap，插入位置也绑定码字。`default_trigger_positions` 和 `resolve_trigger_positions` 定义与检查位置表。
6. 短流达不到指定位置时，在流尾补入剩余诱导 dummy，并在审计中记录回退。真实包顺序及完整开销仍需保持。
7. `_render_fixed_window` 把完整发送事件转换成攻击者可见的前 32 个事件。第一 gap 固定为零，因为观察者没有第一包前的共享时钟。
8. `jitter_trigger_code` 在相对界限内交换两个 dummy 的长度和 gap，总字节与总 gap 保持不变；位置不随这一步抖动改变。

## 预算分配

本包 FADE 配置的总预算是 630 B、1.5 s、最多 6 个新增事件。弱化器分到 347 B、1.05 s、3 个 dummy；诱导码分到 283 B、0.45 s、2 个 dummy。最小 gap 为 4 ms。典型实际 dummy 数为 5，新增事件上限无需用满。

`experiment_support/budget.py` 的 `PhysicalBudget` 审计完整输出的额外字节、完成时间增量和新增发送事件。`experiment_support/metrics.py` 区分最后真实包的延迟、最后发送事件的延迟和尾部覆盖时长。配置中的内部 delay 记账与观测到的完成时间增量应分别理解。

## 训练和轮换

FADE 运行时直接使用配置中的弱化器、码本与映射，不训练一个生成网络。JSON 中的 `selection` 保存候选和选择协议元数据，本学习包的 Method33 CLI 提供 `inspect` 与 `apply`，不提供重新搜索弱化器或码本的功能。

单次评估中，`train_mapping='old'`，`test_mapping='next'`。重训练比例表示攻击者训练集中被旧映射防御样本替换的比例，其余仍是干净样本；即使该比例为 100%，也不表示攻击者见过新映射样本。

多次轮换见 `experiment_support/rotation_history.py`。`mapping_schedule` 与 `next_distinct_mapping` 按码字归一化距离选择下一映射，约束历史类别与码字关联不重复。`experiments/run_cumulative_rotation.py` 处理累计历史的持续攻击者，`run_new_attacker_rotation.py` 处理仅当前映射数据的新攻击者；统一命令为 `run_rotation.py`。

## 基线对应代码

| 方法 | 主要文件 | 关注点 |
| --- | --- | --- |
| Random Rate、Constant Rate、Traffic Morphing | `baselines/traditional.py` | 随机速率、恒速整形、训练分布与变换 |
| IoTGAN | `baselines/iotgan_aligned.py` | 针对查询分类器训练；共同包长与时序约束 |
| iPET | `baselines/ipet/ipet_v2aligned/` | 生成器、判别器、物理 slot 投影和扰动池注入 |
| Glimpse | `baselines/glimpse/glimpse_seq64_oracle_v2/core.py` | DDPG、padding、delay、分段和 oracle 完整窗口 |
| Securitas | `baselines/securitas/securitas_adapted/` | 代理、策略学习、分片、dummy 与 ICMP 响应仿真 |

`baselines/high_budget.py` 虽然名称含 high，实际承担公共预算的运行适配，仍是当前入口的依赖。IoTGAN 双向实现由基线包初始化代码加载，因此也保留；默认实验使用单向模式。Securitas 的 `method32-captures` 是历史命名的数据划分模式，不会调用 Method32 算法。

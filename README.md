# FADE 论文代码学习包

本目录提供 FADE（仓库内部名称 Method33）的完整流量变换代码、对比基线、CIC 与 UNSW 数据处理代码，以及训练评估入口。建议先阅读下面的代码导图，再用合成数据测试，最后准备真实数据运行。

**先看这里**：[代码架构与阅读指南](docs/代码架构与阅读指南.md) 给出分层架构、目录地图、三条阅读路线（算法 / 数据 / 实验）、数据契约和复杂语法速查。

## 内容与阅读顺序

| 顺序 | 文件或目录 | 学习内容 |
| --- | --- | --- |
| 1 | [数据处理与运行说明](docs/数据处理与运行说明.md) | 数据集位置、提取、冻结划分和运行命令 |
| 2 | [FADE 代码导读](docs/FADE代码导读.md) | 弱化器、诱导码、位置轮换、零和抖动与预算 |
| 3 | `methods/method33.py` | 完整 FADE 算法，主要入口 `apply_method33` |
| 4 | `methods/method33_tools.py` | 配置加载、数据对齐、应用、审计和 CLI |
| 5 | `data_pipeline/` | PCAP 解析、会话提取、特征编码和数据划分 |
| 6 | `baselines/` | Random Rate、Constant Rate、Traffic Morphing、IoTGAN、iPET、Glimpse、Securitas |
| 7 | `experiments/adapters.py` 与 `experiments/run_experiment.py` | 把方法接到共同的训练、重训练、测试和开销统计流程 |
| 8 | `experiments/lstm_attacker.py` | LSTM 攻击者；其他攻击者训练函数见 `methods/shared_tools.py` |
| 9 | `experiment_support/` | 物理预算、时间指标、来源校验、断点续跑与多次轮换 |

只保留 FADE 算法，没有 Method32 算法和配置。原运行器共用的分组划分、重训练掩码、攻击者拟合和码字距离函数已摘为 `methods/shared_tools.py`。各算法的核心变换与训练实现保持原仓库内容，整理改动主要是路径、公共函数导入、配置组织和配套测试。

本包不含原始 PCAP、处理后的 NPZ、已训练权重、实验结果、论文 LaTeX/PDF、消融入口、临时试验入口、历史快照、服务器登录资料和虚拟环境。多次映射轮换属于论文方法的评估流程，因此保留了新攻击者与持续攻击者两种入口。

## 环境与首次检查

使用 Python 3.11，在本目录打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B scripts/run_checks.py
```

检查使用合成数据，不需要真实数据集，包括 FADE 行为、会话边界、物理开销、iPET 投影、Glimpse 分片约束和少量攻击者训练。不会运行完整论文实验。

FADE、IoTGAN、iPET、Securitas 和 LSTM 使用根目录依赖即可。Glimpse 的 DDPG 训练另依赖 `baselines/glimpse/requirements.txt`，详见 [环境补充说明](docs/环境补充说明.md)。不安装 TensorFlow 也能运行上述首次检查以及 Glimpse 的核心仿真检查。

## 当前配置

| 入口配置 | 新增字节上限 | 完成时间增量上限 | 新增事件上限 | 数据集 |
| --- | ---: | ---: | ---: | --- |
| `configs/experiments/current.py` | 630 B | FADE 1.5 s | 6 | CIC Active 10 类、UNSW 8 类 |
| `configs/experiments/baselines.py` | 630 B | 基线 3.5 s | 6 | 同上；Securitas 使用独立入口 |
| `configs/experiments/cumulative_rotation.py` | 630 B | FADE 1.5 s | 6 | 同上，三次映射轮换 |

FADE 配置的时间上限为 1.5 秒，最小 gap 为 4 ms。大预算、消融及旧 UNSW 实验配置未包含。

FADE 主评估与轮换配置使用单次重复，基线配置使用两次重复；独立 Securitas 默认三次。

完整训练前先准备数据，并执行对应 `--dry-run`。命令详见 [数据处理与运行说明](docs/数据处理与运行说明.md)。


## 第一阶段实验结论与结果

- 结论汇总：[docs/第一阶段实验结论.md](docs/第一阶段实验结论.md)（v2，固定 RNG）
- 冻结版总结：[docs/第一阶段总结_v2.md](docs/第一阶段总结_v2.md)（结论 + 6 张图）
- 一页简报：[docs/第一阶段简报.md](docs/第一阶段简报.md)
- 自动论文表格：[docs/v2_tables.md](docs/v2_tables.md)
- 评估协议：[docs/评估协议.md](docs/评估协议.md)
- PingPong 审计：[docs/PingPong审计.md](docs/PingPong审计.md)
- 结果 JSON：`artifacts/es/v2_*.json` 为固定 RNG 的权威结果；旧 `artifacts/es/*.json` 仅作历史记录，不可与 v2 混用。
- 数据划分：artifacts/splits/（只保留 split_indices.npz）
- 原始数据、处理后的 NPZ、运行产物不提交；复现见 [docs/数据处理与运行说明.md](docs/数据处理与运行说明.md)。


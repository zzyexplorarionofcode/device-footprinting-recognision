# iPET adapted physical-v5 (PyTorch v2-aligned base)

这是当前维护的 iPET adapted physical-v5 实现目录。它以隔离的 PyTorch
v2-aligned 代码为基础，并明确保留项目运行时适配语义；不宣称上游
TensorFlow 代码的逐字节、逐事件原样复现。当前 API 位于
`baselines/ipet/ipet_v2aligned/`，统一实验通过 `experiments/adapters.py` 接入。

运行时和训练/扰动池使用同一个物理 slot 投影：计数按最近偶数取整，
计数为零时该方向是 no-op；同一 omega slot 的 incoming/outgoing 读取同一
扰动池行；每包帧长按总字节除以计数后裁剪到 60--1514，再按最近偶数取整。
omega 边界使用稳定的 slot 归属规则，长静默段保留其绝对 slot 行号。策略
文件 schema 为 5，旧 schema 或缺少 physical-v5 运行时标记的权重必须重新训练。

## 本版本完成的五项对齐

1. 生成器训练和每个 stage 的判别器重训练统一使用 batch size 32；初始判别器
   仍按官方 `Train_Mseq.py` 使用 batch size 128。
2. 训练生成器时冻结判别器参数，但保持判别器为训练模式，因此 Dropout 0.2
   继续生效。
3. NAdam 显式使用 TensorFlow 2.5 默认参数：`lr=0.001`、`eps=1e-7`、
   `betas=(0.9, 0.999)`、`momentum_decay=0.004`。
4. 初始判别器训练和所有 stage 的判别器重训练复用同一个 Adam 对象，保留
   Adam 的一阶/二阶矩状态和累计 step。
5. 每个设备类别懒加载生成 20,000 个 `40x4` 扰动矩阵，展平后为
   `800,000x4` 的长扰动池。4/6/10分类通过相同的类别无关实现调用该逻辑。

## 独立 TLS 流的起点适配

作者代码把长扰动序列用于连续设备流量。本项目的数据单位是独立 TLS 流。
为了避免每条TLS流都从同一行开始，同时保留“每设备一个长扰动池”的设计，
本版本使用已有的 `session_seed` 从 `40` 行矩阵的边界选择起点。流内按顺序
消费后续行，只有消耗完全部 800,000 行才循环。

扰动池采用 `float32`，每个设备类别约占 12.8 MB；4类约51.2 MB，6类约
76.8 MB。扰动池不写入策略文件，策略加载后根据保存的生成器权重和固定seed
确定性地重建。

## iPET-physical-v5-budgeted（兼容入口 ipet_budgeted_v3）

通用实验器另外注册了派生方法 `ipet_budgeted_v3`，结果显示名为
`iPET-physical-v5-budgeted`。它复用本目录的物理-v5生成器与时间顺序，不改变
真实事件；应用时逐个检查dummy完整帧，仅在逐流剩余字节预算和dummy数量预算
都足够时发出。默认配置为最多新增420 B、最多3个dummy包，与当前10分类
Method32的硬预算一致。

该版本是明确改变运行算法的预算消融，不是上游iPET原样复现。原始无总预算入口
`apply_policy`保持不变；预算入口为`apply_budgeted_policy_v3`。每条流的原始提议
量、拒绝量、实际发出量和硬预算断言都会写入审计记录，预算值则随通用实验配置
进入manifest。

## 目录结构

- `ipet_v2aligned/model.py`：模型、训练循环、优化器状态和20,000矩阵生成。
- `ipet_v2aligned/traffic.py`：slot特征、长扰动池计划和流量注入。
- `ipet_v2aligned/baseline.py`：统一策略接口、类别无关逻辑和序列化。
- `run_self_test.py`：对齐、4/6/10分类及v3硬预算回归测试。
- `run_unsw_comparison.py`：复用现有UNSW比较程序，但只替换其iPET依赖。

## 验证

从仓库任意目录执行：

```powershell
python iot_uap_simple/defense_baselines/ipet-pytorch-v2aligned/run_self_test.py
```

## 运行4分类和6分类实验

默认运行项目允许的单向、预算控制比较，并依次处理4类和6类：

```powershell
python iot_uap_simple/defense_baselines/ipet-pytorch-v2aligned/run_unsw_comparison.py `
  --comparison outbound-budget --classes both
```

运行双向原生动作空间比较：

```powershell
python iot_uap_simple/defense_baselines/ipet-pytorch-v2aligned/run_unsw_comparison.py `
  --comparison bidirectional-native --classes both
```

原实验程序的参数可以继续附加，例如 `--repeat-count`、`--ratios`、
`--budget-mode` 和各训练轮数参数。若没有传入 `--output-dir`，结果会保存在
本目录内，避免与旧iPET结果混淆。

## 仍然保留的项目适配

本版本保留了项目逐流比较所需的前32事件观察窗口、silence置零、
Ethernet帧长度及60--1514字节裁剪。它们是数据集/执行适配，不是官方
TensorFlow端到端语义的等价翻写，详见同目录
[`ipet-pytorch-v2aligned_代码解析.html`](ipet-pytorch-v2aligned_代码解析.html)。

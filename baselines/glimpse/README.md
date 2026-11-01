# Glimpse-Seq64-Oracle-v2

独立的 Seq64 离线 Glimpse 适配实现。整个目录可单独复制使用，代码不导入原版
`Glimpse-Seq64-Oracle`、`our_method`、`defense_baselines.common` 或通用实验运行器。
第三方依赖见 `requirements.txt`。v1 和 v2 使用不同的 Python 包名、策略类及文件格式。

## 四项修改

1. **延迟时序**：恢复包时间，推迟上行包并保持上行FIFO，合并原有下行事件后
   重新排序、计算间隔。首包间隔始终为0，首包等待开销单独记录。
2. **三种动作**：只保留 `payload_padding`、`ip_tcp_fragmentation`、`packet_delay`。
   Actor输出为4维，前三维argmax选动作，第四维表示强度。TTL和reset已从动作、
   解码、Packet操作与动作计数中删除。
3. **部署目标Actor**：原论文算法2（PDF第6页，期刊第1215页）明确输出
   `theta_targ`。训练用主Actor采集经验并更新，最终保存、加载和部署目标Actor。
   推理直接调用目标网络，不使用TF-Agents中绑定主Actor的`agent.policy`。
4. **分片字节与开销**：具体采用TCP载荷分段，每段复制完整头部；载荷按整数
   字节分配，完整保留载荷区间，额外头部及必要的以太网填充均计入帧长和开销。

原文：Tan et al., *You Can Glimpse but You Cannot Identify: Protect IoT Devices
From Being Fingerprinted*, DOI [10.1109/TDSC.2023.3275850](https://doi.org/10.1109/TDSC.2023.3275850)。
算法2第5行以主Actor选择训练动作，第23-24行软更新目标网络，Output指定目标
Actor。这些位置用于区分训练Actor和最终部署Actor；论文伪代码没有公开的细节
仍按本文件说明的适配规则实现。

## 文件与运行

```text
Glimpse-Seq64-Oracle-v2/
  glimpse_seq64_oracle_v2/
    __init__.py
    core.py
  demo.py
  self_test.py
  verify_standalone.py
  config.example.json
  requirements.txt
  README.md
```

使用 Python 3.11，在本目录执行：

```bash
python -m pip install -r requirements.txt
python self_test.py
python verify_standalone.py
```

`self_test.py --core-only` 不加载TensorFlow，检查特征、延迟、三种动作、分片守恒、
完整流开销和输入/策略格式。完整测试额外执行真实DDPG更新、目标网络软更新、
主/目标Actor差异验证、逐设备训练、批量/单流等价及保存/加载。
`verify_standalone.py` 将本目录的代码复制到临时目录，启动新进程并阻止旧版包及
其他项目模块导入，然后运行完整测试。

合成数据demo：

```bash
python demo.py --gamma 0.99 --rho 0.995 --reward-threshold 0.65 --training-iterations 300 --max-delay-seconds 0.1 --max-fragments 4 --target-update-period 1 --replay-buffer-capacity 10000 --exploration-noise-stddev 0.2 --exploration-noise-damping 0.15 --seed 1734000 --frame-header-bytes 54 --save-policy policy_v2.pkl
```

上述参数及 `config.example.json` 是明确公开的实验输入，不代表论文默认值。
合成demo只验证代码通路，不能作为真实数据集上的防御效果结论。

## 对已有NPZ使用独立API

输入 `lengths/gaps/directions/mask` 均为 `[N,32]`。长度为整数帧字节（允许用浮点
数组存储整数），间隔单位为秒，方向为 `+1/-1`，mask为连续有效前缀，首包间隔为0。
Seq64是32个 `direction*length/1514` 加32个 `log1p(min(gap,10)*1000)`。

```python
import json
import numpy as np
from glimpse_seq64_oracle_v2 import GlimpseConfig, GlimpseSeq64OracleV2, apply_policy

data = np.load("flows.npz", allow_pickle=False)
with open("config.example.json", encoding="utf-8") as stream:
    config = GlimpseConfig(**json.load(stream))
arrays = tuple(data[k] for k in ("lengths", "gaps", "directions", "mask"))
labels = data["labels"]

# split.npz由实验协议提供，含train_indices和test_indices，保证与其他方法一致。
# 分组/capture隔离应在划分时完成；本包不读取其他方法的划分代码。
split = np.load("split.npz", allow_pickle=False)
train_indices, test_indices = split["train_indices"], split["test_indices"]
headers = data["header_lengths"] if "header_lengths" in data.files else None
payloads = data["payload_lengths"] if "payload_lengths" in data.files else None

policy = GlimpseSeq64OracleV2.fit(
    *arrays, labels, config=config, train_indices=train_indices,
    header_lengths=headers, payload_lengths=payloads,
)
policy.save("policy_v2.pkl")
restored = GlimpseSeq64OracleV2.load("policy_v2.pkl")
result = apply_policy(
    restored, *(a[test_indices] for a in arrays), labels[test_indices],
    header_lengths=None if headers is None else headers[test_indices],
    payload_lengths=None if payloads is None else payloads[test_indices],
)
lengths_out, gaps_out, directions_out, mask_out, costs, metadata = result
```

每设备的IF和DDPG只使用 `train_indices` 中属于该设备的流。应用时的设备标签用于
选择保护该设备的策略，不是目标识别器的预测。`header_lengths/payload_lengths`
用于包操作，不加入Actor、IF或识别器的64维特征。训练与评估应使用一致的协议
元数据来源，输出metadata会记录使用的是提供值还是默认假设。

## 分片模型与可审计开销

`ip_tcp_fragmentation` 保留原动作键，但v2具体实现的是 **TCP分段**，没有同时声称
实现IPv4分片及其8字节偏移对齐。TCP分段中的每段均有自己的TCP/IP头部，参见
[RFC 9293 §3.1](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.1)。

默认头长为 `frame_header_bytes=54`（Ethernet14 + 无选项IPv4 20 + 无选项TCP20）。
Seq64本身不能还原真实头长，也不能区分原帧中的载荷和二层填充，因此默认将
`帧长-头长` 当作载荷。这是可配置假设。准确核算带选项或无载荷包时，应从原始
PCAP提供逐包 `header_lengths` 与 `payload_lengths`，或直接构造含这些字段的Packet。

令原帧长为L、头长H、载荷P，分为k段：

```text
P = p1 + p2 + ... + pk              # 每段载荷为正整数，完整保留原载荷区间
Li = max(60, H + pi)               # 每段帧长，不含FCS
新增头部 = (k - 1) * H
新增二层填充 = sum(Li - H - pi) - (L - H - P)
新增总字节 = sum(Li) - L
```

以太网最小数据字段为46字节，短IP报文需要二层填充；该填充不属于IP长度，参见
[RFC 894](https://www.rfc-editor.org/rfc/rfc894.html)。这里采用不含FCS的60字节
最小帧口径；不统计FCS、前导码或帧间隙。原抓包的54字节短帧可作为输入，生成的
新分段按最小帧补齐。零载荷或1字节载荷的包无法拆出两个非空段，此动作记一次
尝试但不产生分段或额外字节。

例如501字节、54字节头部的包具有447字节建模载荷，分3段为149/149/149字节：
每段203字节，总609字节，新增108字节。不会产生小数长度，也不会把整个帧长
均分后忽略新头部。

`FlowCost` 提供：

- `added_bytes`：完整变形事件列表与输入的帧字节差，**在截取前32事件之前核算**。
- `added_payload_bytes`：padding动作新增的载荷字节；分片不会重复计入原有载荷。
- `added_header_bytes`：分片新增的头部字节。
- `added_link_padding_bytes`：新旧二层填充差，可因填充被新增载荷占用而为负。
- `added_fragments`：新增段数。
- `added_delay`：策略主动添加的等待秒数之和。
- `action_counts`：仅包含三种动作。

每次动作都检查 `added_bytes = added_payload_bytes + added_header_bytes +
added_link_padding_bytes`。分片的载荷偏移也保存在Packet中，用于检查连续区间
和总字节数。padding仍受1514字节帧长上限约束，实际追加量可能小于请求值。

## 延迟模型和适配边界

`outbound_fifo_timestamps_v1` 在时间轴上推迟选中的上行包，后续上行包按FIFO
排队。下行包沿用原有观测时刻；两方向稳定排序后重新计算间隔。若下行先到，
输出首事件及方向也会改变。首事件gap始终为0，`Flow.start_time`保存共同时间
轴的起点，供连续动作使用，不进入特征。

例如 `[0,20,50]` 毫秒中的首包等待10毫秒、后续无需排队时，时刻为
`[10,20,50]`，间隔为 `[0,10,30]`，10毫秒等待仍记入开销。`added_delay`不是流
时长增量，也不是所有包因FIFO阻塞的等待之和。

当前仍是包对象/时间轴模型：不写出真实TCP/IP包，不生成校验和、ACK、重传或
服务器因果响应，也不模拟链路序列化时间。因此分片开销对声明的头长/载荷/帧长
模型是精确的；真实网络功能仍需协议栈验证。Oracle也不自动构成原论文效果上界。

IF奖励保留v1按Glimpse式(5)使用c(100)和树边数的选择，没有顺带改成另一种IF
评分。奖励阈值、总训练episode、gamma/rho等仍须公开。`training_iterations`
是每设备抽样流的episode数；`training_log.network_updates`报告实际更新次数。

v2将目标Actor权重、配置、延迟模型、分片模型及格式标记一起保存。动作从6维变为
4维，且部署网络和环境语义均发生变化，旧策略不能直接用于v2，应重新训练。
本目录的代码与测试不覆盖v1策略或历史实验结果。

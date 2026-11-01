"""Versioned per-input-prefix resource contract; independent of defense actions."""

# =============================================================================
# 中文阅读导图：所有方法共用的“物理预算协议”
# =============================================================================
# 这个文件不关心 FADE/IoTGAN/iPET 各自怎么变流量，只回答一个问题：
# 把一批防御输出折算成逐流的 added_bytes / wire_completion_delay_seconds /
# added_packets 之后，是否每条流都不超过同一组上限？
#
# PhysicalBudget 三个上限对应 630 B / 3.5 s / 6 个新增事件这一档配额：
#   max_added_bytes          单流新增字节（真实包 padding + dummy 字节）
#   max_wire_delay_seconds   单流“最后发送事件”被推迟的时长
#   max_added_packets        单流新增发送事件数（dummy + 分片）
#
# audit(costs) 的返回值里同时给出 maxima（实际最大值）与 cap_hit_rates
# （有多少比例的流贴着上限），后者用于说明“预算是真的被用满，还是远远没用完”。
# 违规会直接抛 ValueError 并列出前 10 条流的下标，绝不静默截断。
#
# 复杂语法：@dataclass(frozen=True) + __post_init__ 做构造期校验，
# 显式排除 bool（bool 是 int 子类，配置里 true 容易被误当成 1）。
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

PROTOCOL = "physical-prefix-budget-v1"


@dataclass(frozen=True)
class PhysicalBudget:
    max_added_bytes: int
    max_wire_delay_seconds: float
    max_added_packets: int

    def __post_init__(self):
        for key in ("max_added_bytes", "max_added_packets"):
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        value = self.max_wire_delay_seconds
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError("max_wire_delay_seconds must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)

    def audit(self, costs):
        limits = {"added_bytes": (self.max_added_bytes, 1e-7),
                  "wire_completion_delay_seconds": (self.max_wire_delay_seconds, 1e-9),
                  "added_packets": (self.max_added_packets, 1e-9)}
        maxima = {}
        hit_rates = {}
        shapes = set()
        for key, (limit, tolerance) in limits.items():
            values = np.asarray(costs[key], dtype=float)
            shapes.add(values.shape)
            if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < -tolerance):
                raise ValueError(f"Invalid physical cost: {key}")
            violating = np.flatnonzero(values > limit + tolerance)
            if len(violating):
                raise ValueError(f"{PROTOCOL}: {key} exceeds {limit}; flow indices {violating[:10].tolist()}")
            maxima[key] = float(values.max()) if len(values) else 0.0
            hit_rates[key] = float(np.mean(values >= limit - tolerance)) if len(values) else 0.0
        if len(shapes) != 1:
            raise ValueError("Physical cost arrays must have equal shapes")
        return dict(protocol=PROTOCOL, limits=self.to_dict(), maxima=maxima, cap_hit_rates=hit_rates,
                    violation_count=0, violation_rate=0.0)


def load_budget(path):
    return PhysicalBudget(**json.loads(Path(path).read_text(encoding="utf-8-sig")))

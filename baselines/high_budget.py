"""Explicit high-budget variants; original baseline source files remain unchanged.

Rate methods accept chronological native proposals under cumulative budgets.
IoTGAN uses one differentiable physical-domain projection in training and apply.
Glimpse is deliberately outside this module's budget constraints.
"""

# =============================================================================
# 中文阅读导图：把“原生基线”装进统一预算，但不改原生参数
# =============================================================================
# 目的：论文要比较的是“同样 630 B / 3.5 s / 6 事件”下的效果，而原生
# Random Rate / IoTGAN 并没有这套预算。这里的做法是“外部约束”而不是“重写方法”：
#
#   apply_traditional()
#       临时替换 traditional._render_application（进程级 monkeypatch）：
#       按时间顺序检查每个原生提议，dummy 必须整包放得下才接受，真实包的
#       padding/delay 按剩余额度裁剪；结束后 finally 恢复原函数。
#       这样原生速率、padding 参数、损失函数都不变，只是超预算的动作被拒绝/裁剪。
#
#   project_iotgan() / iotgan_budget()
#       IoTGAN 需要一个“可微”的投影：对选中的动作，把长度增量按比例缩放到
#       每流 L1 字节上限，把 gap 增量（秒）同样缩放；直通(straight-through)
#       只影响反向传播的梯度，不影响前向数值。预算只在 fit/apply 期间安装，
#       并通过“已经 patch 过就报错”防止并发/嵌套污染。
#
#   mark_policy() / apply_iotgan()
#       把预算写进 policy.training_metadata，应用前断言“训练与运行使用同一预算”，
#       避免拿旧策略配新预算直接评测。
#
# 注意：这些是显式的“约束型变体”，方法键在 adapters.ADAPTERS 里单独注册
# （*_shared_budget_v1），manifest 会记录 adapter_changes_algorithm=True。
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import numpy as np
import torch
from baselines import traditional, iotgan, iotgan_bidirectional


@dataclass(frozen=True)
class Budget:
    bytes: int = 630
    delay_seconds: float = 3.5
    dummy_packets: int = 6

    def __post_init__(self):
        for key in ('bytes', 'dummy_packets'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(key + ' must be a nonnegative integer')
        if not np.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError('delay_seconds must be finite and nonnegative')


HIGH = Budget()
_BASE_PROJECTION = iotgan._project_generated_raw


def apply_traditional(variant, lengths, gaps, directions, mask, *, seed, budget=HIGH, **kwargs):
    """Constrain native proposals, preserving the configured native parameters."""
    if variant not in ('random', 'constant_rate', 'classic_traffic_morphing'):
        raise ValueError('Unsupported budgeted traditional variant')
    original_renderer = traditional._render_application
    def render(source_l, source_g, source_d, source_m, proposed, old_stats, metadata, old_audits):
        stats = traditional._empty_stats(len(source_l))
        accepted, details = [], []
        for i, events in enumerate(proposed):
            remaining_b, remaining_s = budget.bytes, float(budget.delay_seconds)
            count, rejected, chosen = 0, 0, []
            for event in events:
                if event.is_dummy:
                    fits = (count < budget.dummy_packets and event.length <= remaining_b + 1e-9
                            and event.gap <= remaining_s + 1e-12)
                    if not fits:
                        rejected += 1
                        continue
                    b, s = float(event.length), float(event.gap)
                    chosen.append(event)
                    count += 1
                    for key, value in (('dummy_bytes', b), ('dummy_delay', s), ('dummy_count', 1),
                        ('outbound_dummy_bytes', b), ('outbound_dummy_delay', s),
                        ('outbound_dummy_count', 1), ('rate_slots_filled', 1)):
                        stats[key][i] += value
                else:
                    j = event.source_index
                    original_b, original_s = float(source_l[i,j]), float(source_g[i,j])
                    b = min(float(np.floor(max(0., event.length-original_b)+1e-9)), remaining_b)
                    requested_s = max(0., event.gap-original_s)
                    s = min(requested_s, remaining_s)
                    changed_gap = event.gap if requested_s <= remaining_s else original_s+s
                    assert event.direction == 1 or (b == 0 and s == 0)
                    chosen.append(traditional._TraceEvent(original_b+b, changed_gap,
                        event.direction, False, j))
                    for key, value in (('real_added_bytes', b), ('real_added_delay', s),
                        ('outbound_real_added_bytes', b), ('outbound_real_added_delay', s)):
                        stats[key][i] += value
                remaining_b = max(0., remaining_b-b)
                remaining_s = max(0., remaining_s-s)
            stats['dummy_gap_seconds'][i] = stats['dummy_delay'][i]
            accepted.append(chosen)
            details.append(dict(high_budget=asdict(budget), rejected_dummy_proposals=rejected,
                proposed_dummy_packets=int(old_stats['dummy_count'][i]),
                proposed_added_bytes=float(old_stats['real_added_bytes'][i]+old_stats['dummy_bytes'][i]),
                proposed_added_seconds=float(old_stats['real_added_delay'][i]+old_stats['dummy_delay'][i]),
                budget_enforcement='chronological proposals; real edits capped; complete dummy frames accepted only if all budgets fit',
                complete_event_trace=[dict(length_bytes=float(e.length), gap_seconds=float(e.gap),
                    direction=int(e.direction), is_dummy=bool(e.is_dummy), source_index=e.source_index) for e in chosen]))
        metadata = dict(metadata, high_budget=asdict(budget), adapter_changes_algorithm=True,
            budget_variant='high-budget-v1', dummy_frame_resizing=False,
            native_rate_and_padding_parameters_preserved=True)
        return original_renderer(source_l, source_g, source_d, source_m, accepted, stats, metadata, details)
    traditional._render_application = render
    try:
        return traditional.apply_baseline(variant, lengths, gaps, directions, mask,
            seed=seed, **kwargs)
    finally:
        traditional._render_application = original_renderer


def apply_rate(variant, lengths, gaps, directions, mask, *, seed, budget=HIGH):
    """Historical entry point; the common-budget adapter preserves native caps."""
    return apply_traditional(variant, lengths, gaps, directions, mask, seed=seed,
                             budget=budget, max_dummy_per_flow=budget.dummy_packets)


def project_iotgan(manipulated, selected_raw, selected_features, feature_mean, feature_scale,
                   selected_action_mask, *, straight_through=False, budget=HIGH):
    """Project selected physical increments to two per-flow L1 caps.

    No feature absent from the selected/action mask is changed. Length rounding
    uses the same straight-through convention as the original physical projector.
    Gap increments are measured in seconds, not in log or standardized units.
    """
    raw = _BASE_PROJECTION(manipulated, selected_raw, selected_features, feature_mean,
        feature_scale, selected_action_mask, straight_through=straight_through)
    original = torch.as_tensor(selected_raw, dtype=raw.dtype, device=raw.device)
    indices = torch.as_tensor(selected_features, dtype=torch.int64, device=raw.device)
    length = indices < iotgan.OBSERVED_EVENTS
    action = torch.as_tensor(selected_action_mask, dtype=torch.bool, device=raw.device)
    action = action & (indices != iotgan.OBSERVED_EVENTS)
    source_bytes = torch.round(original.abs()*iotgan.common.MAX_FRAME_LENGTH)
    byte_delta = torch.where(length & action, (raw.abs()*iotgan.common.MAX_FRAME_LENGTH-source_bytes).clamp(min=0), 0.)
    byte_scale = torch.clamp(budget.bytes/(byte_delta.sum(dim=1,keepdim=True)+1e-12), max=1.)
    byte_increment = byte_delta*byte_scale
    integer = torch.floor(byte_increment+1e-9)
    if straight_through:
        integer = byte_increment+(integer-byte_increment).detach()
    constrained_length = original.sign()*(source_bytes+integer)/iotgan.common.MAX_FRAME_LENGTH
    # Restrict exponentiation to log-gap columns; signed lengths are not times.
    old_seconds = torch.expm1(torch.where(length, 0., original))/1000.
    new_seconds = torch.expm1(torch.where(length, 0., raw))/1000.
    delta_seconds = torch.where((~length) & action, (new_seconds-old_seconds).clamp(min=0), 0.)
    time_scale = torch.clamp(budget.delay_seconds/(delta_seconds.sum(dim=1,keepdim=True)+1e-12), max=1.)
    constrained_gap = torch.log1p((old_seconds+delta_seconds*time_scale)*1000.)
    result = torch.where(length, constrained_length, constrained_gap)
    return torch.where(action, result, original)


@contextmanager
def iotgan_budget(budget=HIGH):
    """Install the same projection only within a fit/apply operation."""
    original = iotgan._project_generated_raw
    if original is not _BASE_PROJECTION:
        raise RuntimeError('IoTGAN projection is already patched; use isolated sequential workers')
    def projected(*args, **kwargs):
        return project_iotgan(*args, **kwargs, budget=budget)
    iotgan._project_generated_raw = projected
    try:
        yield
    finally:
        iotgan._project_generated_raw = original


def mark_policy(policy, budget=HIGH):
    policy.training_metadata['high_budget'] = dict(asdict(budget),
        variant='physical-flow-budget-v1', enforced_during_training=True,
        length_projection='scale positive physical increments then floor integer bytes',
        delay_projection='scale positive physical seconds increments',
        applies_to='selected enabled real features in both directions')
    return policy


def apply_iotgan(policy, lengths, gaps, directions, mask, labels=None, *, seed, budget=HIGH):
    record = policy.training_metadata.get('high_budget', {})
    assert record.get('enforced_during_training')
    assert all(record.get(k) == v for k,v in asdict(budget).items())
    with iotgan_budget(budget):
        app = iotgan_bidirectional.apply_policy(policy, lengths, gaps, directions, mask, labels, black_box_target=policy.black_box_target, base_seed=seed)
    stats = app[4]
    assert np.all(stats['real_added_bytes'] <= budget.bytes+1e-7)
    assert np.all(stats['real_added_delay'] <= budget.delay_seconds+1e-9)
    assert np.all(stats['dummy_count'] == 0)
    app[5].update(high_budget=record, adapter_changes_algorithm=True,
        budget_training_constraints_match_runtime=True)
    return app

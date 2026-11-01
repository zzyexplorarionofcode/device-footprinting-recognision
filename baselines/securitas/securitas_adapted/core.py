"""Immutable full-event simulator. No labels, models or observation truncation."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import math
from .config import validate_config
from .icmp import ICMPEnvironment, PreparedFlow


@dataclass(frozen=True)
class Header:
    provenance: str = 'unknown'
    ip_version: int = 0
    l2_bytes: int = 14
    ip_header_bytes: int = 20
    payload_bytes: int | None = None
    dont_fragment: bool = False
    already_fragmented: bool = False
    supported: bool = False
    reason: str = 'protocol_unknown'


@dataclass(frozen=True)
class SourceEvent:
    source_index: int
    original_order: int
    timestamp: float
    gap: float
    length: int
    direction: int
    header: Header = Header()


@dataclass(frozen=True)
class Event:
    role: str
    source_index: int | None
    parent_source_index: int
    original_order: int
    timestamp: float
    gap: float
    length: int
    direction: int
    header: Header
    fragment_order: int | None = None
    payload_offset: int | None = None
    more_fragments: bool = False
    ttl: int | None = None


@dataclass(frozen=True)
class ICMPEvent(Event):
    ip_protocol: int = 1
    icmp_type: int = 11
    icmp_code: int = 0
    icmp_header_bytes: int = 8
    quoted_ip_header_bytes: int = 20
    quoted_payload_bytes: int = 8
    response_to_role: str = 'dummy'


@dataclass(frozen=True)
class Action:
    payload_bytes: int
    operation: str


@dataclass(frozen=True)
class Strategy:
    actions: tuple[Action, ...]
    candidate_id: int = 0


@dataclass(frozen=True)
class TransformResult:
    events: tuple[Event, ...]
    cost: dict
    decisions: tuple[dict, ...]

    def to_dict(self):
        return asdict(self)


def validate_strategy(strategy, config):
    if len(strategy.actions) != config['policy_length']:
        raise ValueError('Strategy length does not match saved configuration')
    for a in strategy.actions:
        if type(a.payload_bytes) is not int or a.payload_bytes not in config['action_payload_bytes']:
            raise ValueError('Strategy action outside configured payload alphabet')
        if a.operation not in ('none','split','dummy') or ((a.payload_bytes == 0) != (a.operation == 'none')):
            raise ValueError('Zero action must be none; nonzero must be split or dummy')


def _real(s, t):
    return Event('real', s.source_index, s.source_index, s.original_order, t, s.gap,
                 s.length, s.direction, s.header)


def _split(s, a, t, config):
    h, p = s.header, config['protocol']
    if not h.supported or h.ip_version != 4:
        return None, h.reason or 'unsupported_protocol'
    if h.ip_header_bytes != 20:
        return None, 'ipv4_options_not_supported'
    if h.dont_fragment:
        return None, 'ipv4_df_set'
    if h.already_fragmented:
        return None, 'source_already_fragmented'
    q = h.payload_bytes
    if q is None or q <= a.payload_bytes:
        return None, 'payload_too_short'
    first, second = q - a.payload_bytes, a.payload_bytes
    if first % 8:
        return None, 'nonfinal_payload_not_8byte_aligned'
    overhead = h.l2_bytes + h.ip_header_bytes
    lengths = (max(p['min_frame_bytes'], overhead + first), max(p['min_frame_bytes'], overhead + second))
    if any(x > p['max_frame_bytes'] for x in lengths):
        return None, 'fragment_frame_bound'
    if sum(lengths) < s.length:
        return None, 'source_padding_cannot_be_reclaimed'
    events = tuple(Event('fragment', s.source_index, s.source_index, s.original_order,
                         t + i * config['delta_seconds'], 0., int(lengths[i]), s.direction,
                         replace(h, payload_bytes=size, already_fragmented=True), i,
                         0 if i == 0 else first, i == 0)
                   for i, size in enumerate((first,second)))
    return events, None


def _dummy(s, a, t, config):
    p = config['protocol']
    length = max(p['min_frame_bytes'], 14 + 20 + a.payload_bytes)
    if length > p['max_frame_bytes'] or a.payload_bytes > 65515:
        return None, 'dummy_frame_bound'
    h = Header('synthetic_ipv4_assumption', 4, 14, 20, a.payload_bytes, False, False, True, '')
    return (_real(s,t), Event('dummy', None, s.source_index, s.original_order,
             t + config['delta_seconds'], 0., int(length), 1, h, ttl=p['dummy_ttl'])), None


def apply_flow(source, strategy: Strategy, config, *, simulation_seed=None) -> TransformResult:
    """Apply one fixed universal table; labels are deliberately absent."""
    validate_config(config)
    validate_strategy(strategy, config)
    source = source if isinstance(source, PreparedFlow) else tuple(source)
    previous = 0.
    for i,s in enumerate(source):
        if s.source_index != i or s.original_order != i or s.direction not in (-1,1):
            raise ValueError('Source index/order must be contiguous; direction is device-relative +/-1')
        if type(s.length) is not int or s.length <= 0 or not math.isfinite(s.timestamp) or not math.isfinite(s.gap) or s.gap < 0:
            raise ValueError('Invalid source length/time')
        if s.timestamp < previous or not math.isclose(s.timestamp - previous,s.gap,rel_tol=1e-8,abs_tol=1e-9):
            raise ValueError('Source timestamp/gap mismatch')
        h = s.header
        if h.supported and (h.payload_bytes is None or h.payload_bytes < 0 or h.l2_bytes < 14 or h.ip_header_bytes < 20 or h.l2_bytes+h.ip_header_bytes+h.payload_bytes > s.length):
            raise ValueError("Header/payload information exceeds the source frame")
        previous = s.timestamp
    environment = ICMPEnvironment(source, config, simulation_seed) if config['simulate_icmp'] else None
    source_duration = source[-1].timestamp if source else 0.
    events, decisions = [], []
    added_bytes = dummy_count = splits = extra_count = uplink = 0
    offset = 0.
    budget, delta = config['budget'], config['delta_seconds']
    for s in source:
        t = s.timestamp + offset
        if s.direction != 1:
            events.append(_real(s,t))
            continue
        position = uplink
        uplink += 1
        if position >= len(strategy.actions):
            events.append(_real(s,t))
            decisions.append(dict(source_index=s.source_index, policy_index=position, status='skipped', reason='policy_exhausted'))
            continue
        a = strategy.actions[position]
        d = dict(source_index=s.source_index, policy_index=position,
                 requested={'payload_bytes':a.payload_bytes, 'operation':a.operation})
        if a.payload_bytes == 0:
            events.append(_real(s,t))
            decisions.append(dict(d,status='skipped',reason='zero_action'))
            continue
        proposal, reason = _split(s,a,t,config) if a.operation == 'split' else _dummy(s,a,t,config)
        realized = a.operation
        if proposal is None and a.operation == 'split':
            d['split_infeasible_reason'] = reason
            proposal, reason = _dummy(s,a,t,config)
            realized = 'dummy'
        if proposal is None:
            events.append(_real(s,t))
            decisions.append(dict(d,status='infeasible',reason=reason))
            continue
        inc = sum(e.length for e in proposal) - s.length
        new_dummy = int(realized == 'dummy')
        # Only locally inserted dummy/second fragment causes a source-stream offset.
        # Network replies arrive on a merged timeline; they do not stall later sources.
        next_offset = (extra_count + 1) * delta
        proposal = (proposal[0],replace(proposal[1],timestamp=s.timestamp+next_offset))
        potential = environment.potential(s, proposal[1]) if environment and new_dummy else None
        if potential and potential['length_bytes'] > config['protocol']['max_frame_bytes']:
            events.append(_real(s,t))
            decisions.append(dict(d,status='infeasible',reason='icmp_frame_bound'))
            continue
        reserve_bytes = (environment.reserved_bytes if environment else 0) + (potential['length_bytes'] if potential else 0)
        reserve_count = (environment.reserved_count if environment else 0) + int(potential is not None)
        failures = []
        if added_bytes + inc + reserve_bytes > budget['bytes']:
            failures.append('byte_budget')
        # Reserve the bounded tail before sending. Never gate admission on a lucky draw.
        reserved_tail = max(environment.tail_latest if environment else 0., potential['tail_latest'] if potential else 0.)
        span_bound = max(next_offset, reserved_tail-source_duration)
        if span_bound > budget['delay_seconds']:
            failures.append('delay_budget')
        prospective_packets = (extra_count + 1 if config.get('shared_budget') else dummy_count + new_dummy)
        if prospective_packets + reserve_count > budget['dummy_packets']:
            failures.append('dummy_budget')
        if environment:
            d['prospective_budget'] = dict(added_bytes=added_bytes+inc+reserve_bytes,
                                          dummy_count=dummy_count+new_dummy+reserve_count,
                                          added_delay_seconds=span_bound,
                                          potential_icmp=potential is not None)
        if failures:
            events.append(_real(s,t))
            decisions.append(dict(d,status='cancelled',reason='+'.join(failures),proposed_added_bytes=inc))
            continue
        if delta > 0 and proposal[1].timestamp <= proposal[0].timestamp:
            raise ValueError('Time magnitude cannot represent delta; use relative timestamps with adequate precision')
        events.extend(proposal)
        if environment:
            environment.reserve(potential)
        added_bytes += inc
        dummy_count += new_dummy
        splits += int(realized == 'split')
        extra_count += 1
        offset = next_offset
        decisions.append(dict(d,status='executed',operation=realized,added_bytes=inc,added_delay_seconds=delta))
    replies = []
    reserved_bytes = added_bytes + (environment.reserved_bytes if environment else 0)
    reserved_dummies = dummy_count + (environment.reserved_count if environment else 0)
    if environment:
        events, replies = environment.merge(events, decisions)
        added_bytes += sum(e.length for e in replies)
    # A true no-op uses the original gaps exactly, avoiding cumsum/diff roundoff in features.
    if extra_count:
        previous = 0.
        for i,e in enumerate(events):
            gap = e.timestamp - previous
            if gap < 0:
                raise AssertionError('Non-monotonic event timeline')
            events[i] = replace(e,gap=gap)
            previous = e.timestamp
    source_bytes = sum(s.length for s in source)
    output_bytes = sum(e.length for e in events)
    source_duration = source[-1].timestamp if source else 0.
    output_duration = events[-1].timestamp if events else 0.
    measured_delay = output_duration - source_duration
    reply_end = max((e.timestamp for e in replies), default=source_duration)
    added_delay = max(offset, reply_end-source_duration) if environment else offset
    if output_bytes - source_bytes != added_bytes or not math.isclose(measured_delay,added_delay,rel_tol=1e-7,abs_tol=1e-9):
        raise AssertionError('Full-event cost and action accounting disagree')
    real_events = tuple(e for e in events if e.role in ('real','fragment'))
    real_completion = max((e.timestamp for e in real_events), default=source_duration)
    wire_completion = max((e.timestamp for e in events), default=source_duration)
    real_completion_delay = max(0., real_completion-source_duration)
    wire_completion_delay = max(0., wire_completion-source_duration)
    cover_tail = max(0., wire_completion-real_completion)
    cost = dict(source_bytes=source_bytes,output_bytes=output_bytes,added_bytes=added_bytes,
                source_duration_seconds=source_duration,output_duration_seconds=output_duration,
                added_delay_seconds=added_delay,timeline_increment_seconds=measured_delay,
                real_completion_delay_seconds=real_completion_delay,
                wire_completion_delay_seconds=wire_completion_delay,
                cover_tail_seconds=cover_tail,
                added_byte_ratio=added_bytes/source_bytes if source_bytes else 0.,
                added_delay_ratio=added_delay/source_duration if source_duration else None,
                dummy_count=dummy_count,split_count=splits,additional_real_segments=splits,
                added_fragments=splits,added_events=extra_count,complete_events=len(events))
    if environment:
        icmp_bytes = sum(e.length for e in replies)
        cost.update(dummy_count=dummy_count+len(replies), uplink_dummy_count=dummy_count,
                    icmp_reply_count=len(replies), icmp_added_bytes=icmp_bytes,
                    uplink_dummy_bytes=sum(e.length for e in events if e.role=='dummy'),
                    added_events=extra_count+len(replies), local_added_events=extra_count,
                    local_timeline_shift_seconds=offset,
                    icmp_tail_extension_seconds=max(0., added_delay-offset),
                    budget_reserved_added_bytes=reserved_bytes,
                    budget_reserved_dummy_count=reserved_dummies,
                    budget_reserved_added_delay_seconds=max(offset,environment.tail_latest-source_duration),
                    icmp_simulation_seed=environment.seed, icmp_flow_key=environment.flow_key)
        if cost['added_bytes'] > budget['bytes'] or cost['dummy_count'] > budget['dummy_packets'] or added_delay > budget['delay_seconds']:
            raise AssertionError('ICMP-inclusive per-flow budget exceeded')
    if config.get('shared_budget') is not None:
        from experiment_support.budget import PhysicalBudget
        cost['added_packets'] = len(events) - len(source)
        PhysicalBudget(**config['shared_budget']).audit(dict(
            added_bytes=[cost['added_bytes']], added_packets=[cost['added_packets']],
            wire_completion_delay_seconds=[cost['wire_completion_delay_seconds']]))
    return TransformResult(tuple(events),cost,tuple(decisions))


def apply_batch(flows, strategy, config, *, simulation_seed=None):
    return [apply_flow(flow,strategy,config,simulation_seed=simulation_seed) for flow in flows]

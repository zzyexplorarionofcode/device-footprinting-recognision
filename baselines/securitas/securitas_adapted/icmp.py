"""Optional, reproducible ICMP environment; never learns or chooses policy actions.

Paper Sec. 4.2: reply probability .48; arrival uniformly between the dummy
and the next real inbound packet. A configurable fixed tail delay closes the
case with no later inbound event. That tail delay is NOT a measurement.
"""
from dataclasses import asdict
import hashlib
import json
import math


def _source_metadata(source):
    encoded = json.dumps([asdict(s) for s in source], sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    flow_key = hashlib.sha256(encoded).hexdigest()
    next_inbound, next_index = {}, None
    for s in reversed(source):
        next_inbound[s.source_index] = next_index
        if s.direction == -1:
            next_index = s.source_index
    return flow_key, next_inbound


class PreparedFlow(tuple):
    """Training-local cache for immutable SourceEvents; no sampled outcomes cached."""

    def __new__(cls, source):
        obj = super().__new__(cls, source)
        obj.icmp_metadata = _source_metadata(obj)
        return obj


class ICMPEnvironment:
    def __init__(self, source, config, simulation_seed=None):
        self.config = config
        self.settings = config['icmp']
        self.seed = self.settings['simulation_seed'] if simulation_seed is None else simulation_seed
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('simulation_seed must be a nonnegative integer')
        # Labels, candidate IDs, batch position/order and action choices are absent.
        self.flow_key, self.next_inbound = (source.icmp_metadata
            if isinstance(source, PreparedFlow) else _source_metadata(source))
        self.pending = []
        self.reserved_bytes = self.reserved_count = 0
        self.tail_latest = 0.0

    def potential(self, s, dummy):
        """Describe a POSSIBLE reply, before drawing any environmental outcome."""
        if self.settings['response_probability'] == 0:
            return None
        quoted = min(8, dummy.header.payload_bytes)
        # Ethernet + outer IPv4 + ICMP type/code/checksum/unused + quoted IP + data.
        payload = 8 + dummy.header.ip_header_bytes + quoted
        length = max(self.config['protocol']['min_frame_bytes'], 14 + 20 + payload)
        next_index = self.next_inbound[s.source_index]
        tail_latest = (dummy.timestamp + self.settings['tail_delay_seconds']
                       if next_index is None else 0.0)
        if not math.isfinite(tail_latest) or (next_index is None and self.settings['tail_delay_seconds'] > 0 and tail_latest <= dummy.timestamp):
            raise ValueError('ICMP tail delay cannot be represented at this timestamp magnitude')
        return dict(parent_source_index=s.source_index, dummy_timestamp=dummy.timestamp,
                    next_inbound_source_index=next_index, payload_bytes=payload,
                    quoted_ip_header_bytes=dummy.header.ip_header_bytes,
                    quoted_payload_bytes=quoted, length_bytes=length,
                    tail_latest=tail_latest)

    def reserve(self, potential):
        if potential is not None:
            self.pending.append(potential)
            self.reserved_bytes += potential['length_bytes']
            self.reserved_count += 1
            self.tail_latest = max(self.tail_latest, potential['tail_latest'])

    def _uniform(self, parent, channel):
        key = f'securitas-icmp-v1:{self.seed}:{self.flow_key}:{parent}:{channel}'.encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') >> 11
        return value / (2**53)

    def replies(self, source_events):
        # Import at call time: core owns event types; there is no import cycle.
        from .core import Header, ICMPEvent
        inbound = {e.source_index: e for e in source_events
                   if e.role == 'real' and e.direction == -1}
        replies, outcomes = [], {}
        for p in self.pending:
            parent = p['parent_source_index']
            u = self._uniform(parent, 'response')
            q = self._uniform(parent, 'time')
            present = u < self.settings['response_probability']
            nxt = p['next_inbound_source_index']
            start = p['dummy_timestamp']
            end = inbound[nxt].timestamp if nxt is not None else p['tail_latest']
            if end < start:
                raise AssertionError('ICMP return interval precedes its triggering dummy')
            timestamp = start + (end-start)*q if nxt is not None else end
            outcomes[parent] = dict(status='generated' if present else 'not_returned',
                                    probability=self.settings['response_probability'],
                                    response_quantile=u, time_quantile=q,
                                    timing='paper_next_inbound' if nxt is not None else 'tail_fixed_delay_assumption',
                                    interval_seconds=[start, end],
                                    next_inbound_source_index=nxt,
                                    arrival_timestamp=timestamp if present else None,
                                    reserved_bytes=p['length_bytes'])
            if not present:
                continue
            header = Header('synthetic_icmp_time_exceeded', 4, 14, 20,
                            p['payload_bytes'], False, False, True, '')
            replies.append(ICMPEvent('icmp', None, parent, parent, timestamp, 0.0,
                                     p['length_bytes'], -1, header,
                                     quoted_ip_header_bytes=p['quoted_ip_header_bytes'],
                                     quoted_payload_bytes=p['quoted_payload_bytes']))
        return replies, outcomes

    def merge(self, events, decisions):
        replies, outcomes = self.replies(events)
        for d in decisions:
            if d.get('status') == 'executed' and d.get('operation') == 'dummy':
                d['icmp'] = outcomes.get(d['source_index'], {'status':'probability_zero'})
        # Existing events win exact timestamp ties. A response never precedes its dummy.
        return sorted([*events, *replies], key=lambda e:e.timestamp), replies

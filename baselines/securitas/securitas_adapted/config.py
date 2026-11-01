"""Resolve a portable snapshot from the existing Method33 JSON configuration."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
from .common import PROJECT_ROOT, read_json, sha256


def validate_config(c):
    if c.get('shared_budget') is not None:
        from experiment_support.budget import PhysicalBudget
        shared = PhysicalBudget(**c['shared_budget'])
        expected = dict(bytes=shared.max_added_bytes, delay_seconds=shared.max_wire_delay_seconds,
                        dummy_packets=shared.max_added_packets)
        if c['budget'] != expected:
            raise ValueError('Securitas internal limits differ from shared budget')
    if c.get('schema') != 'securitas-adapted-config-v1':
        raise ValueError('Unsupported configuration schema')
    if type(c['simulate_icmp']) is not bool:
        raise ValueError('simulate_icmp must be boolean')
    if c['simulate_icmp']:
        icmp = c.get('icmp', {})
        expected = {'response_probability', 'timing_model', 'tail_delay_seconds', 'simulation_seed'}
        if set(icmp) != expected:
            raise ValueError('Enabled ICMP requires exactly: ' + ', '.join(sorted(expected)))
        probability = icmp['response_probability']
        if isinstance(probability, bool) or not isinstance(probability, (int,float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('ICMP response_probability must be in [0,1]')
        if icmp['timing_model'] != 'uniform_before_next_inbound':
            raise ValueError('Unsupported ICMP timing_model')
        delay = icmp['tail_delay_seconds']
        if isinstance(delay, bool) or not isinstance(delay, (int,float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError('ICMP tail_delay_seconds must be finite and nonnegative')
        if type(icmp['simulation_seed']) is not int or icmp['simulation_seed'] < 0:
            raise ValueError('ICMP simulation_seed must be a nonnegative integer')
    for k in ('window_length', 'policy_length'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(k + ' must be a positive integer')
    a = c['action_payload_bytes']
    if not a or a[0] != 0 or len(a) != len(set(a)) or any(type(x) is not int or x < 0 for x in a):
        raise ValueError('Actions must be distinct nonnegative integer IP payload bytes, starting with 0')
    for k, v in c['budget'].items():
        if not math.isfinite(v) or v < 0:
            raise ValueError('Invalid budget: ' + k)
    if type(c['budget']['dummy_packets']) is not int:
        raise ValueError('dummy_packets must be an integer')
    if not math.isfinite(c['delta_seconds']) or c['delta_seconds'] < 0:
        raise ValueError('delta_seconds must be finite and nonnegative')
    if not 0 <= c['split_probability'] <= 1:
        raise ValueError('split_probability must be in [0,1]')
    if c['protocol']['header_mode'] not in ('pcap', 'assume_ethernet_ipv4'):
        raise ValueError('Explicit pcap or assume_ethernet_ipv4 header mode required')
    if c['protocol']['min_frame_bytes'] < 60 or c['protocol']['max_frame_bytes'] < c['protocol']['min_frame_bytes']:
        raise ValueError('Invalid Ethernet frame bounds (excluding FCS)')
    if len(c['loss_weights']) != 3 or any(not math.isfinite(x) or x < 0 for x in c['loss_weights']):
        raise ValueError('Three finite nonnegative cost weights required')
    for part in ('proxy', 'policy'):
        for k in ('epochs', 'batch_size'):
            if type(c[part][k]) is not int or c[part][k] < 1:
                raise ValueError(part + '.' + k + ' must be positive')
    if c['policy']['strategies_per_batch'] < 1 or c['policy']['validation_candidates'] < 1:
        raise ValueError('At least one training strategy and one validation candidate required')
    if c['data'].get('inner_split_mode','ratio') not in ('ratio','method32-captures'):
        raise ValueError('Unknown inner split mode')
    if c['proxy']['solver'] != 'adam':
        raise ValueError('The incremental proxy implementation supports Adam only')
    return c


def load_config(path):
    raw = read_json(path)
    if raw.get('schema') == 'securitas-adapted-config-v1':
        return validate_config(raw)
    ref = Path(raw['method33_config'])
    if not ref.is_absolute():
        ref = PROJECT_ROOT / ref
    m = read_json(ref)
    method = m.get('method33', m.get('method32'))
    if method is None:
        raise ValueError('No Method33-compatible policy configuration')
    ds = m['dataset']
    c = dict(schema='securitas-adapted-config-v1', name='Securitas-adapted',
             dataset_status=raw.get('dataset_status','legacy-data exploratory; not a new UNSW protocol'),
             data_protocol=raw.get('data_protocol'),
             window_length=int(raw.get('window_length', ds['max_packets'])),
             budget=deepcopy(method['hard_budget']), simulate_icmp=False,
             icmp={'response_probability':0.48, 'timing_model':'uniform_before_next_inbound',
                   'tail_delay_seconds':0.01, 'simulation_seed':20260913},
             action_payload_bytes=list(range(0, 89, 8)), split_probability=0.7,
             delta_seconds=1e-6, loss_weights=[0.6, 0.2, 0.2], epsilon=1e-12,
             seed=20260910, device='cpu',
             feature={'max_frame_length':ds['max_frame_length'], 'max_gap_seconds':ds['max_feature_gap_seconds']},
             protocol={'header_mode':'pcap', 'min_frame_bytes':method['limits']['min_dummy_frame_length'],
                       'max_frame_bytes':method['limits']['max_frame_length'], 'dummy_ttl':3,
                       'assumption':'Ethernet II without FCS/tunnel; dummy IPv4 without options; minimum frame padding counted'},
             proxy={'hidden_layer_sizes':[64,32], 'epochs':100, 'batch_size':256,
                    'learning_rate':0.001, 'solver':'adam', 'alpha':0.0001, 'balanced':True},
             policy={'hidden_size':32, 'epochs':500, 'batch_size':256, 'learning_rate':0.001,
                     'strategies_per_batch':4, 'validation_candidates':16, 'mean_baseline':True},
             data={'dataset':raw['dataset'], 'expected_classes':len(ds['devices']), 'class_names':ds['devices'],
                   'outer_train_ratio':m['evaluation']['capture_train_ratio'], 'outer_seed':m['evaluation']['split_seed'],
                   'inner_train_ratio':0.8, 'inner_seed':m['selection']['inner_split_seed'], 'inner_split_mode':'ratio'},
             provenance={'method33_config':str(ref), 'method33_config_sha256':sha256(ref),
                         'budget_source':'method33.hard_budget',
                         'delta_source':'configurable simulation interval; Method33 has only a Trigger-specific 4ms floor'})
    c['policy_length'] = c['window_length']
    if raw.get('shared_budget_file'):
        from experiment_support.budget import load_budget
        budget_path = Path(raw['shared_budget_file'])
        if not budget_path.is_absolute():
            budget_path = PROJECT_ROOT / budget_path
        shared = load_budget(budget_path)
        c['shared_budget'] = shared.to_dict()
        c['budget'] = dict(bytes=shared.max_added_bytes, delay_seconds=shared.max_wire_delay_seconds,
                           dummy_packets=shared.max_added_packets)
        c['provenance'].update(budget_source=str(budget_path), budget_sha256=sha256(budget_path),
                               packet_budget_semantics='all added events including fragments and ICMP replies')
    allowed = set(c) - {'schema','name','budget','feature','provenance','shared_budget'}
    for k, v in raw.items():
        if k in ('method33_config','dataset','shared_budget_file'):
            continue
        if k not in allowed:
            raise ValueError('Unknown/locked setting: ' + k)
        if isinstance(v, dict) and isinstance(c[k], dict):
            if set(v) - set(c[k]):
                raise ValueError('Unknown ' + k + ' setting')
            c[k].update(v)
        else:
            c[k] = v
    return validate_config(c)

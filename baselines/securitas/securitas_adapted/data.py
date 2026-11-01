"""Existing NPZ/capture split and optional real header recovery, without relabeling flows."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
import json
import struct
import numpy as np
from .common import dataset_pcap, PROJECT_ROOT, sha256, write_json, read_json
from .core import Header, SourceEvent
from data_pipeline.unsw import UNSW_DATA_PROTOCOL
from data_pipeline.cic import PROTOCOL as CIC_DATA_PROTOCOL
SUPPORTED_PROTOCOLS = (UNSW_DATA_PROTOCOL, CIC_DATA_PROTOCOL)


def parse_header(frame: bytes) -> Header:
    """IPv4 without IP options; VLAN header length is read, not guessed."""
    if len(frame) < 14:
        return Header(reason='short_ethernet_header')
    ether_type, l2 = struct.unpack('!H',frame[12:14])[0],14
    while ether_type in (0x8100,0x88a8,0x9100):
        if len(frame) < l2 + 4:
            return Header(reason='short_vlan_header')
        ether_type = struct.unpack('!H',frame[l2+2:l2+4])[0]
        l2 += 4
    if ether_type != 0x0800 or len(frame) < l2 + 20:
        return Header(provenance='pcap',reason='not_complete_ipv4')
    v,ihl = frame[l2] >> 4,(frame[l2] & 15)*4
    total,flags = struct.unpack('!H',frame[l2+2:l2+4])[0],struct.unpack('!H',frame[l2+6:l2+8])[0]
    reason = ''
    if v != 4 or ihl < 20 or total < ihl or total + l2 > len(frame):
        reason = 'malformed_or_truncated_ipv4'
    elif ihl != 20:
        reason = 'ipv4_options_not_supported'
    elif len(frame) != max(60,l2+total) and len(frame) != l2+total:
        reason = 'unmodeled_trailer_or_fcs'
    return Header('pcap',v,l2,ihl,max(0,total-ihl),bool(flags&0x4000),bool(flags&0x3fff),not reason,reason)


def assumed_header(length):
    # Explicit assumption mode: not a measurement. Known incompatible metadata is never overwritten.
    if length < 34:
        return Header(provenance='explicit_assumption',reason='too_short_for_assumed_headers')
    return Header('explicit_assumption',4,14,20,length-34,False,False,True,'')


def source_flow(lengths,gaps,directions,mask,*,headers=None,header_mode='pcap'):
    arrays = [np.asarray(x) for x in (lengths,gaps,directions,mask)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('One flow requires four equal 1D arrays')
    n = int(np.count_nonzero(arrays[3]))
    if not np.array_equal(arrays[3],np.r_[np.ones(n),np.zeros(len(arrays[3])-n)]):
        raise ValueError('Mask must be a contiguous valid prefix')
    if any(np.any(a[n:] != 0) for a in arrays[:3]):
        raise ValueError('Masked length/gap/direction must be zero')
    if headers is not None and len(headers) != n:
        raise ValueError('One header record per source event required')
    out,t = [],0.
    for i in range(n):
        length,gap,direction = float(arrays[0][i]),float(arrays[1][i]),int(arrays[2][i])
        if not np.isfinite(length) or not length.is_integer() or length <= 0 or not np.isfinite(gap) or gap < 0 or direction not in (-1,1) or arrays[2][i] != direction:
            raise ValueError('Invalid source event')
        h = headers[i] if headers is not None else (assumed_header(int(length)) if header_mode == 'assume_ethernet_ipv4' else Header())
        t += gap
        out.append(SourceEvent(i,i,t,gap,int(length),direction,h))
    return tuple(out)


def prepare_data(config):
    """Only schema/split metadata are read for outer-test; its flows never enter training/validation."""
    path = Path(config['data']['dataset'])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = dataset_pcap.load_dataset(path)
    declared_protocol = config.get('data_protocol')
    if declared_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f'Unsupported data_protocol: {declared_protocol!r}')
    actual_protocol = data.get('data_protocol')
    if actual_protocol is None or np.asarray(actual_protocol).shape != ():
        raise ValueError('Dataset must declare scalar data_protocol metadata')
    actual_protocol = str(np.asarray(actual_protocol).item())
    if actual_protocol != declared_protocol:
        raise ValueError(f'Dataset/config data_protocol mismatch: {actual_protocol!r} vs {declared_protocol!r}')
    if config['data'].get('inner_split_mode') == 'method32-captures' and actual_protocol != CIC_DATA_PROTOCOL:
        raise ValueError('method32-captures is reserved for the CIC adapter')
    y,groups = np.asarray(data['labels'],int),np.asarray(data['groups']).astype(str)
    classes = np.unique(y)
    if not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError('Use the project dataset class encoding 0..K-1')
    expected = config['data'].get('expected_classes')
    if expected is not None and len(classes) != expected:
        raise ValueError('Class count does not match selected dataset configuration')
    if list(data['device_names'].astype(str)) != config['data']['class_names']:
        raise ValueError('Dataset device class order differs from configuration')
    from data_pipeline.partitions import frozen_partition
    frozen = frozen_partition(data, train_ratio=config['data']['outer_train_ratio'], seed=config['data']['outer_seed'])
    if frozen is not None:
        masks, metadata = frozen
        if (config['data']['inner_seed'] != metadata['inner_seed'] or
                not np.isclose(config['data']['inner_train_ratio'], metadata['inner_train_ratio'])):
            raise ValueError('configured validation split differs from frozen dataset')
        train_ids, val_ids = np.flatnonzero(masks['train']), np.flatnonzero(masks['validation'])
        outer_ids = np.flatnonzero(masks['train'] | masks['validation'])
        test = masks['test']
    else:
        outer,test = dataset_pcap.split_by_capture(y,groups,train_ratio=config['data']['outer_train_ratio'],seed=config['data']['outer_seed'])
        outer_ids = np.flatnonzero(outer)
        if config['data'].get('inner_split_mode') == 'method32-captures':
            from methods.method33_tools import load_config
            from methods.shared_tools import build_policy_selection_split
            project_path = Path(config['provenance']['method33_config'])
            if sha256(project_path) != config['provenance']['method33_config_sha256']:
                raise ValueError('Referenced selection configuration changed')
            project = load_config(project_path)
            if (project.selection.inner_split_seed != config['data']['inner_seed'] or
                    project.evaluation.split_seed != config['data']['outer_seed'] or
                    project.evaluation.capture_train_ratio != config['data']['outer_train_ratio']):
                raise ValueError('Selection configuration does not match split settings')
            from types import SimpleNamespace
            # The shared splitter reads dataset/evaluation/selection and class_count only.
            selection_project = SimpleNamespace(dataset=project.dataset, method32=project.method33,
                                                evaluation=project.evaluation, selection=project.selection)
            selected = build_policy_selection_split(data, selection_project)
            train_ids,val_ids = selected.selection_indices,selected.validation_indices
            np.testing.assert_array_equal(selected.outer_test_indices,np.flatnonzero(test))
        else:
            inner,val = dataset_pcap.split_by_capture(y[outer],groups[outer],train_ratio=config['data']['inner_train_ratio'],seed=config['data']['inner_seed'])
            train_ids,val_ids = outer_ids[inner],outer_ids[val]
    sets = [set(groups[ix]) for ix in (train_ids,val_ids,np.flatnonzero(test))]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError('Capture split leakage')
    split = {'dataset':str(path),'dataset_sha256':sha256(path),'class_names':data['device_names'].tolist(),
             'outer_train_indices':outer_ids.tolist(),'outer_test_indices':np.flatnonzero(test).tolist(),
             'train_indices':train_ids.tolist(),'validation_indices':val_ids.tolist(),
             'train_groups':sorted(sets[0]),'validation_groups':sorted(sets[1]),'test_groups':sorted(sets[2]),
             'outer_seed':config['data']['outer_seed'],'inner_seed':config['data']['inner_seed'],
             'outer_train_ratio':config['data']['outer_train_ratio'],'inner_train_ratio':config['data']['inner_train_ratio']}
    if frozen is None and config['data'].get('inner_split_mode') == 'method32-captures':
        split.update(inner_split_mode='method32-captures', inner_train_ratio=None, selection_audit=selected.audit)
    return data,train_ids,val_ids,split


def recover_headers(data, indices, output_path=None, dataset_sha=None, capture_root=None, progress=None):
    """Read only requested packet IDs via the project's PCAP/PCAPNG iterator. No payload is saved."""
    from data_pipeline.unsw import read_capture
    if 'frame_ids' not in data or 'timestamps' not in data:
        raise ValueError('PCAP recovery requires frame_ids and timestamps')
    protocol = str(np.asarray(data.get('data_protocol', '')).item())
    if protocol == CIC_DATA_PROTOCOL:
        manifest = json.loads(str(data['extraction_manifest_json'].item()))
        if manifest.get('partial') or manifest['protocol']['id'] != protocol:
            raise ValueError('CIC headers require a complete, matching extraction manifest')
        captures = {}
        for entry in manifest['captures']:
            path = Path(entry['input']['path'])
            key = PureWindowsPath(str(path)).name
            if key in captures:
                raise ValueError(f'Ambiguous CIC capture basename: {key}')
            captures[key] = path
        row_captures = np.asarray(data['capture_ids']).astype(str)
        if not np.array_equal(row_captures, np.asarray(data['groups']).astype(str)):
            raise ValueError('CIC capture_ids and groups disagree')
    elif 'unsw_extraction_manifest_json' in data:
        manifest = json.loads(str(data['unsw_extraction_manifest_json'].item()))
        captures = {x['device']:Path(x['capture']) for x in manifest['device_audits']}
        row_captures = np.asarray(data['device_names'])[np.asarray(data['labels'],int)].astype(str)
    else:
        raise ValueError('Missing supported extraction manifest')
    wanted = {}
    for i in indices:
        capture = str(row_captures[i])
        for j in range(int(np.sum(data['mask'][i]))):
            wanted.setdefault(capture,{}).setdefault(int(data['frame_ids'][i,j]),[]).append((int(i),j))
    result,metadata = {},{}
    for device, ids in wanted.items():
        path = captures[device]
        if capture_root is not None:
            path = Path(capture_root) / PureWindowsPath(str(path)).name
        if not path.exists():
            raise FileNotFoundError(f'Capture missing: {path}. Supply --capture-root or explicitly select assume_ethernet_ipv4.')
        remaining = set(ids)
        if progress:
            progress(f'Reading headers: {device}, {len(ids)} packet IDs')
        for record in read_capture(path):
            if record.frame_id not in remaining:
                continue
            header = parse_header(record.frame)
            for i,j in ids[record.frame_id]:
                if len(record.frame) != int(data['lengths'][i,j]) or abs(record.timestamp-float(data['timestamps'][i,j])) > 2e-6:
                    raise ValueError(f'PCAP/NPZ identity mismatch at flow {i}, packet {j}')
                result[f'{i}:{j}'] = asdict(header)
            remaining.remove(record.frame_id)
            if not remaining:
                break
        if remaining:
            raise ValueError(f'Missing packet IDs in {path}: {sorted(remaining)[:5]}')
        metadata[device] = {'path':str(path),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
                            'identity_check':'frame_id + timestamp + captured_length for every requested source event'}
    payload = {'schema':'securitas-headers-v1','dataset_sha256':dataset_sha,'indices':list(map(int,indices)),
               'capture_identity':metadata,'headers':result}
    if output_path is not None:
        write_json(output_path,payload)
    return payload


def load_headers(path, dataset_sha):
    h = read_json(path)
    if h.get('schema') != 'securitas-headers-v1' or h.get('dataset_sha256') != dataset_sha:
        raise ValueError('Header sidecar is not bound to this dataset')
    return h


def flows_from_data(data,indices,config,sidecar=None):
    out = []
    for i in indices:
        n = int(np.sum(data['mask'][i]))
        headers = None
        if sidecar is not None:
            try:
                headers = [Header(**sidecar['headers'][f'{i}:{j}']) for j in range(n)]
            except KeyError as e:
                raise ValueError(f'Header sidecar misses requested event {e}') from e
        if config['protocol']['header_mode'] == 'pcap' and headers is None:
            raise ValueError('PCAP mode requires a verified header sidecar')
        out.append(source_flow(*(data[k][i] for k in ('lengths','gaps','directions','mask')),
                               headers=headers,header_mode=config['protocol']['header_mode']))
    return out


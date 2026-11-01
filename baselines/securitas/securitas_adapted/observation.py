"""The sole classifier observation adapter; the simulator never truncates."""
import numpy as np
from .common import dataset_pcap


def observe_events(events, config):
    width = config['window_length']
    lengths = np.zeros(width, dtype=np.float64)
    gaps = np.zeros(width, dtype=np.float64)
    directions = np.zeros(width, dtype=np.int8)
    mask = np.zeros(width, dtype=np.int8)
    for i,e in enumerate(events[:width]):
        lengths[i],gaps[i],directions[i],mask[i] = e.length,e.gap,e.direction,1
    # The private wait before the first emitted packet remains in the full
    # physical event timestamps/costs, but the classifier clock starts at its
    # first observable event.  Keep the physical Event objects untouched.
    if np.any(mask):
        gaps[0] = 0.0
    arrays = (lengths,gaps,directions,mask)
    features = dataset_pcap.make_features(*(x[None,:] for x in arrays[:3]), **config['feature'])[0]
    return arrays,features


def features_for_flows(flows, config):
    # One feature encoding call for the batch, without allocating audit records.
    width = config['window_length']
    lengths = np.zeros((len(flows), width), dtype=np.float64)
    gaps = np.zeros_like(lengths)
    directions = np.zeros((len(flows), width), dtype=np.int8)
    for row, flow in enumerate(flows):
        for col, event in enumerate(flow[:width]):
            lengths[row, col] = event.length
            gaps[row, col] = event.gap
            directions[row, col] = event.direction
    gaps[:, 0] = 0.0
    return dataset_pcap.make_features(lengths, gaps, directions, **config['feature'])


def observe_results(results, config):
    observed = [observe_events(r.events,config) for r in results]
    arrays = tuple(np.stack([row[0][i] for row in observed]) for i in range(4))
    features = np.stack([row[1] for row in observed])
    audit = []
    width = config['window_length']
    for result in results:
        audit.append({'retained_events':min(width,len(result.events)),
                      'outside_window_events':[{'event_index':i,'role':e.role,'source_index':e.source_index,
                                                'parent_source_index':e.parent_source_index}
                                               for i,e in enumerate(result.events) if i >= width]})
    return {'arrays':arrays,'features':features,'observation_audit':audit}

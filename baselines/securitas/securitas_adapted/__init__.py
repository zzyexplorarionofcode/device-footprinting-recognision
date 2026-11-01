"""Securitas-adapted: independent offline event-level adaptation, not an original IoT reproduction."""
from .config import load_config
from .core import Header,SourceEvent,Event,ICMPEvent,Action,Strategy,TransformResult,apply_flow,apply_batch
from .data import source_flow
from .observation import observe_events,observe_results,features_for_flows

# Training/serialization imports are lazy, so deployment does not require PyTorch/sklearn model loading.
def __getattr__(name):
    if name in ('train_proxy','save_proxy','load_proxy'):
        from . import proxy
        return getattr(proxy,name)
    if name in ('train_policy','save_strategy','load_strategy'):
        from . import policy
        return getattr(policy,name)
    raise AttributeError(name)

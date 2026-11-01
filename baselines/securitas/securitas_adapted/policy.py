"""Universal autoregressive LSTM policy; each sampled table owns its REINFORCE cost."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import time
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from .common import write_json,read_json
from .config import validate_config
from .core import Action,Strategy,apply_batch,validate_strategy
from .observation import features_for_flows
from .icmp import PreparedFlow


class PolicyNetwork(nn.Module):
    def __init__(self,length,choices,hidden_size=32):
        super().__init__()
        self.length,self.choices = length,choices
        self.embeddings = nn.ModuleList([nn.Embedding(choices,hidden_size) for _ in range(length)])
        self.lstm = nn.LSTM(hidden_size,hidden_size,batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden_size,choices) for _ in range(length)])

    def sample(self,count,greedy=False):
        state = torch.zeros(count,dtype=torch.long,device=next(self.parameters()).device)
        hidden,actions,log_probs = None,[],[]
        for i in range(self.length):
            output,hidden = self.lstm(self.embeddings[i](state)[:,None,:],hidden)
            distribution = torch.distributions.Categorical(logits=self.heads[i](output[:,0,:]))
            state = distribution.logits.argmax(-1) if greedy else distribution.sample()
            actions.append(state)
            log_probs.append(distribution.log_prob(state))
        return torch.stack(actions,1),torch.stack(log_probs,1).sum(1)


def concrete_strategy(indices,config,rng,candidate_id):
    actions = []
    for index in indices:
        size = config['action_payload_bytes'][int(index)]
        operation = 'none' if size == 0 else ('split' if rng.random() < config['split_probability'] else 'dummy')
        actions.append(Action(size,operation))
    return Strategy(tuple(actions),int(candidate_id))


def _feature_cost(features, labels, clean_features, byte_ratios, proxy, config):
    probabilities = proxy.predict_proba(features)
    y = np.asarray(labels,int)
    confidence = float(probabilities[np.arange(len(y)),y].mean())
    distance = float(np.mean(np.linalg.norm(features-clean_features,axis=1) /
                             np.maximum(np.linalg.norm(clean_features,axis=1),config['epsilon'])))
    bandwidth = float(np.mean(byte_ratios))
    w = config['loss_weights']
    cost = w[0]*confidence-w[1]*distance+w[2]*bandwidth
    return cost,{'true_class_probability':confidence,'normalized_feature_l2':distance,'added_byte_ratio':bandwidth},probabilities


def strategy_cost(strategy,flows,labels,clean_features,proxy,config,*,simulation_seed=None):
    results = apply_batch(flows,strategy,config,simulation_seed=simulation_seed)
    features = features_for_flows([result.events for result in results],config)
    ratios = [r.cost['added_bytes']/max(r.cost['source_bytes'],config['epsilon']) for r in results]
    cost, parts, probabilities = _feature_cost(features,labels,clean_features,ratios,proxy,config)
    return cost,parts,results,probabilities


def validation_cost(strategy, flows, labels, clean_features, proxy, config):
    """Score all validation rows, retaining at most 256 flows' event objects.

    Proxy prediction and reductions still run once in original row order so
    chunking the deterministic simulator does not change candidate selection.
    """
    features = np.empty_like(clean_features)
    byte_ratios = np.empty(len(flows), dtype=np.float64)
    added_bytes = np.empty(len(flows), dtype=np.int64)
    for start in range(0, len(flows), 256):
        stop = min(start+256, len(flows))
        results = apply_batch(flows[start:stop], strategy, config)
        features[start:stop] = features_for_flows([r.events for r in results], config)
        byte_ratios[start:stop] = [r.cost['added_bytes']/max(r.cost['source_bytes'],config['epsilon']) for r in results]
        added_bytes[start:stop] = [r.cost['added_bytes'] for r in results]
        del results
    cost, parts, probabilities = _feature_cost(features,labels,clean_features,byte_ratios,proxy,config)
    return cost, parts, float(np.mean(added_bytes)), probabilities


def selection_key(ba,mean_bytes,candidate_id):
    return float(ba),float(mean_bytes),int(candidate_id)


def train_policy(train_flows,train_labels,validation_flows,validation_labels,proxy,config,*,progress=None):
    validate_config(config)
    cfg = config['policy']
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use device=cpu')
    seed = int(config['seed'])
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed+1)
    validation_rng = np.random.default_rng(seed+2)
    network = PolicyNetwork(config['policy_length'],len(config['action_payload_bytes']),cfg['hidden_size']).to(device)
    optimizer = torch.optim.Adam(network.parameters(),lr=cfg['learning_rate'])
    proxy.eval()
    proxy_before = proxy.fingerprint()
    preparation_started = time.perf_counter()
    if config['simulate_icmp']:
        if progress:
            progress('Preparing immutable ICMP flow metadata (once per training run)')
        train_flows = [flow if isinstance(flow, PreparedFlow) else PreparedFlow(flow) for flow in train_flows]
        validation_flows = [flow if isinstance(flow, PreparedFlow) else PreparedFlow(flow) for flow in validation_flows]
    clean = features_for_flows(train_flows,config)
    clean_val = features_for_flows(validation_flows,config)
    y,yv = np.asarray(train_labels,int),np.asarray(validation_labels,int)
    if len(y)!=len(train_flows) or len(yv)!=len(validation_flows) or not len(y) or not len(yv):
        raise ValueError('Nonempty label-matched train and validation flows required')
    initial = {k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    history,candidates,best,best_key,best_state = [],[],None,None,None
    candidate_id = 0
    preparation_seconds = time.perf_counter() - preparation_started
    if progress:
        work = cfg['strategies_per_batch']*len(y) + cfg['validation_candidates']*len(yv)
        progress(f'Policy ready: train={len(y)}, validation={len(yv)}, '
                 f'flow simulations/epoch={work}, epochs={cfg["epochs"]}, '
                 f'preparation={preparation_seconds:.1f}s')
    timings = []
    for epoch in range(cfg['epochs']):
        epoch_started = time.perf_counter()
        network.train()
        permutation = rng.permutation(len(y))
        for start in range(0,len(y),cfg['batch_size']):
            ix = permutation[start:start+cfg['batch_size']]
            batch = [train_flows[int(i)] for i in ix]
            indices,log_probs = network.sample(cfg['strategies_per_batch'])
            strategies = [concrete_strategy(row,config,rng,j) for j,row in enumerate(indices.detach().cpu().numpy())]
            costs,components = [],[]
            # Common random environment for competing tables in this batch;
            # fresh environment next batch. Validation/replay use saved base seed.
            environment_seed = config['icmp']['simulation_seed'] + 1 + len(history) if config['simulate_icmp'] else None
            for strategy in strategies:
                cost,parts,_,_ = strategy_cost(strategy,batch,y[ix],clean[ix],proxy,config,simulation_seed=environment_seed)
                costs.append(cost)
                components.append(parts)
            costs_t = torch.tensor(costs,dtype=log_probs.dtype,device=device)
            # Leave-one-out batch mean is independent of that strategy's sampled action.
            baseline = (costs_t.sum()-costs_t)/(len(costs)-1) if cfg['mean_baseline'] and len(costs)>1 else torch.zeros_like(costs_t)
            loss = ((costs_t-baseline).detach()*log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append({'epoch':epoch,'batch_start':start,'flow_indices_in_training_subset':ix.tolist(),
                            'strategy_costs':costs,'strategy_components':components,
                            'strategy_log_probabilities':log_probs.detach().cpu().tolist(),
                            'baseline':baseline.detach().cpu().tolist(),'policy_gradient_loss':float(loss.detach().cpu())})
            if config['simulate_icmp']:
                history[-1]['icmp_environment_seed'] = environment_seed
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        network.eval()
        with torch.no_grad():
            # First candidate is greedy, remaining candidates sample this exact network.
            greedy,_ = network.sample(1,greedy=True)
            rows = [greedy[0].cpu().numpy()]
            if cfg['validation_candidates'] > 1:
                sampled,_ = network.sample(cfg['validation_candidates']-1)
                rows.extend(sampled.cpu().numpy())
        for row in rows:
            strategy = concrete_strategy(row,config,validation_rng,candidate_id)
            _,parts,mean_bytes,prob = validation_cost(strategy,validation_flows,yv,clean_val,proxy,config)
            ba = float(balanced_accuracy_score(yv,prob.argmax(1)))
            key = selection_key(ba,mean_bytes,candidate_id)
            candidates.append({'candidate_id':candidate_id,'epoch':epoch,'validation_ba':ba,'mean_added_bytes':mean_bytes,
                               'components':parts,'actions':[asdict(a) for a in strategy.actions]})
            if best_key is None or key < best_key:
                best,best_key = strategy,key
                best_state = {k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
            candidate_id += 1
        validation_seconds = time.perf_counter() - validation_started
        elapsed = time.perf_counter() - epoch_started
        timings.append(dict(epoch=epoch+1, training_seconds=training_seconds,
                            validation_seconds=validation_seconds, seconds=elapsed))
        if progress:
            remaining = sum(item['seconds'] for item in timings) / len(timings) * (cfg['epochs']-epoch-1)
            progress(f'Policy epoch {epoch+1}/{cfg["epochs"]}: selected candidate={best.candidate_id}, '
                     f'validation BA={best_key[0]:.6f}; train={training_seconds:.1f}s, '
                     f'validation={validation_seconds:.1f}s, remaining~{remaining/60:.1f}min')
    if proxy.fingerprint() != proxy_before:
        raise AssertionError('Proxy changed during policy optimization')
    metadata = {'seed':seed,'cost_formula':'w_probability*p_true - w_distance*normalized_feature_L2 + w_bandwidth*added_byte_ratio',
                'cost_weights':config['loss_weights'],'baseline':'leave-one-out strategy batch mean; zero for one strategy',
                'selection':['validation_ba','mean_added_bytes','candidate_id'],'selected_candidate_id':best.candidate_id,
                'selected_key':list(best_key),'history':history,'candidates':candidates,
                'proxy_fingerprint':proxy_before,'proxy_unchanged':True,
                'network_parameters_changed':any(not torch.equal(initial[k],v.detach().cpu()) for k,v in network.state_dict().items()),
                'train_count':len(y),'validation_count':len(yv),
                'performance':{'preparation_seconds':preparation_seconds,'epochs':timings}}
    if config['simulate_icmp']:
        metadata['icmp_environment'] = {'config':deepcopy(config['icmp']),
            'training':'fresh seed per batch, common across strategies; outcomes keyed by source flow and uplink parent',
            'validation_and_replay':'saved base seed, independent of candidate ID and batch ordering',
            'budget':'reserve one potential response before dummy; dummy cap includes uplink dummy and ICMP',
            'final_test_used':False}
    return best,metadata,best_state


def save_strategy(strategy,config,path,*,metadata=None):
    validate_config(config)
    validate_strategy(strategy,config)
    write_json(path,{'schema':'securitas-adapted-strategy-v1','config':deepcopy(config),
                     'strategy':asdict(strategy),'metadata':deepcopy(metadata or {})})


def load_strategy(path):
    payload = read_json(path)
    if payload.get('schema') != 'securitas-adapted-strategy-v1':
        raise ValueError('Unsupported strategy schema')
    c = validate_config(payload['config'])
    s = Strategy(tuple(Action(**a) for a in payload['strategy']['actions']),payload['strategy']['candidate_id'])
    validate_strategy(s,c)
    return s,c,payload['metadata']


def save_policy_network(state,config,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':state,'policy_config':config['policy'],'policy_length':config['policy_length'],
                'action_payload_bytes':config['action_payload_bytes']},path)

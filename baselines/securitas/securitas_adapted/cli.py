"""Single-method closed loop; intentionally no full comparison experiment machinery."""
import argparse
import gc
import json
import shutil
from pathlib import Path
import platform
import time
import numpy as np
import sklearn
import torch
from .common import ARTIFACTS_ROOT,PROJECT_ROOT,jsonable,read_json,write_json,sha256,dataset_pcap
from .config import load_config
from .data import prepare_data,recover_headers,load_headers,flows_from_data
from .proxy import train_proxy,save_proxy,load_proxy
from .policy import train_policy,save_strategy,load_strategy,save_policy_network
from .core import apply_batch
from .observation import observe_results


def output_directory(value):
    path = Path(value).resolve()
    legacy_root = ARTIFACTS_ROOT / 'legacy'
    if not path.is_relative_to(ARTIFACTS_ROOT) or path.is_relative_to(legacy_root):
        raise ValueError('Outputs must be inside project artifacts and outside artifacts/legacy (read-only)')
    path.mkdir(parents=True,exist_ok=True)
    return path


def smoke_subset(indices,data,count):
    selected = []
    for label in np.unique(data['labels'][indices]):
        ix = [int(i) for i in indices if data['labels'][i]==label]
        ix.sort(key=lambda i:int(max(data['frame_ids'][i])) if 'frame_ids' in data else i)
        selected.extend(ix[:count])
    return np.asarray(selected,int)


def save_application(results,indices,c,out,prefix):
    observed = observe_results(results,c)
    np.savez_compressed(out/(prefix+'.npz'),**dict(zip(('lengths','gaps','directions','mask'),observed['arrays'])),features=observed['features'])
    path=out/(prefix+'_events.json')
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8') as stream:
            stream.write('{"flow_indices":')
            json.dump(jsonable(indices),stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            stream.write(',"results":[')
            for i,result in enumerate(results):
                if i:stream.write(',')
                json.dump(jsonable(result.to_dict()),stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            stream.write('],"observation_audit":[')
            for i,audit in enumerate(observed['observation_audit']):
                if i:stream.write(',')
                json.dump(jsonable(audit),stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            stream.write(']}\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_command(args):
    started = time.perf_counter()
    c = load_config(args.config)
    torch.set_num_threads(args.threads)
    if args.header_mode:
        c['protocol']['header_mode'] = args.header_mode
    if args.smoke:
        c['proxy']['epochs'] = 2
        c['policy'].update(epochs=1,batch_size=16,strategies_per_batch=2,validation_candidates=2)
    out = output_directory(args.output)
    if (out/'strategy.json').exists():
        raise FileExistsError('Use a new output directory; existing strategies are not overwritten')
    data,train_ix,val_ix,split = prepare_data(c)
    if args.smoke:
        train_ix,val_ix = smoke_subset(train_ix,data,4),smoke_subset(val_ix,data,2)
    split.update(consumed_train_indices=train_ix.tolist(),consumed_validation_indices=val_ix.tolist(),smoke_only=bool(args.smoke))
    sidecar = None
    if args.headers:
        sidecar = load_headers(args.headers,split['dataset_sha256'])
        if Path(args.headers).resolve() != (out/'headers.json').resolve():
            shutil.copy2(args.headers,out/'headers.json')
    elif c['protocol']['header_mode']=='pcap':
        sidecar = recover_headers(data,np.r_[train_ix,val_ix],out/'headers.json',split['dataset_sha256'],args.capture_root,print)
    train_flows,val_flows = (flows_from_data(data,ix,c,sidecar) for ix in (train_ix,val_ix))
    write_json(out/'resolved_config.json',c)
    write_json(out/'split.json',split)
    proxy = load_proxy(args.proxy,c,split) if args.proxy else train_proxy(train_flows,data['labels'][train_ix],val_flows,data['labels'][val_ix],c,split_metadata=split,progress=print)
    save_proxy(proxy,out/'proxy.joblib')
    strategy,training,state = train_policy(train_flows,data['labels'][train_ix],val_flows,data['labels'][val_ix],proxy,c,progress=print)
    write_json(out/'training.json',training)
    save_policy_network(state,c,out/'policy_network.pt')
    metadata = {'split':split,'proxy':proxy.metadata,'proxy_fingerprint':proxy.fingerprint(),
                'proxy_artifact_sha256':sha256(out/'proxy.joblib'),'selection':training['selection'],
                'selected_key':training['selected_key'],'smoke_only':bool(args.smoke),'simulate_icmp':c['simulate_icmp'],
                'first_packet_wait_observable':False,
                'observation_origin':'first_emitted_packet',
                'assumptions':['event-sequence simulation only','no tunnel/FCS','no endpoint behavior',
                               'all subsequent inbound/outbound source events shift by accumulated delta']}
    if c['simulate_icmp']:
        metadata['icmp_environment'] = training['icmp_environment']
        metadata['assumptions'].extend(['ICMP probability/timing are simulation assumptions',
            'paper timing uses the next real inbound event in the complete offline source flow',
            'ICMP merges by arrival time and does not itself delay source events'])
    save_strategy(strategy,c,out/'strategy.json',metadata=metadata)
    restored,rc,_ = load_strategy(out/'strategy.json')
    before,after = apply_batch(val_flows,strategy,c),apply_batch(val_flows,restored,rc)
    if any(a.to_dict()!=b.to_dict() for a,b in zip(before,after)):
        raise AssertionError('Saved strategy replay differs')
    del before,val_flows,train_flows,proxy,state
    gc.collect()
    save_application(after,val_ix,c,out,'validation_application')
    report = {'name':'Securitas-adapted','smoke_only':bool(args.smoke),'classes':len(data['device_names']),
              'train_flows':len(train_ix),'validation_flows':len(val_ix),'final_test_flows_used_for_training_or_validation':0,
              'outer_split_counts':[len(split['outer_train_indices']),len(split['outer_test_indices'])],
              'capture_partitions_disjoint':True,'proxy_frozen':training['proxy_unchanged'],
              'policy_parameters_updated':training['network_parameters_changed'],'strategy_save_load_identical':True,
              'selected_candidate_id':strategy.candidate_id,'selected_validation_ba_internal_only':training['selected_key'][0],
              'header_mode':c['protocol']['header_mode'],'simulate_icmp':c['simulate_icmp'],'budget':c['budget'],
              'delta_seconds':c['delta_seconds'],'validation_total_added_bytes':sum(r.cost['added_bytes'] for r in after),
              'validation_total_dummy_count':sum(r.cost['dummy_count'] for r in after),
              'validation_total_split_count':sum(r.cost['split_count'] for r in after),
              'validation_total_real_completion_delay_seconds':sum(r.cost['real_completion_delay_seconds'] for r in after),
              'validation_total_wire_completion_delay_seconds':sum(r.cost['wire_completion_delay_seconds'] for r in after),
              'validation_total_cover_tail_seconds':sum(r.cost['cover_tail_seconds'] for r in after),
              'timing_metric_semantics':{'added_delay_seconds':'legacy compatibility alias for wire_completion_delay_seconds',
                                         'real_completion_delay_seconds':'last real/fragment timestamp minus source duration',
                                         'wire_completion_delay_seconds':'last emitted wire event timestamp minus source duration',
                                         'cover_tail_seconds':'wire completion minus real/fragment completion'},
              'seconds':time.perf_counter()-started,'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__}}
    if c['simulate_icmp']:
        report.update(icmp_config=c['icmp'],
            validation_total_icmp_reply_count=sum(r.cost['icmp_reply_count'] for r in after),
            validation_total_icmp_added_bytes=sum(r.cost['icmp_added_bytes'] for r in after),
            validation_total_uplink_dummy_count=sum(r.cost['uplink_dummy_count'] for r in after))
    write_json(out/'run_verification.json',report)
    print('Closed loop verified; outputs:',out,flush=True)
    return report


def finalize_command(args):
    """Complete a saved training run after validation-event serialization failed."""
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    started=time.perf_counter()
    out=output_directory(args.output)
    if (out/'run_verification.json').exists():
        raise FileExistsError('Training is already verified')
    required=('strategy.json','proxy.joblib','policy_network.pt','training.json',
              'resolved_config.json','split.json','headers.json','validation_application.npz')
    missing=[name for name in required if not (out/name).exists()]
    if missing:
        raise FileNotFoundError(f'Saved training artifacts are incomplete: {missing}')
    strategy,c,metadata=load_strategy(out/'strategy.json')
    if c!=read_json(out/'resolved_config.json'):
        raise ValueError('Saved strategy/config mismatch')
    split=read_json(out/'split.json')
    training=read_json(out/'training.json')
    if (metadata['split']!=split or metadata['selected_key']!=training['selected_key']
            or metadata['proxy_artifact_sha256']!=sha256(out/'proxy.joblib')
            or training['selected_candidate_id']!=strategy.candidate_id):
        raise ValueError('Saved strategy, split, proxy or training record mismatch')
    dataset_path=Path(c['data']['dataset'])
    if not dataset_path.is_absolute():
        dataset_path=PROJECT_ROOT/dataset_path
    if sha256(dataset_path)!=split['dataset_sha256']:
        raise ValueError('Dataset changed since training')
    data=dataset_pcap.load_dataset(dataset_path)
    val_ix=np.asarray(split['consumed_validation_indices'],dtype=int)
    with np.load(out/'validation_application.npz',allow_pickle=False) as saved:
        expected={name:saved[name] for name in ('lengths','gaps','directions','mask','features')}
    if any(len(array)!=len(val_ix) for array in expected.values()):
        raise ValueError('Saved validation arrays have the wrong flow count')
    sidecar=load_headers(out/'headers.json',split['dataset_sha256'])
    path=out/'validation_application_events.json'
    temporary=path.with_suffix(path.suffix+'.tmp')
    audit_temp=out/'validation_application_audit.tmp'
    totals={name:0. for name in ('added_bytes','dummy_count','split_count',
                                'real_completion_delay_seconds','wire_completion_delay_seconds',
                                'cover_tail_seconds','icmp_reply_count','icmp_added_bytes',
                                'uplink_dummy_count')}
    try:
        with temporary.open('w',encoding='utf-8') as stream, audit_temp.open('w',encoding='utf-8') as audits:
            stream.write('{"flow_indices":')
            json.dump(val_ix.tolist(),stream,separators=(',',':'),allow_nan=False)
            stream.write(',"results":[')
            for start in range(0,len(val_ix),args.batch_size):
                end=min(start+args.batch_size,len(val_ix))
                flows=flows_from_data(data,val_ix[start:end],c,sidecar)
                results=apply_batch(flows,strategy,c)
                observed=observe_results(results,c)
                for name,actual in zip(('lengths','gaps','directions','mask'),observed['arrays']):
                    np.testing.assert_array_equal(actual,expected[name][start:end])
                np.testing.assert_array_equal(observed['features'],expected['features'][start:end])
                for i,result in enumerate(results,start):
                    if i:stream.write(',')
                    json.dump(jsonable(result.to_dict()),stream,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                    for name in totals:
                        totals[name]+=result.cost[name]
                for i,audit in enumerate(observed['observation_audit'],start):
                    if i:audits.write(',')
                    json.dump(jsonable(audit),audits,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                if end==len(val_ix) or end//args.batch_size%20==0:
                    print(f'Validation replay {end}/{len(val_ix)}',flush=True)
            stream.write('],"observation_audit":[')
        with temporary.open('a',encoding='utf-8') as stream, audit_temp.open('r',encoding='utf-8') as audits:
            shutil.copyfileobj(audits,stream)
            stream.write(']}\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        audit_temp.unlink(missing_ok=True)
    report={'name':'Securitas-adapted','smoke_only':bool(metadata['smoke_only']),
            'classes':len(data['device_names']),'train_flows':len(split['consumed_train_indices']),
            'validation_flows':len(val_ix),'final_test_flows_used_for_training_or_validation':0,
            'outer_split_counts':[len(split['outer_train_indices']),len(split['outer_test_indices'])],
            'capture_partitions_disjoint':True,'proxy_frozen':training['proxy_unchanged'],
            'policy_parameters_updated':training['network_parameters_changed'],
            'strategy_save_load_identical':True,'selected_candidate_id':strategy.candidate_id,
            'selected_validation_ba_internal_only':training['selected_key'][0],
            'header_mode':c['protocol']['header_mode'],'simulate_icmp':c['simulate_icmp'],
            'budget':c['budget'],'delta_seconds':c['delta_seconds'],
            'validation_total_added_bytes':totals['added_bytes'],
            'validation_total_dummy_count':totals['dummy_count'],
            'validation_total_split_count':totals['split_count'],
            'validation_total_real_completion_delay_seconds':totals['real_completion_delay_seconds'],
            'validation_total_wire_completion_delay_seconds':totals['wire_completion_delay_seconds'],
            'validation_total_cover_tail_seconds':totals['cover_tail_seconds'],
            'timing_metric_semantics':{'added_delay_seconds':'legacy compatibility alias for wire_completion_delay_seconds',
                'real_completion_delay_seconds':'last real/fragment timestamp minus source duration',
                'wire_completion_delay_seconds':'last emitted wire event timestamp minus source duration',
                'cover_tail_seconds':'wire completion minus real/fragment completion'},
            'seconds':None,'finalization_seconds':time.perf_counter()-started,
            'recovered_from_saved_strategy':True,'validation_replay_checked_against_saved_arrays':len(val_ix),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,
                        'sklearn':sklearn.__version__,'torch':torch.__version__}}
    if c['simulate_icmp']:
        report.update(icmp_config=c['icmp'],
                      validation_total_icmp_reply_count=totals['icmp_reply_count'],
                      validation_total_icmp_added_bytes=totals['icmp_added_bytes'],
                      validation_total_uplink_dummy_count=totals['uplink_dummy_count'])
    write_json(out/'run_verification.json',report)
    print('Saved strategy and all validation flows verified; outputs:',out,flush=True)
    return report


def apply_command(args):
    strategy,c,_ = load_strategy(args.strategy)
    path = Path(args.dataset)
    data = dataset_pcap.load_dataset(path)
    ix = np.asarray([int(i) for i in args.indices.split(',')],int)
    if len(ix)==0 or np.any(ix<0) or np.any(ix>=len(data['lengths'])):
        raise ValueError('Invalid requested flow indices')
    out = output_directory(args.output)
    h = load_headers(args.headers,sha256(path)) if args.headers else None
    if h is None and c['protocol']['header_mode']=='pcap':
        h = recover_headers(data,ix,out/'headers.json',sha256(path),args.capture_root,print)
    # These arrays, not device labels, are passed to the actual transform.
    flows = flows_from_data(data,ix,c,h)
    results = apply_batch(flows,strategy,c,simulation_seed=args.simulation_seed)
    save_application(results,ix,c,out,'application')
    write_json(out/'application_metadata.json',{'strategy_sha256':sha256(args.strategy),'dataset_sha256':sha256(path),
               'simulate_icmp':c['simulate_icmp'],
               'icmp_simulation_seed':(c['icmp']['simulation_seed'] if args.simulation_seed is None else args.simulation_seed) if c['simulate_icmp'] else None})
    print('Applied fixed strategy:',out,flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Securitas-adapted standalone event simulator')
    sub = parser.add_subparsers(dest='command',required=True)
    train = sub.add_parser('train',help='Proxy -> policy -> save/load -> validation replay')
    train.add_argument('--config',required=True)
    train.add_argument('--output',required=True)
    train.add_argument('--smoke',action='store_true',help='Tiny closed-loop verification, not a performance experiment')
    train.add_argument('--threads',type=int,default=1)
    train.add_argument('--proxy',help='Trusted saved proxy with identical data/split/features')
    train.add_argument('--header-mode',choices=['pcap','assume_ethernet_ipv4'])
    train.add_argument('--headers')
    train.add_argument('--capture-root')
    train.set_defaults(func=train_command)
    finalize = sub.add_parser('finalize',help='Verify and finish a saved training run without retraining')
    finalize.add_argument('--output',required=True)
    finalize.add_argument('--batch-size',type=int,default=256)
    finalize.set_defaults(func=finalize_command)
    apply = sub.add_parser('apply',help='Apply saved fixed strategy')
    apply.add_argument('--strategy',required=True)
    apply.add_argument('--dataset',required=True)
    apply.add_argument('--indices',required=True)
    apply.add_argument('--output',required=True)
    apply.add_argument('--headers')
    apply.add_argument('--capture-root')
    apply.add_argument('--simulation-seed',type=int,help='Optional ICMP environment realization; fixed policy actions remain unchanged')
    apply.set_defaults(func=apply_command)
    args = parser.parse_args(argv)
    args.func(args)


if __name__=='__main__':
    main()

"""ICMP evaluation on current UNSW and CIC session datasets."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import numpy as np
import joblib
import sklearn
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from securitas_adapted.common import read_json,write_json,sha256
from data_pipeline import pcap as dataset_pcap
from securitas_adapted.data import recover_headers,load_headers,flows_from_data,prepare_data,SUPPORTED_PROTOCOLS
from securitas_adapted.policy import load_strategy
from securitas_adapted.config import load_config
from securitas_adapted.core import apply_batch
from securitas_adapted.observation import observe_results
from experiments import run_experiment as engine
from dataclasses import asdict
from experiments.lstm_attacker import LSTMConfig
from configs.experiments.current import EVALUATION
LSTM_PARAMS = asdict(LSTMConfig())

RATIOS=tuple(EVALUATION['retraining_ratios'])
MODELS=('MLP-full','RF-full','SVM-full','LSTM-full')
DEFAULT_ROOT=ROOT/'artifacts/runs/securitas/icmp_high_20260913'


def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()


def safe(name):
    return name.replace('-','_')


def assert_current_partition(data, ids, classes):
    actual_protocol=str(np.asarray(data['data_protocol']).item())
    if actual_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f'Unexpected dataset protocol: {actual_protocol!r}')
    labels=np.asarray(data['labels'],int);groups=np.asarray(data['groups']).astype(str)
    if len(np.unique(labels))!=classes or len(data['device_names'])!=classes:
        raise AssertionError('Dataset class coverage differs from the selected experiment')
    from data_pipeline.partitions import frozen_partition
    frozen = frozen_partition(data, train_ratio=.75, seed=42)
    if frozen is not None:
        for role in ('train', 'validation', 'test'):
            np.testing.assert_array_equal(ids[role], np.flatnonzero(frozen[0][role]))
        return
    expected_train,expected_test=dataset_pcap.split_by_capture(labels,groups,train_ratio=.75,seed=42)
    np.testing.assert_array_equal(ids['train'],np.flatnonzero(expected_train))
    np.testing.assert_array_equal(ids['test'],np.flatnonzero(expected_test))
    if len(ids['train'])+len(ids['test'])!=len(labels):
        raise AssertionError('Outer partition does not cover every flow')
    if set(groups[ids['train']])&set(groups[ids['test']]):
        raise AssertionError('Outer partition shares a capture group')
    if set(groups[ids['train']])|set(groups[ids['test']])!=set(groups):
        raise AssertionError('Outer partition lost a capture group')
    for label in np.unique(labels):
        if not np.any(labels[ids['train']]==label) or not np.any(labels[ids['test']]==label):
            raise AssertionError(f'Class {label} is absent from one outer partition')


def setup(out,classes,config_path=None,*,persist=True,repeats=3):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    config_path=Path(config_path or HERE/f'configs/unsw{classes}_high_icmp.json').resolve()
    project=load_config(config_path)
    dataset=ROOT/project['data']['dataset']
    data,proxy_train,proxy_validation,proxy_split=prepare_data(project)
    from data_pipeline.partitions import frozen_partition
    frozen=frozen_partition(data,train_ratio=project['data']['outer_train_ratio'],seed=project['data']['outer_seed'])
    if frozen is not None:
        ids={role:np.flatnonzero(mask) for role,mask in frozen[0].items()}
    else:
        train,test=dataset_pcap.split_by_capture(data['labels'],data['groups'],train_ratio=project['data']['outer_train_ratio'],seed=project['data']['outer_seed'])
        ids={'train':np.flatnonzero(train),'test':np.flatnonzero(test)}
    assert_current_partition(data,ids,classes)
    feature=project['feature']
    clean=dataset_pcap.make_features(*(data[k] for k in ('lengths','gaps','directions')),**feature)
    assert len(data['device_names'])==classes and project['simulate_icmp'] is True
    source_paths=list((HERE/'securitas_adapted').glob('*.py'))+[Path(__file__),Path(engine.__file__),Path(engine.shared_tools.__file__),ROOT/'experiments/lstm_attacker.py',ROOT/'data_pipeline/pcap.py',ROOT/'data_pipeline/unsw.py',config_path,Path(project['provenance']['method33_config']),ROOT/'configs/experiments/current.py',HERE/'run.py']
    source_paths.extend([ROOT/'data_pipeline/partitions.py',ROOT/'data_pipeline/cic.py',ROOT/'methods/method33_tools.py'])
    sources={str(p.relative_to(ROOT)):sha256(p) for p in source_paths}
    config={'classes':classes,'method':'Securitas-adapted-ICMP','data_status':project['dataset_status'],'data_protocol':project['data_protocol'],'icmp':project['icmp'],'budget':project['budget'],'models':MODELS,'ratios':RATIOS,'repeats':repeats,
            'mask_seed':731100,'model_seed':42,'seed_step':10000,'feature':feature,'lstm':LSTM_PARAMS,
            'policy_training':project['policy'],'proxy_training':project['proxy'],
            'dataset_sha256':sha256(dataset),'train_indices_sha256':array_hash(ids['train']),
            'test_indices_sha256':array_hash(ids['test']),'source_sha256':sources,
            'proxy_train_indices_sha256':array_hash(proxy_train),'proxy_validation_indices_sha256':array_hash(proxy_validation),
            'proxy_train_count':len(proxy_train),'proxy_validation_count':len(proxy_validation)}
    if not persist:
        return data,ids,clean,config
    cp=out/'evaluation_protocol.json'
    if cp.exists():
        assert read_json(cp)==json.loads(json.dumps(config))
    else:
        write_json(cp,config)
        for p in source_paths:
            dst=out/'evaluation_source_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        np.savez_compressed(out/'evaluation_split.npz',**ids)
    return data,ids,clean,config


class Cache:
    def __init__(self,out,config):
        self.path=out/'model_cache';self.path.mkdir(exist_ok=True)
        self.config=config

    def fit(self,name,x,y,seed):
        desc={'model':name,'train_features_sha256':array_hash(x),'train_labels_sha256':array_hash(y),
              'source_sha256':self.config['source_sha256'],'python':platform.python_version(),
              'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
              'seed':seed,'lstm':LSTM_PARAMS if name=='LSTM-full' else None}
        key=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest()
        file=self.path/(key+'.joblib');meta=file.with_suffix('.json')
        if meta.exists():
            rec=read_json(meta)
            if rec.get('descriptor')!=desc or not file.exists() or sha256(file)!=rec.get('model_sha256'):
                raise ValueError('Incompatible or corrupted current-output model cache; refit in a new output directory')
            scaler,model=joblib.load(file)
        else:
            started=time.perf_counter()
            origin={'kind':'new_fit_for_securitas'}
            print(f'FIT {name} seed={seed}',flush=True)
            scaler,model=engine.fit_one_model(name,x,y,seed,lstm_params=LSTM_PARAMS)
            tmp=file.with_suffix('.tmp');joblib.dump((scaler,model),tmp,compress=3);tmp.replace(file)
            rec={'descriptor':desc,'model_sha256':sha256(file),'origin':origin,'fit_seconds':time.perf_counter()-started,
                 'n_iter':getattr(model,'n_iter_',None),'training_protocol':'full-data-class-weight-v1','fit_rows':len(y)}
            write_json(meta,rec)
        expected=StandardScaler().fit(x)
        np.testing.assert_array_equal(scaler.mean_,expected.mean_);np.testing.assert_array_equal(scaler.scale_,expected.scale_)
        return scaler,model,key,rec


def assert_strategy_partition(meta, ids, dataset_sha):
    split = meta['split']
    if split['dataset_sha256'] != dataset_sha:
        raise ValueError('Strategy was trained on a different dataset')
    expected_outer = np.sort(np.r_[ids['train'], ids.get('validation', [])]).astype(int)
    np.testing.assert_array_equal(split['outer_train_indices'], expected_outer)
    np.testing.assert_array_equal(split['outer_test_indices'], ids['test'])
    if 'validation' in ids:
        np.testing.assert_array_equal(split['consumed_train_indices'], ids['train'])
        np.testing.assert_array_equal(split['consumed_validation_indices'], ids['validation'])


def assert_training_configuration(saved, protocol):
    for part in ('proxy', 'policy'):
        if saved[part] != protocol[part + '_training']:
            raise ValueError(f'Saved {part} training configuration differs from selected experiment config')


def prepare_applications(out,data,ids,config,*,capture_root=None,test_headers=None):
    strategy,c,meta=load_strategy(out/'training/strategy.json')
    assert meta['smoke_only'] is False and c['simulate_icmp'] is True
    assert c['icmp']==config['icmp'] and c['budget']==config['budget']
    assert_training_configuration(c, config)
    verification=read_json(out/'training/run_verification.json')
    assert verification['proxy_frozen'] and verification['strategy_save_load_identical'] and not verification['smoke_only']
    assert c['feature']==config['feature'] and c['window_length']==32
    assert c['data']['outer_seed']==42 and c['data']['outer_train_ratio']==.75
    assert_strategy_partition(meta, ids, config['dataset_sha256'])
    for name in ('train','validation'):
        assert array_hash(np.asarray(meta['split'][f'consumed_{name}_indices'],dtype=np.int64))==config[f'proxy_{name}_indices_sha256']
    apps={}
    for part in ('train','test'):
        destination=out/'applications';destination.mkdir(exist_ok=True)
        info=destination/(part+'.json');npz=destination/(part+'.npz')
        if info.exists():
            rec=read_json(info);assert rec['strategy_sha256']==sha256(out/'training/strategy.json') and rec['arrays_sha256']==sha256(npz)
            for name,digest in rec['audit_files_sha256'].items():assert sha256(destination/name)==digest
            with np.load(npz,allow_pickle=False) as z:apps[part]=z['features'].copy()
            continue
        if part=='train':
            h=load_headers(out/'training/headers.json',config['dataset_sha256'])
        else:
            hp=out/'test_headers.json'
            if test_headers is not None and not hp.exists():
                load_headers(test_headers,config['dataset_sha256'])
                shutil.copy2(test_headers,hp)
            h=load_headers(hp,config['dataset_sha256']) if hp.exists() else recover_headers(data,ids[part],hp,config['dataset_sha256'],capture_root=capture_root,progress=print)
        flows=flows_from_data(data,ids[part],c,h)
        results=apply_batch(flows,strategy,c)
        observed=observe_results(results,c);apps[part]=observed['features']
        full_width=max(len(r.events) for r in results)
        full={k:np.zeros((len(results),full_width),dtype=np.float64) for k in ('full_lengths','full_gaps','full_times')}
        full.update(full_directions=np.zeros((len(results),full_width),np.int8),roles=np.full((len(results),full_width),-1,np.int8),parents=np.full((len(results),full_width),-1,np.int32),payload_offsets=np.full((len(results),full_width),-1,np.int32),payload_bytes=np.full((len(results),full_width),-1,np.int32))
        for i,(source,r) in enumerate(zip(flows,results)):
            assert r.cost['added_bytes']<=c['budget']['bytes'] and r.cost['added_delay_seconds']<=c['budget']['delay_seconds'] and r.cost['dummy_count']<=c['budget']['dummy_packets']
            source_ids=[e.source_index for e in r.events if e.role in ('real','fragment')];assert list(dict.fromkeys(source_ids))==list(range(len(source)))
            for j,e in enumerate(r.events):
                full['full_lengths'][i,j]=e.length;full['full_gaps'][i,j]=e.gap;full['full_times'][i,j]=e.timestamp
                full['full_directions'][i,j]=e.direction;full['roles'][i,j]={'real':0,'fragment':1,'dummy':2,'icmp':3}[e.role]
                full['parents'][i,j]=e.parent_source_index
                full['payload_offsets'][i,j]=e.payload_offset if e.payload_offset is not None else -1
                full['payload_bytes'][i,j]=e.header.payload_bytes if e.header.payload_bytes is not None else -1
        costs={k:np.array([r.cost[k] for r in results]) for k in ('source_bytes','source_duration_seconds','added_bytes','added_delay_seconds','real_completion_delay_seconds','wire_completion_delay_seconds','cover_tail_seconds','dummy_count','split_count','added_events','uplink_dummy_count','icmp_reply_count','icmp_added_bytes','uplink_dummy_bytes','local_added_events','local_timeline_shift_seconds','icmp_tail_extension_seconds','budget_reserved_added_bytes','budget_reserved_dummy_count','budget_reserved_added_delay_seconds')}
        costs['added_packets'] = costs['added_events'].copy()
        np.savez_compressed(npz,indices=ids[part],features=apps[part],**dict(zip(('lengths','gaps','directions','mask'),observed['arrays'])),**full,**costs)
        from collections import Counter
        reasons=Counter(d.get('reason','executed') for r in results for d in r.decisions)
        fallback=Counter(d['split_infeasible_reason'] for r in results for d in r.decisions if 'split_infeasible_reason' in d)
        overhead={}
        for k,values in costs.items():overhead[k]={'mean':float(np.mean(values)),'p95':float(np.percentile(values,95)),'max':float(np.max(values)),'sum':float(np.sum(values))}
        overhead['bytes_overhead_percent']=float(100*np.sum(costs['added_bytes'])/np.sum(costs['source_bytes']))
        overhead['seconds_overhead_percent']=float(100*np.sum(costs['wire_completion_delay_seconds'])/np.sum(costs['source_duration_seconds']))
        overhead['real_completion_seconds_overhead_percent']=float(100*np.sum(costs['real_completion_delay_seconds'])/np.sum(costs['source_duration_seconds']))
        overhead['wire_completion_seconds_overhead_percent']=overhead['seconds_overhead_percent']
        overhead['cover_tail_seconds_overhead_percent']=float(100*np.sum(costs['cover_tail_seconds'])/np.sum(costs['source_duration_seconds']))
        overhead['timing_metric_semantics']={'added_delay_seconds':'legacy compatibility alias for wire_completion_delay_seconds',
                                             'real_completion_delay_seconds':'last real/fragment timestamp minus source duration',
                                             'wire_completion_delay_seconds':'last emitted wire event timestamp minus source duration',
                                             'cover_tail_seconds':'wire completion minus real/fragment completion'}
        shared_audit = None
        if c.get('shared_budget') is not None:
            from experiment_support.budget import PhysicalBudget
            shared_audit = PhysicalBudget(**c['shared_budget']).audit(costs)
        rec={'part':part,'flow_count':len(results),'strategy_sha256':sha256(out/'training/strategy.json'),'arrays_sha256':sha256(npz),
             'overhead':overhead,'shared_budget':c.get('shared_budget'),'budget_audit':shared_audit,'decision_reasons':dict(reasons),'split_fallback_reasons':dict(fallback),'budget':c['budget'],
             'timing_metrics':overhead['timing_metric_semantics'],
             'deterministic_application_shared_across_repeats':True,'complete_events_before_window':True,'source_order_preserved':True}
        # Full action/cancellation audit, one row per input flow; not part of classifier features.
        with (destination/(part+'_decisions.jsonl')).open('w',encoding='utf-8') as f:
            for i,r in zip(ids[part],results):f.write(json.dumps({'flow_index':int(i),'decisions':r.decisions},ensure_ascii=False)+'\n')
        with (destination/(part+'_events.jsonl')).open('w',encoding='utf-8') as f:
            for i,r in zip(ids[part],results):f.write(json.dumps({'flow_index':int(i),**r.to_dict()},ensure_ascii=False)+'\n')
        rec['audit_files_sha256']={part+suffix:sha256(destination/(part+suffix)) for suffix in ('_decisions.jsonl','_events.jsonl')}
        rec['icmp']=c['icmp']
        rec['event_roles']={'real':0,'fragment':1,'dummy':2,'icmp':3}
        rec['tail_icmp_events']=sum(d.get('icmp',{}).get('status')=='generated' and d.get('icmp',{}).get('timing')=='tail_fixed_delay_assumption' for r in results for d in r.decisions)
        rec['all_icmp_returned']=sum(r.cost['icmp_reply_count'] for r in results)
        rec['simulated_response_opportunities']=sum(r.cost['uplink_dummy_count'] for r in results)
        write_json(info,rec)
    return apps


def run(out,classes,prepare_only=False,*,capture_root=None,test_headers=None,config_path=None,dry_run=False,repeats=3):
    out=Path(out).resolve();artifacts=ROOT/'artifacts';legacy=artifacts/'legacy'
    if not out.is_relative_to(artifacts) or out.is_relative_to(legacy):
        raise ValueError('Outputs must be inside project artifacts and outside artifacts/legacy (read-only)')
    data,ids,clean,config=setup(out,classes,config_path,persist=not dry_run,repeats=repeats)
    if dry_run:
        print(json.dumps({'status':'passed','data_protocol':config['data_protocol'],'attacker_train':len(ids['train']),'test':len(ids['test']),
                          'proxy_train':config['proxy_train_count'],'proxy_validation':config['proxy_validation_count'],
                          'models':MODELS,'ratios':RATIOS,'repeats':repeats,'dataset_sha256':config['dataset_sha256'],'headers_checked':False}),flush=True)
        return
    y=np.asarray(data['labels'],int);train_y=y[ids['train']];test_y=y[ids['test']]
    cache=Cache(out,config)
    if prepare_only:
        for name in MODELS:
            for repeat in range(repeats):cache.fit(name,clean[ids['train']],train_y,42+repeat*10000)
        write_json(out/'clean_models_ready.json',{'count':len(MODELS)*repeats,'for_ratio':0,'no_clean_test_evaluation':True})
        return
    assert (out/'training/run_verification.json').exists(),'Formal training has not completed'
    apps=prepare_applications(out,data,ids,config,capture_root=capture_root,test_headers=test_headers)
    masks={r:engine.shared_tools.build_stratified_nested_masks(train_y,RATIOS,seed=731100+r*10000) for r in range(repeats)}
    np.savez_compressed(out/'retraining_masks.npz',**{f'repeat{r}_ratio{int(q*100)}':masks[r][q] for r in range(repeats) for q in RATIOS})
    rows=read_json(out/'observations.json') if (out/'observations.json').exists() else []
    state={'status':'running','completed_observations':len(rows),'total_observations':len(MODELS)*len(RATIOS)*repeats,'pid':os.getpid()}
    write_json(out/'evaluation_state.json',state)
    try:
        for q in RATIOS:
            for name in MODELS:
                for repeat in range(repeats):
                    if any(r['model']==name and r['ratio']==q and r['repeat']==repeat for r in rows):continue
                    x=clean[ids['train']].copy();x[masks[repeat][q]]=apps['train'][masks[repeat][q]]
                    seed=42+repeat*10000
                    scaler,model,key,record=cache.fit(name,x,train_y,seed)
                    predicted=model.predict(scaler.transform(apps['test']))
                    pp=out/'predictions'/f'{safe(name)}_ratio{int(q*100)}_repeat{repeat}.npy';pp.parent.mkdir(exist_ok=True);np.save(pp,predicted,allow_pickle=False)
                    row={'method':'Securitas-adapted-ICMP','classes':classes,'model':name,'ratio':q,'repeat':repeat,'seed':seed,
                         'accuracy':float(accuracy_score(test_y,predicted)),'balanced_accuracy':float(balanced_accuracy_score(test_y,predicted)),
                         'protected_train_flows':int(masks[repeat][q].sum()),'model_key':key,'model_sha256':record['model_sha256'],
                         'prediction_file':str(pp.relative_to(out)),'prediction_sha256':sha256(pp),'confusion_matrix':confusion_matrix(test_y,predicted,labels=np.arange(classes)).tolist()}
                    rows.append(row);write_json(out/'observations.json',rows)
                    state.update(completed_observations=len(rows),current_model=name,current_ratio=q,current_repeat=repeat);write_json(out/'evaluation_state.json',state)
                selected=[r for r in rows if r['model']==name and r['ratio']==q]
                unit={'classes':classes,'model':name,'ratio':q,'repeats':repeats,'BA_mean':float(np.mean([r['balanced_accuracy'] for r in selected])),
                      'BA_std':float(np.std([r['balanced_accuracy'] for r in selected],ddof=1)) if repeats > 1 else None,
                      'ACC_mean':float(np.mean([r['accuracy'] for r in selected])),
                      'ACC_std':float(np.std([r['accuracy'] for r in selected],ddof=1)) if repeats > 1 else None}
                write_json(out/'units'/f'{safe(name)}_{int(q*100)}.json',unit)
                print('UNIT_COMPLETE '+json.dumps(unit),flush=True)
        assert len(rows)==len(MODELS)*len(RATIOS)*repeats
        for p,h in config['source_sha256'].items():assert sha256(ROOT/p)==h
        state.update(status='complete',source_unchanged=True,completed_observations=len(rows));write_json(out/'evaluation_state.json',state)
    except BaseException as e:
        state.update(status='failed',error=repr(e));write_json(out/'evaluation_state.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--classes',type=int,choices=(6,8,10),required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,help='Explicit dataset config; required for CIC')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--capture-root',type=Path)
    parser.add_argument('--test-headers',type=Path)
    parser.add_argument('--prepare-clean-models',action='store_true')
    parser.add_argument('--wait-for-training',action='store_true')
    parser.add_argument('--repeats',type=int,default=3,help='Attacker evaluation repetitions (default: 3)')
    args=parser.parse_args();out=args.output or (ROOT/'artifacts/runs/securitas'/args.config.stem if args.config else DEFAULT_ROOT/str(args.classes))
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.dry_run and (args.wait_for_training or args.prepare_clean_models):
        parser.error('--dry-run cannot train or wait')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.wait_for_training:
            run(out,args.classes,prepare_only=True,config_path=args.config,repeats=args.repeats)
            write_json(out/'evaluation_state.json',{'status':'waiting_for_formal_strategy','classes':args.classes,'pid':os.getpid()})
            while not (out/'training/run_verification.json').exists():
                error=out/'training.err'
                if error.exists() and 'Traceback (most recent call last)' in error.read_text(encoding='utf-8',errors='replace'):
                    raise RuntimeError('Training failed; inspect training.err')
                time.sleep(10)
        run(out,args.classes,prepare_only=args.prepare_clean_models,capture_root=args.capture_root,test_headers=args.test_headers,config_path=args.config,dry_run=args.dry_run,repeats=args.repeats)

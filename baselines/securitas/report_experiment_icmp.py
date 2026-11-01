"""Verify and report this Securitas-only experiment without fitting models."""
from pathlib import Path
import csv
import json
import math
import sys
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from experiment_icmp import DEFAULT_ROOT,MODELS,RATIOS,ROOT,HERE,array_hash,engine,LSTM_PARAMS
from securitas_adapted.data import load_headers,flows_from_data
from securitas_adapted.core import apply_batch
from securitas_adapted.common import read_json,write_json,sha256,dataset_pcap
from securitas_adapted.policy import load_strategy
from threadpoolctl import threadpool_limits
import torch


class EventRows:
    """Index JSONL byte offsets without retaining all parsed flow events."""
    def __init__(self,path):
        self.path=Path(path)
        self.offsets=[]
        with self.path.open('rb') as stream:
            while True:
                offset=stream.tell()
                if not stream.readline():break
                self.offsets.append(offset)
        self.stream=self.path.open('rb')
        self.cached_index=None
        self.cached_row=None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self,index):
        if index!=self.cached_index:
            self.stream.seek(self.offsets[index])
            self.cached_row=json.loads(self.stream.readline())
            self.cached_index=index
        return self.cached_row

    def close(self):
        self.stream.close()


def verify_and_report(out,classes):
    state=read_json(out/'evaluation_state.json');assert state['status']=='complete'
    protocol=read_json(out/'evaluation_protocol.json')
    MODELS=tuple(protocol['models']);RATIOS=tuple(protocol['ratios'])
    repeats=int(protocol['repeats'])
    expected_count=len(MODELS)*len(RATIOS)*repeats
    rows=read_json(out/'observations.json');assert len(rows)==expected_count
    assert len({(r['model'],r['ratio'],r['repeat']) for r in rows})==expected_count
    strategy,c,meta=load_strategy(out/'training/strategy.json')
    assert c['simulate_icmp'] is True and meta['smoke_only'] is False
    assert c['icmp']==protocol['icmp'] and c['budget']==protocol['budget']
    dataset_path=ROOT/c['data']['dataset']
    data=dataset_pcap.load_dataset(dataset_path)
    assert sha256(dataset_path)==protocol['dataset_sha256']
    assert str(np.asarray(data['data_protocol']).item())==protocol['data_protocol']
    with np.load(out/'evaluation_split.npz',allow_pickle=False) as z:ids={k:z[k] for k in z.files}
    labels=np.asarray(data['labels'],int);groups=np.asarray(data['groups']).astype(str)
    assert len(ids['train'])+len(ids['test'])+len(ids.get('validation',()))==len(labels)
    from experiment_icmp import assert_current_partition
    assert_current_partition(data,ids,classes)
    assert not set(groups[ids['train']])&set(groups[ids['test']])
    assert set(meta['split']['consumed_train_indices']).isdisjoint(ids['test'])
    assert set(meta['split']['consumed_validation_indices']).isdisjoint(ids['test'])
    assert set(meta['split']['train_groups']).isdisjoint(meta['split']['validation_groups'])
    apps={}
    complete_event_count=0
    for part in ('train','test'):
        info=read_json(out/'applications'/f'{part}.json')
        file=out/'applications'/f'{part}.npz';assert sha256(file)==info['arrays_sha256']
        with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        apps[part]=a
        header_file=out/'training/headers.json' if part=='train' else out/'test_headers.json'
        header_payload=load_headers(header_file,protocol['dataset_sha256'])
        headers=header_payload['headers']
        for name,digest in info['audit_files_sha256'].items():assert sha256(out/'applications'/name)==digest
        event_path=out/'applications'/f'{part}_events.jsonl'
        event_rows=EventRows(event_path)
        assert len(event_rows)==len(ids[part])
        with event_path.open('r',encoding='utf-8') as stream:
            for start in range(0,len(ids[part]),256):
                batch_ids=ids[part][start:start+256]
                flows=flows_from_data(data,batch_ids,c,header_payload)
                replay=apply_batch(flows,strategy,c)
                for index,repeated in zip(batch_ids,replay):
                    original=json.loads(stream.readline())
                    assert original==json.loads(json.dumps({'flow_index':int(index),**repeated.to_dict()}))
            assert not stream.readline()
        np.testing.assert_array_equal(a['indices'],ids[part])
        expected=dataset_pcap.make_features(a['lengths'],a['gaps'],a['directions'],**protocol['feature'])
        np.testing.assert_array_equal(expected,a['features'])
        n=(a['roles']>=0).sum(1);complete_event_count+=int(n.sum())
        orig_bytes=np.sum(data['lengths'][ids[part]].astype(float)*data['mask'][ids[part]],axis=1)
        orig_time=np.sum(data['gaps'][ids[part]].astype(float)*data['mask'][ids[part]],axis=1)
        np.testing.assert_array_equal(orig_bytes,a['source_bytes'])
        np.testing.assert_allclose(orig_time,a['source_duration_seconds'],rtol=0,atol=1e-10)
        np.testing.assert_array_equal(a['full_lengths'].sum(1)-orig_bytes,a['added_bytes'])
        np.testing.assert_allclose(a['full_gaps'].sum(1)-orig_time,a['added_delay_seconds'],rtol=0,atol=1e-9)
        for metric in ('real_completion_delay_seconds','wire_completion_delay_seconds','cover_tail_seconds'):
            assert metric in a
        real_end=[];wire_end=[]
        for row_n,row_roles,row_times,row_source in zip(n,a['roles'],a['full_times'],a['source_duration_seconds']):
            roles=row_roles[:row_n];times=row_times[:row_n]
            real_times=times[np.isin(roles,[0,1])]
            real_end.append(float(np.max(real_times)) if real_times.size else float(row_source))
            wire_end.append(float(np.max(times)) if times.size else float(row_source))
        real_end=np.asarray(real_end);wire_end=np.asarray(wire_end)
        np.testing.assert_allclose(a['real_completion_delay_seconds'],np.maximum(0.,real_end-a['source_duration_seconds']),rtol=0,atol=1e-9)
        np.testing.assert_allclose(a['wire_completion_delay_seconds'],np.maximum(0.,wire_end-a['source_duration_seconds']),rtol=0,atol=1e-9)
        np.testing.assert_allclose(a['cover_tail_seconds'],np.maximum(0.,wire_end-real_end),rtol=0,atol=1e-9)
        np.testing.assert_allclose(a['added_delay_seconds'],a['wire_completion_delay_seconds'],rtol=0,atol=1e-9)
        np.testing.assert_array_equal(np.isin(a['roles'],[2,3]).sum(1),a['dummy_count'])
        np.testing.assert_array_equal((a['roles']==2).sum(1),a['uplink_dummy_count'])
        np.testing.assert_array_equal((a['roles']==3).sum(1),a['icmp_reply_count'])
        np.testing.assert_array_equal((a['roles']==1).sum(1),2*a['split_count'])
        np.testing.assert_array_equal(a['added_events'],a['dummy_count']+a['split_count'])
        np.testing.assert_allclose(a['local_timeline_shift_seconds'],a['local_added_events']*c['delta_seconds'],rtol=0,atol=1e-15)
        np.testing.assert_array_equal(a['local_added_events'],a['split_count']+a['uplink_dummy_count'])
        np.testing.assert_array_equal(a['budget_reserved_dummy_count'],2*a['uplink_dummy_count'])
        np.testing.assert_array_equal(a['budget_reserved_added_bytes'],a['added_bytes']+70*(a['uplink_dummy_count']-a['icmp_reply_count']))
        assert np.all(a['added_bytes']<=a['budget_reserved_added_bytes']) and np.max(a['budget_reserved_added_bytes'])<=c['budget']['bytes']
        assert np.max(a['budget_reserved_dummy_count'])<=c['budget']['dummy_packets']
        assert np.all(a['added_delay_seconds']<=a['budget_reserved_added_delay_seconds']+1e-12) and np.max(a['budget_reserved_added_delay_seconds'])<=c['budget']['delay_seconds']
        assert np.max(a['added_bytes'])<=c['budget']['bytes'] and np.max(a['added_delay_seconds'])<=c['budget']['delay_seconds'] and np.max(a['dummy_count'])<=c['budget']['dummy_packets']
        for i,index in enumerate(ids[part]):
            count=int(data['mask'][index].sum());roles=a['roles'][i,:n[i]];parents=a['parents'][i,:n[i]]
            assert list(dict.fromkeys(parents[np.isin(roles,[0,1])].tolist()))==list(range(count))
            assert np.all(a['full_gaps'][i,:n[i]]>=0) and np.all(np.diff(a['full_times'][i,:n[i]])>=0)
            source_times=np.cumsum(data['gaps'][index,:count].astype(float))
            local_rank=-1
            for j,parent in enumerate(parents):
                if roles[j]!=3:
                    local_rank+=1
                    expected_time=source_times[parent]+(local_rank-int(parent))*c['delta_seconds']
                    assert math.isclose(a['full_times'][i,j],expected_time,rel_tol=0,abs_tol=1e-9)
                else:
                    event=event_rows[i]['events'][j]
                    assert (event['ip_protocol'],event['icmp_type'],event['icmp_code'])==(1,11,0)
                    assert event['source_index'] is None and event['direction']==-1 and event['length']==70
                    assert event['header']['payload_bytes']==36
                    dummy_pos=np.flatnonzero((parents==parent)&(roles==2));assert len(dummy_pos)==1
                    dummy_time=a['full_times'][i,dummy_pos[0]]
                    d=next(x for x in event_rows[i]['decisions'] if x['source_index']==parent)
                    icmp=d['icmp'];assert d['status']=='executed' and d['operation']=='dummy' and icmp['status']=='generated'
                    assert 0<=icmp['response_quantile']<c['icmp']['response_probability']
                    later=[nxt for nxt in range(int(parent)+1,count) if data['directions'][index,nxt]==-1]
                    next_index=later[0] if later else None
                    assert next_index==icmp['next_inbound_source_index']
                    if next_index is not None:
                        inbound_pos=np.flatnonzero((parents==next_index)&(roles==0));assert len(inbound_pos)==1
                        end=a['full_times'][i,inbound_pos[0]]
                        expected_time=dummy_time+(end-dummy_time)*icmp['time_quantile']
                    else:
                        expected_time=dummy_time+c['icmp']['tail_delay_seconds']
                    assert math.isclose(a['full_times'][i,j],expected_time,rel_tol=0,abs_tol=1e-12)
                    assert a['full_times'][i,j]>=dummy_time
                if roles[j]==2:
                    assert a['full_directions'][i,j]==1 and data['directions'][index,parent]==1
                    assert a['full_lengths'][i,j]==max(c['protocol']['min_frame_bytes'],34+a['payload_bytes'][i,j])
            observed=min(c['window_length'],int(n[i]))
            np.testing.assert_array_equal(a['full_lengths'][i,:observed],a['lengths'][i,:observed])
            np.testing.assert_array_equal(a['full_gaps'][i,:observed],a['gaps'][i,:observed])
            for parent in range(count):
                pos=np.flatnonzero((parents==parent)&np.isin(roles,[0,1]))
                assert len(pos) in (1,2)
                assert np.all(a['full_directions'][i,pos]==data['directions'][index,parent])
                if len(pos)==2:
                    assert a['payload_offsets'][i,pos[0]]==0
                    assert a['payload_offsets'][i,pos[1]]==a['payload_bytes'][i,pos[0]]
                    assert a['payload_bytes'][i,pos[0]]%8==0
                    h=headers[f'{index}:{parent}']
                    assert a['payload_bytes'][i,pos].sum()==h['payload_bytes']
                    assert not h['dont_fragment'] and not h['already_fragmented']
                    for j in pos:
                        assert a['full_lengths'][i,j]==max(c['protocol']['min_frame_bytes'],h['l2_bytes']+h['ip_header_bytes']+a['payload_bytes'][i,j])
        event_rows.close()
    clean=dataset_pcap.make_features(*(data[k] for k in ('lengths','gaps','directions')),**protocol['feature'])
    reused=0
    with np.load(out/'retraining_masks.npz',allow_pickle=False) as masks:
        for repeat in range(repeats):
            expected=engine.method32_tools.build_stratified_nested_masks(labels[ids['train']],RATIOS,seed=731100+repeat*10000)
            for q in RATIOS:np.testing.assert_array_equal(masks[f'repeat{repeat}_ratio{int(q*100)}'],expected[q])
        for row in rows:
            p=out/'model_cache'/(row['model_key']+'.joblib');mr=read_json(p.with_suffix('.json'))
            assert sha256(p)==row['model_sha256']==mr['model_sha256']
            x=clean[ids['train']].copy();mask=masks[f'repeat{row["repeat"]}_ratio{int(row["ratio"]*100)}']
            x[mask]=apps['train']['features'][mask]
            assert array_hash(x)==mr['descriptor']['train_features_sha256']
            assert array_hash(labels[ids['train']])==mr['descriptor']['train_labels_sha256']
            assert mr['descriptor']['source_sha256']==protocol['source_sha256']
            assert all(mr['descriptor'][key] for key in ('python','numpy','sklearn','torch'))
            scaler,model=joblib.load(p);expected=StandardScaler().fit(x)
            np.testing.assert_array_equal(scaler.mean_,expected.mean_);np.testing.assert_array_equal(scaler.scale_,expected.scale_)
            predicted=model.predict(scaler.transform(apps['test']['features']))
            pf=out/row['prediction_file'];assert sha256(pf)==row['prediction_sha256']
            np.testing.assert_array_equal(predicted,np.load(pf,allow_pickle=False))
            assert balanced_accuracy_score(labels[ids['test']],predicted)==row['balanced_accuracy']
            assert accuracy_score(labels[ids['test']],predicted)==row['accuracy']
            np.testing.assert_array_equal(confusion_matrix(labels[ids['test']],predicted,labels=np.arange(classes)),row['confusion_matrix'])
            reused+=mr['origin']['kind']=='verified_identical_training_cache'
    for p,h in protocol['source_sha256'].items():assert sha256(ROOT/p)==h
    # Recompute byte overhead from the verified integer arrays. The training
    # writer historically evaluated 100 * np.sum(int32), which can overflow
    # before converting the result to float on Windows/NumPy.
    for part in ('train','test'):
        info_path=out/'applications'/f'{part}.json'
        info=read_json(info_path)
        source_bytes=int(np.sum(apps[part]['source_bytes'],dtype=np.uint64))
        added_bytes=int(np.sum(apps[part]['added_bytes'],dtype=np.uint64))
        if source_bytes<=0:
            raise ValueError(f'Invalid total source bytes for {part}: {source_bytes}')
        info['overhead']['bytes_overhead_percent']=float(100.0*added_bytes/source_bytes)
        write_json(info_path,info)
    overhead=read_json(out/'applications/test.json')['overhead']
    summary=[]
    for name in MODELS:
        for q in RATIOS:
            rs=[r for r in rows if r['model']==name and r['ratio']==q];assert len(rs)==repeats
            summary.append({'classes':classes,'model':name,'ratio':q,'BA_mean':float(np.mean([r['balanced_accuracy'] for r in rs])),
                            'BA_std':float(np.std([r['balanced_accuracy'] for r in rs],ddof=1)) if repeats > 1 else None,
                            'ACC_mean':float(np.mean([r['accuracy'] for r in rs])),
                            'ACC_std':float(np.std([r['accuracy'] for r in rs],ddof=1)) if repeats > 1 else None})
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (out/'observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=[k for k in rows[0] if k!='confusion_matrix'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write_json(out/'summary.json',summary)
    units={(r['model'],r['ratio']):r for r in summary}
    if c['budget']=={'bytes':630,'delay_seconds':3.5,'dummy_packets':6}:
        budget_name='低预算'
    elif c['budget']=={'bytes':3000,'delay_seconds':7.0,'dummy_packets':32}:
        budget_name='高预算'
    else:
        budget_name='自定义预算'
    def metric(unit,key):
        mean=f'{unit[key+"_mean"]*100:.2f}'
        std=unit[key+'_std']
        return mean if std is None else f'{mean} ± {std*100:.2f}'
    text=[f'# Securitas-adapted-ICMP {classes}分类{budget_name}实验','',
          '仅评估开启ICMP的Securitas-adapted；数据协议见 evaluation_protocol.json，BA/ACC越低表示越能阻碍当前识别器。',
          '', '## 识别效果（%）','',
          f'{repeats}次评测；重复评测改变攻击者模型种子与受保护训练流掩码。通用防御策略只训练一次，固定ICMP环境种子，训练/测试使用同一动作表。样本标准差仅在至少两次评测时显示。',
          '', '| 识别器 | '+' | '.join(f'{q:g}% BA' for q in np.array(RATIOS)*100)+' |','|---|'+'---:|'*len(RATIOS)]
    for name in MODELS:text.append('| '+name+' | '+' | '.join(metric(units[name,q],'BA') for q in RATIOS)+' |')
    text+=['','| 识别器 | '+' | '.join(f'{q:g}% ACC' for q in np.array(RATIOS)*100)+' |','|---|'+'---:|'*len(RATIOS)]
    for name in MODELS:text.append('| '+name+' | '+' | '.join(metric(units[name,q],'ACC') for q in RATIOS)+' |')
    test_info=read_json(out/'applications/test.json')
    text+=['','## 完整流实际开销','',f'基于{len(ids["test"])}条测试流，包含窗口外事件。固定策略与ICMP环境在识别器/比例/重复间共用，开销不当作多份独立样本。','',
           f'- 平均新增字节：{overhead["added_bytes"]["mean"]:.6f} B（新增总字节÷原始总字节：{overhead["bytes_overhead_percent"]:.6f}%）。',
           f'- 平均真实事件完成时延：{overhead["real_completion_delay_seconds"]["mean"]:.12f} s（比例：{overhead["real_completion_seconds_overhead_percent"]:.9f}%）。',
           f'- 平均线缆事件完成时延：{overhead["wire_completion_delay_seconds"]["mean"]:.12f} s（比例：{overhead["wire_completion_seconds_overhead_percent"]:.9f}%）；覆盖尾部：{overhead["cover_tail_seconds"]["mean"]:.12f} s（比例：{overhead["cover_tail_seconds_overhead_percent"]:.9f}%）。旧字段 `added_delay_seconds` 保留为线缆完成时延兼容别名。',
           f'- 每流平均上行dummy：{overhead["uplink_dummy_count"]["mean"]:.6f}；平均下行ICMP：{overhead["icmp_reply_count"]["mean"]:.6f}；虚假事件总数：{overhead["dummy_count"]["mean"]:.6f}。',
           f'- 每流平均ICMP新增字节：{overhead["icmp_added_bytes"]["mean"]:.6f} B；平均拆分次数：{overhead["split_count"]["mean"]:.6f}；平均总新增事件：{overhead["added_events"]["mean"]:.6f}。',
           f'- 最大新增字节/线缆完成时延/覆盖尾部/虚假事件数：{overhead["added_bytes"]["max"]:g} B / {overhead["wire_completion_delay_seconds"]["max"]:.12f} s / {overhead["cover_tail_seconds"]["max"]:.12f} s / {overhead["dummy_count"]["max"]:g}。',
           f'- 测试集中ICMP返回总数：{test_info["all_icmp_returned"]}，其中没有后续真实下行而使用尾部时延假设的响应：{test_info["tail_icmp_events"]}。',
           '', '预算：'+json.dumps(c['budget'],ensure_ascii=False)+'。dummy上限计上行dummy及ICMP，不包含真实片段；潜在响应预算在发送前预留。上面是实际新增量，预留量另见applications/test.json，不能混同。',
           '', '附加时延按完整流结束时间增量计算：本地分片/插包使后续源事件按1微秒累计后移，ICMP按返回时刻合并，只有超过原扰动流尾部才额外延长。这里是事件仿真，不是真实网络测量。',
           '', '## 协议与限制','',
           f'- 数据共{len(labels)}条流、{len(data["device_names"])}类，逐类数量为{np.bincount(labels).tolist()}；外层训练/测试流数：{len(ids["train"])}/{len(ids["test"])}, 按数据集 capture 组切分且组集合无交集，最终测试不参与代理或策略选择。',
           f'- 代理MLP干净训练{c["proxy"]["epochs"]}轮，策略LSTM/REINFORCE训练{c["policy"]["epochs"]}轮，每轮{c["policy"]["validation_candidates"]}个验证候选；在完整内部验证集按BA、实际新增字节、候选编号选择固定动作表。',
            '- 识别器使用记录中的训练配置、参数及种子；前32个事件的64维方向长度/log-gap特征，攻击者观察首gap固定为零，完整物理事件仍保留首包等待并计入成本。记录中的 ratios 为攻击者固定训练规模中受保护流比例。',
           '- ICMP环境：'+json.dumps(c['icmp'],ensure_ascii=False)+'；概率0.48与区间来自原论文假设，尾部10ms为显式适配假设；没有真实网络测量或服务器因果配对。',
           '- 使用PCAP真实报头，完整IPv4分片成本；仅实际插入的dummy触发ICMP，响应70B。不是原作者系统完整部署复现。',
           f'- {expected_count}条模型评估；模型缓存只允许来自当前输出目录并严格匹配训练输入/标签/源码与运行环境，本次记录中复用{reused}份，其余{expected_count-reused}份新拟合；不会继承 artifacts/legacy。未运行其他防御方法，也没有无防御测试评估。',
           '', '## 交付文件','',
           '- summary.csv/json与observations.csv/json：汇总和逐次结果。',
           '- training/：代理、策略、配置、capture划分、训练轨迹；applications/：完整事件、动作、开销与窗口特征。',
           '- model_cache/、predictions/：模型与逐流预测；verification.json：完整事件重放、预测重放、Scaler与预算核验。']
    (out/'RESULTS_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    verification={'status':'passed','prediction_replays':len(rows),'training_input_and_scaler_checks':len(rows),'complete_event_count_checked':complete_event_count,
                  'capture_leakage':False,'budgets_passed':True,'source_unchanged':True,'reused_identical_models':reused,'new_models':expected_count-reused,'other_methods_evaluated':False,'icmp_event_replay_passed':True,'simulation_config':c['icmp']}
    write_json(out/'verification.json',verification)
    print(json.dumps({'summary':summary,'overhead':overhead,'verification':verification}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--classes',type=int,choices=(6,8,10),required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):verify_and_report(args.output or DEFAULT_ROOT/str(args.classes),args.classes)

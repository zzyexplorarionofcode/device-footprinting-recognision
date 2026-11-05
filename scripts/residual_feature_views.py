"""Observable residual views under frozen Full New5; no provenance input."""
import hashlib
import json
import numpy as np
from scripts.online_nonstationary import evaluate, fit_lr, summarize, draw_states
from scripts.deakin_external_codebook import fit_rf
from scripts.paired_mapping_policies import render
from pathlib import Path


def views(x):
    signed = x[:, :32] * 1514
    length = np.abs(signed)
    valid = signed != 0
    down = signed < 0
    large = length > 256
    bins = np.linspace(0, 1514, 17)
    def hist(selection):
        return np.asarray([np.histogram(row[sel], bins=bins)[0] / max(1, sel.sum())
                           for row, sel in zip(length, selection)])
    def stats(selection):
        out=[]
        for row,sel in zip(length,selection):
            z=row[sel]
            out.append([z.sum(), z.mean(), z.std(), z.min(), z.max()] if len(z) else [0]*5)
        return np.asarray(out)
    a=np.c_[hist(down),down.sum(axis=1)]
    b=np.c_[hist(large),valid.sum(axis=1)]
    c=np.c_[a,b,stats(down),stats(large),large.sum(axis=1)]
    return {'A':a,'B':b,'C':c}


def run(family,seed,data,project,mappings,warm,warm_x,ids,days,local,ends,uniform,candidates,cost_reference,manifest_path,digest):
    root=Path(__file__).resolve().parents[1]
    baseline=json.loads((root/f'artifacts/es/state_count_{family}_s{seed}.json').read_bytes())
    states=draw_states(uniform[ids],np.full(6,1/6))
    parts=[]
    for stage,candidate in enumerate(candidates):
        lo=local[stage*5][0][0]
        hi=local[(stage+1)*5][0][0] if (stage+1)*5<len(days) else len(ids)
        x,cost=render(data,ids[lo:hi],states[lo:hi],mappings,candidate)
        assert np.allclose(cost,cost_reference[lo:hi],rtol=0,atol=1e-8)
        parts.append(x)
    x=np.concatenate(parts)
    vx,vw=views(x),views(warm_x)
    shuffled=x.copy()
    rng=np.random.default_rng(8123)
    for row in shuffled:
        order=rng.permutation(32)
        row[:32]=row[:32][order]
        row[32:]=row[32:][order]
    for name,value in views(shuffled).items():
        assert np.allclose(value,vx[name])
    output=root/f'artifacts/es/residual_views_{family}_s{seed}.json'
    report={'family':family,'seed':seed,'dataset_sha256':digest,
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'definitions':{'A':'16-bin normalized downlink length histogram + downlink count',
                           'B':'16-bin normalized >256B histogram + total valid count',
                           'C':'A+B + sum/mean/std/min/max of downlink and >256B lengths + >256B count',
                           'D':'original signed length and log gap 64D'},
            'results':{'D':baseline['results']['6']}}
    for view in vx:
        report['results'][view]={}
        for model,fit in [('lr',fit_lr),('rf',fit_rf)]:
            rows=evaluate(vx[view],data['labels'][ids],days,local,ends[warm],data['labels'][warm],vw[view],{},regime_days=5,fit_model=fit)
            ref=baseline['results']['6'][model]['days']
            for r,b in zip(rows,ref):
                for key in ('day','blind_support','labels_available'):
                    assert r[key]==b[key]
                for attack in ('cumulative','sliding'):
                    assert r['attacks'][attack]['labels_used']==b['attacks'][attack]['labels_used']
            result={'days':rows,'summary':summarize(rows),'dimensions':vx[view].shape[1]}
            report['results'][view][model]=result
            output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(family,seed,view,model,{a:v['mean_post_ba'] for a,v in result['summary'].items()},flush=True)

"""Fixed five-event T2_W1 vs existing T1_W3 Full New5 online pilot."""
import json
from pathlib import Path
import numpy as np
from scripts.two_trigger_pairs import render_two
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import draw_states,evaluate,fit_lr,summarize
from scripts.deakin_external_codebook import fit_rf
from scripts.residual_feature_views import views


def run(family,seed,data,project,mappings,warm,warm_x,ids,days,local,ends,uniform,candidates):
    root=Path(__file__).resolve().parents[1]
    states=draw_states(uniform[ids],np.full(6,1/6))
    # Common pre-deployment T1_W3 warm history, as in original Full New5.
    parts=[];costs=[];audits=[]
    for stage,candidate in enumerate(candidates):
        lo=local[stage*5][0][0];hi=local[(stage+1)*5][0][0] if (stage+1)*5<len(days) else len(ids)
        second=generated_project(project,300000+1000*seed+stage)
        x,c,a=render_two(data,ids[lo:hi],states[lo:hi],mappings,candidate,second)
        parts.append(x);costs.append(c);audits.append(a)
    x=np.concatenate(parts);cost=np.concatenate(costs);audit=np.concatenate(audits)
    output=root/f'artifacts/es/two_pairs_{family}_s{seed}.json'
    report={'family':family,'seed':seed,'definition':'T2=two pairs four packets; W1=one60B dummy',
            'warm_history':'common original T1_W3 six-State warmup',
            'budgets':{'trigger':[566,.9,4],'weakener':[64,.6,1]},
            'cost_mean':cost.mean(axis=0).tolist(),'cost_max':cost.max(axis=0).tolist(),
            'real_padding_delay_mean':audit.mean(axis=0).tolist(),'results':{}}
    for view,features,wfeatures in [('D',x,warm_x),('C',views(x)['C'],views(warm_x)['C'])]:
        report['results'][view]={}
        for model,fit in [('lr',fit_lr),('rf',fit_rf)]:
            rows=evaluate(features,data['labels'][ids],days,local,ends[warm],data['labels'][warm],wfeatures,{},regime_days=5,fit_model=fit)
            report['results'][view][model]={'days':rows,'summary':summarize(rows)}
            output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(family,seed,view,model,{a:v['mean_post_ba'] for a,v in summarize(rows).items()},flush=True)

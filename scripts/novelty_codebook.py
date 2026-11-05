"""History-distance candidate selection with no attack feedback."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.deakin_external_codebook import fit_rf
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import ATTACKS, balanced_accuracy, draw_states, evaluate, fit_lr, summarize
from scripts.paired_mapping_policies import render


POOL_SIZE = 4
HISTORY_SIZE = 3


def distance(a, b):
    pa, pb = (np.asarray(p.method33.trigger_positions, dtype=float) / 16 for p in (a, b))
    ca, cb = (np.asarray(p.method33.trigger_codes, dtype=float) for p in (a, b))
    terms = [np.mean((pa-pb)**2)]
    for column, budget in ((0, a.method33.trigger_budget.bytes), (1, a.method33.trigger_budget.delay_seconds)):
        terms.append(np.mean(((ca[:,:,column]-cb[:,:,column])/budget)**2))
    return float(np.sqrt(np.mean(terms)))


def run_novelty(family, repeat, data, project, mappings, warm, warm_x, ids, days, local,
                dates, uniform, baseline_seeds, source, cost_reference, manifest_path, dataset_path):
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts/es"
    label = data["labels"]
    maps = mappings[:6]
    states = draw_states(uniform[ids], np.full(6,1/6))
    pools = [(first, *(100000+10000*repeat+10*stage+j for j in range(1,POOL_SIZE)))
             for stage,first in enumerate(baseline_seeds)]
    configs = {s: generated_project(project,s) for pool in pools for s in pool}
    histories = {"Full": [], "Novelty": []}
    decisions = {"Full": [], "Novelty": []}
    selections = {"Full": [], "Novelty": []}
    for pool in pools:
        for policy in histories:
            past = histories[policy][-HISTORY_SIZE:]
            scores = [min(distance(configs[s],p) for p in past) if past else None for s in pool]
            chosen = max(range(POOL_SIZE),key=lambda j:(scores[j],-j)) if past and policy=="Novelty" else 0
            chosen_seed=pool[chosen]
            decisions[policy].append({"pool":pool,"scores":scores,"chosen_seed":chosen_seed,
                                      "distance":scores[chosen],"first_candidate_distance":scores[0]})
            selections[policy].append(chosen_seed)
            histories[policy].append(configs[chosen_seed])
    assert selections["Full"]==baseline_seeds
    if family=="unsw8":
        frozen=json.loads((artifacts/f"state_count_unsw8_s{repeat}.json").read_bytes())
        source["rf"]=frozen["results"]["6"]["rf"]
    results={}
    baseline_features=None
    for policy in ("Full","Novelty"):
        parts=[]
        for stage,s in enumerate(selections[policy]):
            lo=local[stage*5][0][0]
            hi=local[(stage+1)*5][0][0] if (stage+1)*5<len(days) else len(ids)
            part,cost=render(data,ids[lo:hi],states[lo:hi],mappings,configs[s])
            assert np.allclose(cost,cost_reference[lo:hi],rtol=0,atol=1e-8)
            parts.append(part)
        x=np.concatenate(parts)
        if policy=="Full":
            baseline_features=x[:local[5][0][0]].copy()
        else:
            assert np.array_equal(x[:local[5][0][0]],baseline_features)
        controls={}
        for i in range(5,len(days),5):
            blind=local[i][1]
            previous=selections[policy][i//5-1]
            controls[i],_=render(data,ids[blind],states[blind],mappings,configs[previous])
        results[policy]={}
        scaler=StandardScaler().fit(warm_x)
        sx=scaler.transform(x)
        for model,fit in (("lr",fit_lr),("rf",fit_rf)):
            if policy=="Full":
                rows=json.loads(json.dumps(source[model]["days"]))
                # Add only counterfactual switch diagnostics; full-horizon scores are reused.
                for i in range(5,len(days),5):
                    previous=np.concatenate([local[j][0] for j in range(i)])
                    hx=np.concatenate((warm_x,x[previous]))
                    hy=np.r_[label[warm],label[ids[previous]]]
                    hd=np.r_[dates[warm],np.repeat(days[:i],8)]
                    assert np.all(hd<days[i])
                    for attack in ATTACKS:
                        eligible=np.ones(len(hy),dtype=bool) if attack=="cumulative" else hd>=days[i]-np.timedelta64(14,"D")
                        clf=fit(scaler.transform(hx[eligible]),hy[eligible]) if len(np.unique(hy[eligible]))>=2 else fit(scaler.transform(warm_x),label[warm])
                        blind=local[i][1]
                        current=balanced_accuracy(clf,sx[blind],label[ids[blind]])
                        assert np.isclose(current,rows[i]["attacks"][attack]["pre"])
                        rows[i]["attacks"][attack]["pre_under_previous_mix"]=balanced_accuracy(clf,scaler.transform(controls[i]),label[ids[blind]])
            else:
                rows=evaluate(x,label[ids],days,local,dates[warm],label[warm],warm_x,controls,
                              regime_days=5,fit_model=fit)
            results[policy][model]={"days":rows,"summary":summarize(rows)}
            print(family,repeat,policy,model,{a:results[policy][model]["summary"][a]["mean_post_ba"] for a in ATTACKS},flush=True)
    result={"family":family,"repeat":repeat,"pool_size":POOL_SIZE,"history_size":HISTORY_SIZE,
            "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "dataset":str(dataset_path),"decisions":decisions,"selections":selections,
            "candidate_parameters":{s:{"positions":p.method33.trigger_positions,"codes":p.method33.trigger_codes}
                                    for s,p in configs.items()},"results":results}
    (artifacts/f"novelty_new5_{family}_s{repeat}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

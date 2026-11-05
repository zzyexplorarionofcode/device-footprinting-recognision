"""Experimental two-Trigger-pair render, isolated from production jitter wrapper."""
from dataclasses import replace
import numpy as np
from methods.method33 import (Budget,apply_method33_base,jitter_trigger_code,
                              _flow_token,_flow_jitter_quantiles)
from scripts.fold_review import take
from scripts.position_placebo import features


def render_two(data,ids,states,mappings,project,second_project):
    base=project.method33
    # Independent second pair, same generated legal space. Sort anchors and carry packets together.
    positions=[];codes=[];pair_ids=[]
    for p,q,c,d in zip(base.trigger_positions,second_project.method33.trigger_positions,
                       base.trigger_codes,second_project.method33.trigger_codes):
        # Four distinct anchors, preserve first pair, move conflicting second anchors deterministically.
        used=set(p); extra=[]
        for anchor in q:
            available=[a for a in range(1,17) if a not in used]
            chosen=min(available,key=lambda a:(abs(a-anchor),a));used.add(chosen);extra.append(chosen)
        entries=sorted([(a,packet,0) for a,packet in zip(p,c)]+[(a,packet,1) for a,packet in zip(extra,d)])
        positions.append(tuple(e[0] for e in entries));codes.append(tuple(e[1] for e in entries));pair_ids.append([e[2] for e in entries])
    weak=replace(base.weakener,dummy_length_targets=(60.,),dummy_length_probabilities=(1.,))
    cfg=replace(base,trigger_packet_count=4,weakener_dummy_count=1,
                trigger_budget=Budget(566.,.9,4),weakener_budget=Budget(64.,.6,1),
                weakener=weak,trigger_codes=tuple(codes),trigger_positions=tuple(positions))
    output=np.empty((len(ids),64));costs=[];audits=[]
    for state in np.unique(states):
        rows=np.flatnonzero(states==state)
        lengths,gaps,directions,mask,labels=take(data,ids[rows]);lengths=np.minimum(lengths,1514)
        app=apply_method33_base(lengths,gaps,directions,mask,labels,cfg,731800,
                               mapping=mappings[state],include_event_trace=True)
        for i,row in enumerate(app[-1]):
            code=row['trigger_code_index'];events=[e for e in row['complete_event_trace'] if e['role']=='trigger_dummy']
            assert len(events)==4
            quantiles=_flow_jitter_quantiles(731800,_flow_token(lengths,gaps,directions,mask,i))
            for pair in (0,1):
                indices=[j for j,g in enumerate(pair_ids[code]) if g==pair]
                selected=tuple(codes[code][j] for j in indices)
                paircfg=replace(base,trigger_codes=(selected,))
                jitter=jitter_trigger_code(selected,paircfg,project.trigger_jitter,
                    length_uniform_quantile=quantiles[0],gap_uniform_quantile=quantiles[1])
                for j,packet in zip(indices,jitter.packets):
                    e=events[j]
                    if e['in_observation_window']:
                        pos=e['event_position']-1;app[0][i,pos]=packet[0];app[1][i,pos]=packet[1]
            assert row['dummy_packets']==5 and row['all_per_flow_hard_budgets_satisfied']
            assert row['total_added_bytes']<=630+1e-9 and row['total_added_delay']<=1.5+1e-9
            audits.append([row.get(k,0) for k in ('weakener_real_padding_bytes','weakener_real_added_delay')])
        output[rows]=features(app)
        costs.extend([[r['total_added_bytes'],r['total_added_delay'],r['dummy_packets']] for r in app[-1]])
    return output, np.asarray(costs),np.asarray(audits)

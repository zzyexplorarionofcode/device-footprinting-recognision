from pathlib import Path
import json,re
import numpy as np
root=Path(__file__).resolve().parents[1]
lines=['# Full New5 可观测残余特征诊断','', '防御冻结为 K=6、均匀逐流 IID、每五观测日更新合法完整码本。复用相同原始流、标签到达、Blind 与三个配对 Seed；累计与14日滑动 LR/RF 参数不变。D 复用原结果。UNSW8 为150日开发回放，Deakin10为50日跨数据集开发验证。','', 'A：下行包长16箱归一化直方图与数量（17维）。B：>256 B事件16箱归一化直方图与有效事件总数（17维）。C：A+B，加入下行及大包长度的和、均值、标准差、最小、最大，以及大包数（45维）。所有视图只从实际可见的64维长度/方向输入重建，不读取role、真实包索引、State或标签；训练与评分均使用同一视图。已检查事件任意重排后A–C不变。','', '| 数据 | 模型 | 更新 | A | B | C | D | C−D |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
rows=[]
for family in ('unsw8','deakin10'):
    results=[json.loads((root/f'artifacts/es/residual_views_{family}_s{s}.json').read_bytes()) for s in range(3)]
    for model in ('lr','rf'):
        for attack in ('cumulative','sliding'):
            values={v:[r['results'][v][model]['summary'][attack]['mean_post_ba'] for r in results] for v in 'ABCD'}
            means={v:float(np.mean(z)) for v,z in values.items()}
            lines.append('| '+ ' | '.join([family,model,attack]+[f'{means[v]:.4f}' for v in 'ABCD']+[f"{means['C']-means['D']:+.4f}"])+' |')
            rows.append((family,model,attack,means,values))
lines+=['','## 配对重复差值','', '| 数据 | 模型 | 更新 | C−D（三个Seed） |','| --- | --- | --- | --- |']
for family,model,attack,means,values in rows:
    lines.append('| '+' | '.join([family,model,attack,' / '.join(f'{c-d:+.4f}' for c,d in zip(values['C'],values['D']))])+' |')
lines+=['','## 判读边界','','A–C的表现测量可观测信息的充分性，不测量原完整模型实际使用这些信息的必要性或因果贡献。C优于D时说明所测有限标签训练下，去除顺序和时序的表示可能更易学习，D不构成理论上限。A只包含下行事件；B的大包可能经过上行padding；C同时含真实流量残余与可观测规模，不能称为纯真实包视图。窗口截断仍可能改变下行覆盖，短流数量统计也可能包含采集行为。当前未单独测试时序残余。','','结果支持程度应按数据集、模型和历史使用规则分别判断，不能用C/D解释贡献百分比。新的归因结果不证明当前字节预算下不存在更有效防御，也不能从观察充分性直接推导最优防御。','', '代码：scripts/residual_feature_views.py；复现：python -X utf8 -B -m scripts.state_count_ablation --family unsw8（或deakin10） --seed 0（或1/2） --residual。逐日结果：artifacts/es/residual_views_{family}_s{seed}.json。']
(root/'docs/FullNew5残余特征归因.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=root/'docs/项目汇报.html'
s=p.read_bytes().decode('utf-8'); oldstyle=re.search(r'<style>.*?</style>',s,re.S).group()
html='  <h3>Full New5 下的可观测残余信息</h3>\n  <p>冻结防御与在线标签账本，仅切换攻击者特征。A为下行包长直方图与数量，B为大包直方图与有效事件数，C为两者及长度汇总统计，D为原64维输入。A–C均通过顺序打乱不变性检查，不读取内部真实包标记。</p>\n  <table><tr><th>数据 / 攻击</th><th>A</th><th>B</th><th>C</th><th>D</th></tr>\n'
for family,model,attack,means,values in rows:
    html+='<tr><td>'+family+' / '+model+' / '+attack+'</td>'+''.join(f'<td class="num">{means[v]:.4f}</td>' for v in 'ABCD')+'</tr>\n'
html+='</table>\n<p>这些结果衡量可观测残余特征能否支持识别，不构成原完整模型的因果归因。完整输入也不是有限样本学习的理论上限。数据均属于已见开发回放。<a href="FullNew5残余特征归因.md">协议、配对差值与边界</a></p>\n'
anchor='  <h3>新码本怎样改变历史标签的价值？</h3>'
assert s.count(anchor)==1
s=s.replace(anchor,html+anchor)
assert re.search(r'<style>.*?</style>',s,re.S).group()==oldstyle
p.write_bytes(s.encode('utf-8'))
with (root/'agent.md').open('a',encoding='utf-8') as f:f.write('\n- Full New5可观测残余特征四视图诊断已完成，两数据集三Seed，原完整模型结果复用。见docs/FullNew5残余特征归因.md。仅支持信息充分性，不能直接推导原模型因果贡献。\n')
print('\n'.join(lines[:18]))

"""Reorder the existing report for an advisor-facing web presentation.

The experimental tables remain in a collapsible archive. This script only edits
the narrative and layout; it does not recalculate research results.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "项目汇报.html"
PAGE = ROOT / "docs" / "index.html"


def section_map(document: str) -> dict[str, str]:
    return {
        match.group(1): match.group(0)
        for match in re.finditer(r'<section id="([^"]+)">.*?</section>', document, re.S)
    }


source = REPORT.read_text(encoding="utf-8")
sections = section_map(source)
expected = {
    "summary", "roadmap", "protocol", "early", "mapping", "dynamic", "history",
    "learning", "refresh", "controls", "lower", "residual", "conclusion",
    "audit", "reproduce",
}
if set(sections) != expected:
    if {"summary", "method", "core", "residual", "decision"}.issubset(sections):
        PAGE.write_text(source.replace('.md"', '.html"'), encoding="utf-8")
        raise SystemExit(0)
    raise RuntimeError(f"Unexpected report sections: {set(sections) ^ expected}")

early = sections["early"]
branch_match = re.search(r'<div id="branch">.*?</div>\s*(?=<h3>是 trigger code)', early, re.S)
if not branch_match:
    raise RuntimeError("Could not isolate the T2_W1 branch")
branch = branch_match.group(0).replace('<div id="branch">', '<section id="branch">', 1)
branch = re.sub(r'</div>\s*$', '</section>', branch)
branch = branch.replace('<h3>T2_W1与T1_W3：固定五事件的后续在线对照</h3>', '<h2>附录：T2_W1 与 T1_W3 后续支线</h2>')
early = early[:branch_match.start()] + early[branch_match.end():]
early = early.replace(
    '<b>关键否证：</b>类特定 code 与 mapping rotation 的贡献很小。T1 与 T0 的巨大差异不能归因于“code 编码”。',
    '<b>对照边界：</b>Real 与 Placebo 保留相同的类相关位置投放，联合改变类别 code 内容及其映射；该联合差异较小。它不能估计位置映射切换的贡献，后者由下方无轮换对照单独检验。T1 与 T0 的差异也不能单独归因于 code。',
)
early = early.replace(
    '同样 2 个 trigger 事件、同样位置和预算，但所有类别共享同一个 code，没有 mapping rotation。',
    '同样 2 个 trigger 事件、同样类相关位置和预算，但所有类别共享同一个长度 code。原记录中的“无 mapping rotation”针对 code 内容映射；位置投放仍被保留。',
)
early = early.replace('既然不是 code，我们继续拆位置。', '由于 code 内容的联合差异较小，我们继续单独控制类—位置关联与位置映射切换。')
sections["early"] = early

roadmap = sections["roadmap"].replace(
    '下表按实际研究推进的先后排列；后文按相同逻辑展开。',
    '下表按实际研究推进的先后排列；以下实验档案沿用这一顺序。',
).replace('T2支线按主题并入三', 'T2支线独立置于附录')
sections["roadmap"] = roadmap

conclusion = sections["conclusion"].replace('可以提交审阅的主张、边界与冻结决定', '研究主张、证据边界与冻结决定')
conclusion = conclusion.replace('13.1 当前阶段表述', '当前阶段表述').replace('13.2 审稿限制与尚未完成的检验', '证据限制与尚未完成的检验').replace('13.3 冻结与后续研究边界', '冻结与后续研究边界')
sections["conclusion"] = conclusion

summary = '''<section id="summary" class="opening">
<h2>研究概览与主要发现</h2>
<p class="lead">核心问题：在相同通信预算与逐日标签到达规则下，持续刷新合法扰动分布，能否让持续学习的设备指纹攻击者反复付出适应成本？已有静态扰动主要检验一次变换后的识别下降；这里进一步评价攻击者持续更新后的全期 Post BA。</p>
<div class="result-grid" role="table" aria-label="四项核心研究判断">
  <div class="result-head" role="row"><span>研究维度</span><span>代表性结果</span><span>解释边界</span></div>
  <div class="result-row" role="row"><strong>动态刷新与在线适应</strong><b>0.8054 → 0.7683</b><span>UNSW8 累计 LR；Full New5 相对固定码本 + 逐流 IID</span></div>
  <div class="result-row" role="row"><strong>跨数据集与模型观察</strong><b>0.6295 → 0.5730</b><span>Deakin10 累计 RF；该攻击设置下观察到收益</span></div>
  <div class="result-row" role="row"><strong>残余可识别信息</strong><b>BA 0.9647</b><span>UNSW8 无序可观测视图 C，仍保留很强的设备识别信号</span></div>
  <div class="result-row" role="row"><strong>学习型策略的增量</strong><b>尚无稳定增量</b><span>已测排序、联合控制与历史感知规则未优于简单随机方案</span></div>
</div>
<p class="small">以上均为已参与研究判断的开发回放。UNSW8 采用 150 观测日，Deakin10 采用 50 观测日；不同数据集的绝对 BA 不直接比较。完整配对口径见下方实验档案。</p>
</section>'''

method = '''<section id="method">
<h2>实际实现：双时间尺度 Full New5</h2>
<p>防御端从可信设备身份取得类别 <code>Y</code>。慢层每 5 个观测日随机刷新一套合法码本；快层对每条流独立、均匀抽取一个 Mapping State。State 是<b>类别到 Code 的置换</b>，六个 State 不是六套独立 Trigger 参数。</p>
<div class="mechanism" aria-label="Full New5 防御与在线评价流程">
  <div class="mech-node"><small>可信部署端</small><strong>设备流量与类别 Y</strong><span>类别不是从 Blind 标签读取</span></div>
  <div class="mech-arrow" aria-hidden="true">↓</div>
  <div class="mech-pair">
    <div class="mech-node slow"><small>慢层 · 每 5 观测日</small><strong>刷新合法码本 C<sub>t</sub></strong><span>每个 Code 的位置、长度、Gap 重新随机生成</span></div>
    <div class="mech-node fast"><small>快层 · 每条 flow</small><strong>独立抽取 State S<sub>f</sub></strong><span>六个类别→Code 置换中均匀 IID 抽样</span></div>
  </div>
  <div class="mech-arrow" aria-hidden="true">↓</div>
  <div class="mech-node center"><small>准确的索引关系</small><strong>Y → S<sub>f</sub>(Y) → C<sub>t</sub>[Code] → Trigger 参数</strong><span>同一 State 改变类别对应的 Code；码本刷新才改变 Code 参数</span></div>
  <div class="mech-arrow" aria-hidden="true">↓</div>
  <div class="mech-node"><small>T1_W3 · 物理预算</small><strong>2 Trigger + 3 Weakener</strong><span>上行插入、有限真实包 padding 与延迟；上限 630 B / 1.5 s / 6 dummy</span></div>
  <div class="mech-arrow" aria-hidden="true">↓</div>
  <div class="mech-node"><small>持续学习攻击者</small><strong>先评分当日 Blind，再释放 8 个新标签</strong><span>累计与 14 日滑动训练；报告全期每日 Post BA</span></div>
</div>
<p class="small">防御者在线不观察攻击者模型、识别 BA 或 Blind 标签。上述流程是特征级回放；类别身份来源与实网可实现性仍需部署验证。</p>
</section>'''

core = '''<section id="core">
<h2>最有说服力的实验发现</h2>
<div class="two">
  <div class="card"><h3>同预算的动态刷新</h3><p>UNSW8 累计 LR：固定码本 + IID 为 <b>0.8054</b>，Full New5 为 <b>0.7683</b>。位置刷新贡献了 LR 收益的大部分；Deakin 累计 RF 中 Full New5 比 Position5 另低约 <b>0.0252 BA</b>，说明长度与 Gap 的增量具有模型依赖性。</p></div>
  <div class="card"><h3>稳定结论的边界</h3><p>固定六 State 的概率调度没有稳定的全期增益；已测学习型排序、历史感知选择与联合控制也未建立可靠优势。当前保留固定码本 + IID、Position5 与 Full New5 三个可解释基线。</p></div>
</div>
<p class="small">主比较使用相同原始流、时间划分、Blind 与标签到达规则；各表仍有数据集、模型和实际开销的限定。<a href="新码本组件消融.md">组件明细</a> · <a href="Deakin外部短期码本回放.md">跨数据集开发回放</a></p>
</section>'''

residual = '''<section id="residual">
<h2>关键反证：换一种攻击表示，身份仍可识别</h2>
<p>冻结 Full New5 与在线协议，只改变攻击者读到的<b>防御后可观测特征</b>。UNSW8 累计 RF 中，顺序无关视图 C 汇总下行包长、大包与事件规模；它不读取真实包标记、真实 State 或精确位置。</p>
<div class="bar-compare" role="img" aria-label="UNSW8 累计随机森林：无序视图 C 的 BA 为 0.9647，原 64 维视图 D 为 0.8818">
  <div><span>无序统计 C</span><div class="bar-track"><i style="width:96.47%"></i></div><strong>0.9647</strong></div>
  <div><span>原 64 维 D</span><div class="bar-track"><i style="width:88.18%"></i></div><strong>0.8818</strong></div>
</div>
<p><b>解释：</b>原序列攻击器受到动态扰动，但另一个可观测表示仍能识别设备。Deakin 累计 RF 的 C 为 0.5561、D 为 0.5730，不能把 UNSW8 的下行主导解释直接推广到别的数据集。</p>
<div class="callout warn"><b>尚缺的同协议对照：</b><code>BA(C_clean)</code> 与 <code>BA(C_FullNew5)</code>。当前证明了防御后残余信息足以支持识别，尚未直接量化 Full New5 对 C 视图的抑制幅度，也未证明预算内无法进一步改善。</div>
<p class="small"><a href="FullNew5残余特征归因.md">完整 A–D 视图与逐日结果</a>；详细的动作权限分析保留在下方实验档案。</p>
</section>'''

decision = '''<section id="decision">
<h2>下一阶段研究建议</h2>
<div class="card"><p>当前已完成动态码本调度、更新周期、State 数量、组件消融及学习型选择的开发性验证。证据支持固定预算下的随机动态刷新能够干扰部分持续学习攻击者，同时显示顺序无关特征仍保留显著识别能力。</p>
<p><b>建议：</b>暂时停止码本调度与 PPO 优化，把下一阶段重点放在真实流量残余信息与防御动作权限的关系。优先冻结隔离捕获与攻击协议，补齐 <code>C_clean</code> 对 <code>C_FullNew5</code> 的配对比较，再判断是否需要设计新的防御操作。</p>
<p>需要导师判断的是：在既定研究周期与部署约束下，是否启动改变残余信息的新机制；如要继续，允许调整哪些真实包和下行特征？</p></div>
</section>'''

archive_order = ["roadmap", "protocol", "early", "mapping", "dynamic", "history", "learning", "refresh", "controls", "lower", "conclusion", "audit", "reproduce"]
archive_sections = []
for key in archive_order:
    value = sections[key]
    value = re.sub(r'(<h2>)[一二三四五六七八九十]+、', r'\1', value)
    archive_sections.append(value)
archive_sections.insert(10, branch)
archive = '<details id="archive" class="archive"><summary>展开实验档案、协议与复现索引 <span>保留完整实验表和失败修正记录</span></summary>\n' + '\n'.join(archive_sections) + '\n</details>'

new_main = '\n'.join([summary, method, core, residual, decision, archive])
source = re.sub(r'<main class="wrap">.*?</main>', '<main class="wrap">\n' + new_main + '\n</main>', source, count=1, flags=re.S)
source = source.replace('FADE：动态码本的收益、残余泄漏与安全边界', '面向持续学习攻击者的动态设备指纹防御：机制、效果与残余泄漏')
source = source.replace('动态码本设备指纹防御：收益与残余信息', '面向持续学习攻击者的动态设备指纹防御：机制、效果与残余泄漏')
source = source.replace('阶段性研究汇报 · 按研究演进整理 · 开发证据与安全边界', '阶段性导师汇报 · 动态防御、在线学习与残余信息')
source = source.replace('从等预算扰动、映射适应到双时间尺度随机化：为什么既定攻击器会失配，为什么设备身份仍可识别？', '固定预算下的随机动态刷新能带来多少长期收益？哪些身份信息仍然可见？')
source = re.sub(r'<nav>.*?</nav>', '<nav><div class="wrap"><a href="#summary">研究概览</a><a href="#method">防御机制</a><a href="#core">核心结果</a><a href="#residual">残余信息</a><a href="#decision">后续方向</a><a href="#archive">实验档案</a></div></nav>', source, count=1, flags=re.S)
source = source.replace('位置池大小 k_pos</td>', '位置池大小 <code>k_pos</code></td>')
source = source.replace('State数量K=3/6/12', 'State数量 K_state=3/6/12')
source = source.replace('K=3/6/12', 'K_state=3/6/12')
source = re.sub(r'(?<![a-zA-Z_])K=(3|6|12)', r'K_state=\1', source)
source = source.replace('K_state_state', 'K_state')
source = source.replace('T2支线按主题并入三', 'T2支线置于档案附录')
source = source.replace('重排前版本保留于 artifacts/reports/', '详细实验与修正保留于本页折叠档案')
source = source.replace('footer{', '''.opening{margin-top:4px}.result-grid{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff;margin:16px 0}.result-head,.result-row{display:grid;grid-template-columns:1.05fr .9fr 1.65fr;gap:16px;align-items:center;padding:11px 16px}.result-head{background:#eaf1f8;color:var(--navy);font-size:12px;font-weight:700}.result-row{border-top:1px solid var(--line);font-size:13.5px}.result-row b{color:var(--teal);font-size:17px;font-variant-numeric:tabular-nums}.result-row span{color:#455468}.mechanism{max-width:820px;margin:18px auto;padding:20px;border:1px solid var(--line);background:#eef4f7;border-radius:14px}.mech-node{background:white;border:1px solid #b9cad8;border-left:5px solid var(--teal);border-radius:9px;padding:13px 17px;display:flex;flex-direction:column;gap:2px}.mech-node small{color:var(--blue);font-weight:700}.mech-node strong{font-size:17px;color:var(--navy)}.mech-node span{font-size:12.5px;color:var(--muted)}.mech-node.slow{border-left-color:#a16207}.mech-node.fast{border-left-color:#0369a1}.mech-node.center{background:#e5f5ef}.mech-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mech-arrow{text-align:center;color:var(--teal);font-size:20px;line-height:1.4}.bar-compare{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:16px 0}.bar-compare>div{display:grid;grid-template-columns:120px 1fr 70px;align-items:center;gap:12px;margin:9px 0;font-size:13px}.bar-compare strong{text-align:right;font-variant-numeric:tabular-nums}.bar-track{height:15px;background:#e7eef4;border-radius:99px;overflow:hidden}.bar-track i{display:block;height:100%;background:var(--teal);border-radius:99px}.bar-compare>div+div .bar-track i{background:#6f8ca4}.archive{border-top:3px solid var(--line);margin-top:40px;padding-top:16px}.archive>summary{cursor:pointer;background:#edf3f7;border:1px solid var(--line);border-radius:10px;padding:15px 18px;font-weight:700;color:var(--navy)}.archive>summary span{display:block;font-size:12px;font-weight:400;color:var(--muted)}.archive>section{margin-top:28px}.archive h2{font-size:19px}footer{''')
source = source.replace('@media(max-width:820px){.two{grid-template-columns:1fr}', '@media(max-width:820px){.two,.mech-pair{grid-template-columns:1fr}.result-head{display:none}.result-row{grid-template-columns:1fr;gap:2px}.bar-compare>div{grid-template-columns:95px 1fr 55px;gap:6px}')

REPORT.write_text(source, encoding="utf-8")
PAGE.write_text(source.replace('.md"', '.html"'), encoding="utf-8")


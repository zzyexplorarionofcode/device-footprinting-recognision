"""Rebuild the advisor deck as 12 focused slides while keeping the existing shell."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "docs/index.html"
text = path.read_text(encoding="utf-8")

# Font and layout corrections: no in-slide scrolling, larger projection text.
replacements = {
    ".slide{display:none;position:relative;overflow:auto;": ".slide{display:none;position:relative;overflow:hidden;",
    "p{font-size:clamp(16px,1.42vw,20px)": "p{font-size:clamp(18px,1.55vw,22px)",
    "h3{font-size:20px": "h3{font-size:22px",
    ".card p{font-size:16px}": ".card p{font-size:17px}",
    ".table-wrap{overflow:auto;": ".table-wrap{overflow:visible;",
    "table{border-collapse:collapse;width:100%;font-size:15px;min-width:680px}": "table{border-collapse:collapse;width:100%;font-size:16px;min-width:0}",
    "td,th{padding:12px 14px;": "td,th{padding:9px 12px;",
    ".slide.compact th,.slide.compact td{padding:7px 10px;font-size:13px}": ".slide.compact th,.slide.compact td{padding:6px 9px;font-size:15px}",
    ".note{color:var(--muted);font-size:14px}": ".note{color:var(--muted);font-size:14px}",
}
for old, new in replacements.items():
    text = text.replace(old, new)

extra_css = """
.eventrow{display:flex;align-items:center;gap:7px;margin:9px 0;flex-wrap:nowrap}
.eventrow .lbl{width:112px;flex:0 0 112px;font-size:14px;color:var(--muted);font-weight:800}
.ev{min-width:46px;padding:8px 7px;text-align:center;border-radius:7px;font-size:13px;font-weight:800;white-space:nowrap}
.ev.r{background:#dce9ef;color:#173f56}.ev.t{background:#f3d7a2;color:#6b4307}.ev.w{background:#d8e7df;color:#235b46}.ev.x{background:#eef0f1;color:#9aa6ad;text-decoration:line-through}
.formula{background:#0f2c40;color:#fff;border-radius:12px;padding:15px 18px;font-size:17px;line-height:1.65;font-family:Consolas,monospace;margin:14px 0}
.nums{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:14px 0}
.numcard{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px 14px}.numcard b{display:block;font-size:30px;color:var(--teal);line-height:1.1}.numcard span{font-size:13px;color:var(--muted)}
.verdict{font-size:clamp(24px,2.6vw,36px);line-height:1.45;font-weight:800;color:var(--navy);max-width:1050px}
.dark .verdict{color:#fff}
"""
text = text.replace("</style>", extra_css + "\n</style>")

start = text.index('<main class="deck"')
end = text.index('</main>') + len('</main>')
main = r'''<main class="deck" aria-label="导师阶段汇报">
<section class="slide dark active">
  <div class="eyebrow">导师阶段汇报 · 2026-11</div>
  <h1>动态流量扰动<br>能否持续削弱设备指纹识别？</h1>
  <div class="rule"></div>
  <p class="lead">从码本刷新、真实包 Padding 到概率长 Trigger</p>
  <div class="cover-meta">UNSW8 P8-48 · Deakin10 · 相同预算、相同 Blind、零部署后标签对照</div>
</section>

<section class="slide">
  <div class="eyebrow">01 · 机制</div>
  <h2>原始流量与防御后流量：改的是事件结构</h2>
  <div class="eventrow"><span class="lbl">原始流量</span><span class="ev r">R1</span><span class="ev r">R2</span><span class="ev r">R3</span><span class="ev r">R4</span><span class="ev r">R5</span><span class="ev r">R6</span></div>
  <div class="eventrow"><span class="lbl">Full New5</span><span class="ev r">R1+pad</span><span class="ev t">T1</span><span class="ev w">W1</span><span class="ev r">R2</span><span class="ev t">T2</span><span class="ev w">W2</span><span class="ev r">R3</span><span class="ev r">R4</span><span class="ev x">R6 被挤出</span></div>
  <div class="grid three">
    <div class="card"><h3>真实包 Padding</h3><p>R1 变长，但不删除真实包。</p></div>
    <div class="card"><h3>Dummy Mixing</h3><p>T/W 插入，改变长度、Gap 与事件计数。</p></div>
    <div class="card"><h3>插入后截断</h3><p>32-event 窗口在插入之后裁剪，部分真实包被 displaced。</p></div>
  </div>
  <div class="callout">关键不是“加了几个包”，而是<b>位置、长度、Gap 与窗口截断</b>一起改变了可观测序列。</div>
</section>

<section class="slide">
  <div class="eyebrow">02 · 攻击者与评价</div>
  <h2>同一批 Blind，比较不同“可观测表示”</h2>
  <div class="grid">
    <div class="card"><h3>D 视图</h3><p>原 64 维：32 个带方向包长 + 32 个 log Gap，依赖事件顺序。</p></div>
    <div class="card"><h3>C 视图</h3><p>45 维顺序无关统计：下行包长直方图、大包长度与规模统计。</p></div>
  </div>
  <div class="grid three">
    <div class="card"><span class="tag">训练</span><h3>累计 / 14 日滑动</h3><p>累计用全部历史标签；滑动只看近 14 日。</p></div>
    <div class="card"><span class="tag">标签</span><h3>先评分，后释放</h3><p>每天先预测 Blind，再释放 8 个新标签。</p></div>
    <div class="card"><span class="tag">指标</span><h3>BA 越低越强</h3><p>每日 BA 对当日出现类别等权，再对天数平均。</p></div>
  </div>
  <div class="callout"><b>P8-48 的边界：</b>UNSW8 只有从 2017-02-23 起 48 个观测日能同时覆盖 8 类；150 日结果只能标为“5 类 warmup + 8 类 target”，两者不可混用。</div>
</section>

<section class="slide">
  <div class="eyebrow">03 · 方案演化</div>
  <h2>从静态映射走到双时间尺度</h2>
  <div class="flow">
    <div class="step"><b>Fixed Mapping</b><span>长期共享一套类别→Code 映射；旧模型可稳定学习。</span></div>
    <div class="arrow">→</div>
    <div class="step"><b>Periodic Rotation</b><span>每段换一套映射；旧捷径失效，但仍可由少量标签识别共享 State。</span></div>
    <div class="arrow">→</div>
    <div class="step"><b>Per-flow IID</b><span>每条 flow 独立抽 State；逐流状态推断成为瓶颈。</span></div>
    <div class="arrow">→</div>
    <div class="step"><b>Full New5</b><span>5 日刷新整套码本 + 逐流 IID State。</span></div>
  </div>
  <div class="callout">这条线的核心变化不是“更随机”，而是<b>把可学习的稳定捷径逐步拆掉</b>：先换映射，再换码本，再逐流换 State。</div>
</section>

<section class="slide">
  <div class="eyebrow">04 · 核心结果</div>
  <h2>Full New5 改变了序列，但没有关闭 C 视图</h2>
  <div class="grid">
    <div class="card"><span class="tag">UNSW8</span><div class="big">0.9647</div><p>C 视图累计 RF；D 视图为 0.8818。</p></div>
    <div class="card"><span class="tag">Deakin10</span><div class="big">0.5561</div><p>C 视图累计 RF；D 视图为 0.5730。</p></div>
  </div>
  <div class="callout">动态刷新对部分序列攻击者有增量，但顺序无关统计几乎不受影响；这就是后续所有“残余”问题的起点。<br><span class="note">UNSW8 150 日结果是 5 类 warmup + 8 类 target，不能证明“初始认识 8 类”。</span></div>
</section>

<section class="slide">
  <div class="eyebrow">05 · 机制审计</div>
  <h2>为什么真实包 Padding 难以奏效</h2>
  <div class="grid three">
    <div class="card"><div class="big">347 B</div><p>Weakener 字节分账。</p></div>
    <div class="card"><div class="big">≈305 B</div><p>三个 Weakener dummy 的期望占用。</p></div>
    <div class="card"><div class="big">≈42 B</div><p>真正留给真实包 padding 的空间。</p></div>
  </div>
  <p>真实包 padding 只影响约 <b>4%</b> 的上行事件；Hybrid 把有限预算偏向长包，C-RF 有小幅改善，但 LR/MLP 不稳定，部分数据集 D 视图反而上升。</p>
  <div class="callout">结论：不是“padding 没用”，而是<b>当前 630 B 分账下，可操作的真实包 padding 太少</b>，不足以重塑长度统计。</div>
</section>

<section class="slide">
  <div class="eyebrow">06 · 新方案</div>
  <h2>概率长 Trigger：基于可操作余量追加预算</h2>
  <div class="eventrow"><span class="lbl">A 原 Full New5</span><span class="ev r">R1</span><span class="ev t">T1</span><span class="ev w">W1</span><span class="ev r">R2</span><span class="ev t">T2</span><span class="ev w">W2</span></div>
  <div class="eventrow"><span class="lbl">B 分散追加</span><span class="ev r">R1</span><span class="ev t">T1+</span><span class="ev w">W1</span><span class="ev r">R2</span><span class="ev t">T2+</span><span class="ev w">W2</span></div>
  <div class="eventrow"><span class="lbl">C 集中追加</span><span class="ev r">R1</span><span class="ev t">T1++</span><span class="ev w">W1</span><span class="ev r">R2</span><span class="ev t">T2</span><span class="ev w">W2</span></div>
  <div class="formula">extra_i = min( r_b · P_i , 256 B , first-trigger headroom ) × Bernoulli(0.5)</div>
  <div class="nums">
    <div class="numcard"><b>50%</b><span>触发概率</span></div>
    <div class="numcard"><b>256 B</b><span>单流追加上限</span></div>
    <div class="numcard"><b>≈105 B</b><span>平均可见追加</span></div>
  </div>
  <p class="note">630 B 是原始预算上限，不是每条 flow 的实际开销；没有显式“长流门槛”，只看可 padding headroom 与概率门控。</p>
</section>

<section class="slide">
  <div class="eyebrow">07 · Deakin10</div>
  <h2>长 Trigger 在 C 视图上带来明显改善</h2>
  <div class="compare-bars"><span>C 累计 RF</span><div class="rail"><i style="left:5%;width:45%"></i></div><b>−0.0716</b></div>
  <div class="compare-bars"><span>C 滑动 RF</span><div class="rail"><i style="left:17%;width:33%"></i></div><b>−0.0533</b></div>
  <div class="compare-bars"><span>C 累计 MLP</span><div class="rail"><i style="left:43%;width:7%"></i></div><b>−0.0120</b></div>
  <div class="compare-bars"><span>C 滑动 MLP</span><div class="rail"><i style="left:15%;width:35%"></i></div><b>−0.0557</b></div>
  <p class="note">中线为零，向左表示攻击 BA 下降（防御改善）。D 视图没有一致收益，D 累计 RF 反而约 +0.0095。</p>
  <div class="callout">Deakin 的改善集中在 C 视图，说明追加字节确实进入了顺序无关的长度统计；但不能外推到 D 或所有模型。</div>
</section>

<section class="slide">
  <div class="eyebrow">08 · UNSW8 P8-48</div>
  <h2>同一方案，在 UNSW 上收益有限</h2>
  <div class="grid">
    <div class="card"><span class="tag">C 累计 RF</span><div class="big">0.9854 → 0.9842</div><p>仍接近识别天花板。</p></div>
    <div class="card"><span class="tag">Frozen-clean C/RF</span><div class="big">0.9109 → 0.9081</div><p>零部署后标签，残余几乎不变。</p></div>
  </div>
  <p>C 滑动 RF 有下降（0.8447 → 0.8352），但 C 累计 RF 与 Frozen-clean 残余都没有被解决；C/LR 的 Frozen-clean 反而上升（0.8615 → 0.8930）。</p>
  <div class="callout">同一个“概率长 Trigger”，在 Deakin 改善 C、在 UNSW 几乎不动。差异不是随机波动，而是<b>两套数据的可操作统计不同</b>。</div>
</section>

<section class="slide">
  <div class="eyebrow">09 · 解释</div>
  <h2>两个数据集为何不同：一个待验证假设</h2>
  <div class="grid">
    <div class="card"><h3>Deakin10</h3><p>设备差异更多落在 LT 能触及的<b>大包长度 / 规模统计</b>；追加字节能改变这些统计。</p></div>
    <div class="card"><h3>UNSW8</h3><p>强残余更多来自<b>当前预算不可操作</b>的统计；少量追加只影响边缘。</p></div>
  </div>
  <div class="callout">这是机制假设，不是已经完成的特征因果归因。它给出的下一步不是继续调比例，而是回答：<b>哪些稳定特征在允许动作下不可触及？</b></div>
</section>

<section class="slide">
  <div class="eyebrow">10 · 五类基线</div>
  <h2>哪些方案成立，哪些不成立</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>方案</th><th>机制</th><th>证据</th><th>状态</th></tr></thead>
    <tbody>
      <tr><td>固定码本 + IID</td><td>不刷新码本，逐流 State</td><td>作为参照，C 残余高</td><td>Reference</td></tr>
      <tr><td>Position5</td><td>只刷新位置</td><td>解释大部分 LR 收益</td><td>机制基线</td></tr>
      <tr><td>Full New5</td><td>5 日刷新全码本 + IID</td><td>D 有增量，C 基本不动</td><td>当前主基线</td></tr>
      <tr><td>Hybrid</td><td>Weakener 预算偏向长包</td><td>C-RF 有时改善，LR/MLP 不一致</td><td>不替代</td></tr>
      <tr><td>概率长 Trigger</td><td>按 headroom 追加少量字节</td><td>Deakin C 改善，UNSW 有限</td><td>探索性</td></tr>
    </tbody>
  </table></div>
  <div class="callout">没有一种方案同时解决 D 与 C；目前的强结论是<b>表示相关</b>，不是“找到一个通用更强防御”。</div>
</section>

<section class="slide dark">
  <div class="eyebrow">11 · 总结</div>
  <h2>我们做了什么</h2>
  <div class="grid">
    <div class="card"><h3>机制</h3><p>用位置、Gap、Dummy mixing 与窗口截断解释防御为什么改变序列表示。</p></div>
    <div class="card"><h3>演化</h3><p>从 Fixed Mapping 到 Periodic、Per-flow IID，再到 Full New5。</p></div>
    <div class="card"><h3>边界</h3><p>Full New5 能打崩 D，但对 C 视图与 Frozen-clean 残余作用有限。</p></div>
    <div class="card"><h3>新证据</h3><p>概率长 Trigger 在 Deakin 的 C 上有效，在 UNSW 上收益有限。</p></div>
  </div>
  <p class="verdict">动态扰动改变了序列表示，但没有消除跨防御稳定的无序统计指纹。</p>
</section>
</main>'''

text = text[:start] + main + text[end:]
path.write_text(text, encoding="utf-8")
print("slides rebuilt", text.count('<section class="slide'))

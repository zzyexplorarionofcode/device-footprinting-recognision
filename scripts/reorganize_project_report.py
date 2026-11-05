from pathlib import Path
import re,html,json
root=Path(__file__).resolve().parents[1]; p=root/'docs/项目汇报.html'
old=p.read_bytes().decode('utf-8'); style=re.search(r'<style>.*?</style>',old,re.S).group()
backup=root/'artifacts/reports/项目汇报_重排前_20261105.html';backup.parent.mkdir(parents=True,exist_ok=True)
if not backup.exists():backup.write_bytes(p.read_bytes())
old=backup.read_bytes().decode('utf-8').replace('\r\n','\n')
style=re.search(r'<style>.*?</style>',old,re.S).group()

def section(id,title,content):return f'<section id="{id}">\n<h2>{title}</h2>\n{content}\n</section>\n'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>\n'+''.join('<tr>'+''.join('<td'+(' class="num"' if re.fullmatch(r'[+−\-]?\d+\.\d+%?',str(c)) else '')+'>'+str(c)+'</td>' for c in row)+'</tr>\n' for row in rows)+'</tbody></table>'
def source(name,label='完整协议与结果'):return f'<p class="small">来源：<a href="{name}">{label}</a>。主文保留关键结果，逐日、逐块及逐 Seed 明细见链接。</p>'
def oldbody(id):return re.search(r'<section id="'+id+r'">(.*?)</section>',old,re.S).group(1)
def sub(start,end):return old[old.index(start):old.index(end)]
parts=[]
parts.append(section('summary','研究摘要与当前判断','''
<p class="lead">在固定通信预算下，持续生成新 Trigger 码本并保持逐流随机 Mapping，能否让设备指纹攻击者持续付出适应成本？研究从组件优化、映射逆变换推进到在线回放，最后通过攻击表示切换检验残余信息。</p>
<div class="card"><h3>已观察到的收益</h3><p>随机动态码本降低了部分既定 LR/RF/MLP 攻击器的全期 Post BA，收益依赖数据集、模型和历史使用方式。位置刷新在已测条件中有重要贡献，长度与 Gap 对 Deakin RF 提供附加收益。</p>
<h3>改变中心结论的新证据</h3><p>Full New5 完全冻结后，UNSW8 的可观测顺序无关组合使累计 RF 达到 <b>0.9647 BA</b>，高于原64维攻击器的0.8818。Deakin同一视图在累计 RF 下为0.5561，接近原64维的0.5730。攻击表示的选择显著影响防御评价。</p>
<h3>阶段性结论</h3><p><b>扰动某种序列表示，不等于充分隐藏设备身份。</b> 动态更新的既有收益仍成立，但现有 T1_W3 留下了可跨码本利用的观测信息。该实验说明这些信息足以支持识别，尚未证明原模型的因果依赖或固定预算的不可突破下界。</p></div>
<div class="callout warn"><b>证据定位：</b>本报告为已见开发数据上的阶段性研究汇报。UNSW8 与 Deakin 均参与多轮设计判断，无全新隔离捕获上的最终验证。冻结码本调度、参数及学习型选择搜索，保留 Full New5 与 Position5 为研究基线。</div>
'''))
parts.append(section('roadmap','一、研究演进：每一阶段解决了什么','''
<p>下表按实际研究推进的先后排列；后文按相同逻辑展开。实验日期与流量日期不是同一概念，未记录的实施日期不作补写。</p>'''+table(['顺序','问题与实验','得到的证据','下一步为何发生'],[
['1','等成本位置／长度／Gap搜索；真实包与dummy分解','位置维度值得研究，早期已发现包长直方图残余','拆分Trigger、Weakener与位置效应'],
['2','Placebo、类内位置池、Oracle逆变换','静态位置可学习，映射改变产生可恢复的表示失配','研究未知State时的适应成本'],
['3','Periodic／Per-flow；统一权限配对','共享State易恢复，逐流推断限制当前逆变换攻击','转向真正随时间变化的防御'],
['4','固定库非平稳调度→新码本→逐流IID','单换混合概率没有稳定长期增益；新码本+逐流随机对部分攻击有效','跨数据集、历史价值及可学习选择'],
['5','Deakin验证；历史替代与边际价值；静态／历史条件选择','旧标签相对价值下降；尚无可靠可迁移选择信号','回到简单更新规则与组件'],
['6','5/10/25/50周期、复用、组件、RF/MLP、联合控制','5日成为既有参照，Position5不能普遍替代Full','检验State数、差异生成与更短周期'],
['7','K=3/6/12、Novelty、1/2日、可观测残余视图','细调无统一优势，无序统计保留较强识别信息','冻结码本优化，限定防御安全主张'],
['8','T2_W1 vs T1_W3小支线','单Seed结果随数据和视图变化，UNSW无序泄漏仍高','保留探索记录，不恢复配比搜索']])+'''
<p class="small">阅读路线：二为协议；三至四为早期机制；五至六为动态防御及历史解释；七至十为优化与否证；十一至十二为最新残余诊断及配比支线；十三给出最终主张与审稿边界。</p>'''))
parts.append(section('protocol','二、统一定义、威胁模型与评价协议','''
<h3>2.1 防御的实际操作与权限</h3><p>输入为前32个真实事件；当前方法仅对上行真实包做有限 padding 和增量延迟，插入上行 dummy，保持真实包方向和相对顺序，最后取前32个输出事件。截断是攻击者观察窗口的裁剪，不是网络丢包。下行真实包长度不被padding，插入事件仍可能改变其可见覆盖。</p>
<p>T表示Trigger<b>对数</b>。标准T1_W3为两Trigger包加三Weakener包；硬上限630 B／1.5 s／最多6 dummy，标准分账Trigger 283 B／0.45 s、Weakener 347 B／1.05 s。预算上限不等于实际开销；已做实际成本配对的实验在各自记录中注明，T2_W1仅等上限。</p>
<p>防御者按类别选择码字，实验假设部署端可由可信设备身份确定类别；尚未验证身份未知或共享网关条件下的可实现性。在线防御不读取攻击者模型、当前BA或Blind标签。流量重放属于特征级模拟，未证明端到端网络部署效果。</p>
<h3>2.2 名词避免混用</h3>'''+table(['名词','本报告中的定义'],[
['码本','每个code的Trigger位置、长度与Gap组合；生成新码本会改变这些参数'],
['Mapping State','类别到code的置换；六个State不是六个设备类别，也不是六套新码本'],
['Fixed Mapping','早期统一配对中的长期共享单一State'],
['固定码本 + IID','后期Fixed基线：保留首套码本，每条流仍独立抽State；与Fixed Mapping不同'],
['位置池大小 k_pos','早期每类可用位置对数1/2/4/8；与后期State数量K=3/6/12不同'],
['Full New5 / Position5','每5个观测日更新全码本／只更新位置，均保持K=6均匀逐流IID']])+'''
<h3>2.3 攻击者信息权限</h3><p>普通在线攻击器读取防御后32个带方向包长与32个log Gap，仅用已到达的设备标签更新。真实State标签、内部role和真实包索引不可作为普通攻击输入。Oracle逆变换和历史监督State分类器为特权诊断，单独标注；Oracle受重构和分类器限制，是参考性能而非理论上界。</p>
<h3>2.4 两类协议不可混表</h3>'''+table(['协议','训练与评分','统计口径'],[
['早期机制／块级适应','capture或时间块划分；当块Calibration先于Blind；标签预算、类别支持按原实验','原表保留自身划分；部分有按天配对bootstrap；不能与在线J直接比较'],
['在线UNSW8开发','20观测日热身抽4000标签；随后150观测日，2016-10-20至2017-04-12；17259 Blind','每日8新标签、最多128 Blind；三Seed共用日期与样本清单'],
['在线Deakin10开发','按实际UTC重建时间，排除6条跨午夜流；20观测日热身4000标签；50观测日，2023-07-10至09-03；6400 Blind','400个在线新标签，热身覆盖7类；不是全新最终测试']])+'''
<p>每天先用严格早于当日的标签训练并评分当日Blind，再释放当日8个Calibration标签。Post是使用此前已到达标签的每日模型，<b>不包含当天新标签</b>；Pre在每个调度段首日拟合后整段冻结。Blind标签只用于评分或事后诊断。滑动窗按前14个日历日取标签，样本不足两类时回退热身模型并记录；码本周期按观测日计。</p>
<p>在线LR：StandardScaler仅拟合热身特征，LogisticRegression最多500迭代、seed=0。RF：64树、深度12、叶节点最少2、seed=7500。Deakin MLP为64/32隐层、固定30轮，仅作补充压力测试；早期机制实验的模型参数以原记录为准。</p>
<h3>2.5 主指标、配对与不确定性</h3><p>每日BA对当日Blind实际出现的设备类等权，历史未知类计零召回。J<sub>H</sub>为H个观测日BA的等权平均，再对配对Seed等权；不是所有流合并后的BA，也不要求各日出现同一类集合。相同日期跨臂使用相同原始流，因此跨臂差值配对有效，跨日期曲线仍受类别集合变化影响。J<sub>150</sub>与J<sub>50</sub>分别报告。</p>
<p>在线主表是描述性均值，未统一执行显著性／等效性检验。三Seed不等于三套独立设备捕获；未提供CI的表不作“显著”宣称。切换冲击、前三日BA、T<sub>0.70</sub>、State准确率与历史价值是辅助诊断；阈值可能有天花板效应，低固定预算BA不能直接解释为适应时间延长。State RNG与样本划分分离；早期受Calibration split影响的A–F绝对BA只在同run内比较。</p>
'''+source('非平稳动态防御.md','在线时间边界审计')+source('第二阶段_MappingPolicy配对.md','块级权限与配对审计')+source('学习型码本搜索与外部时间数据.md','UTC清单与开发数据边界')))
# Keep early detailed evidence, rewrite overly broad interpretations and place in collapsible audit trail.
early=''
for key in ['phase1','trigger','mechanism','stability']:
    b=oldbody(key); b=re.sub(r'<h2>.*?</h2>', '', b, count=1,flags=re.S)
    title={'phase1':'3.1 等成本单组件搜索与早期包长残余','trigger':'3.2 Trigger／Weakener与Placebo对照','mechanism':'3.3 位置关联与映射切换','stability':'3.4 类内位置池稳定性（k_pos）'}[key]
    if key=='phase1':
        b=b.replace('位置搜索是唯一在四个数据集上都为正、且幅度稳定的组件','这些是开发搜索均值；held-out结果依赖数据集与分类器，不能视为统一泛化收益')
        b=b.replace('在等成本下，优先优化“插入位置”，而不是继续增加长度扰动。Dummy 混入是主要机制，真实包位移不是主因。','在这些搜索与clean→defended分解中，位置维度值得继续研究，dummy混入解释了主要路径差值；不代表动态在线防御或所有攻击者都由同一机制主导。')
    if key=='mechanism':
        b=b.replace('这说明“类别相关的位置”是必要的。','这支持类别相关位置是该协议中产生较大降幅的重要条件。')
        b=b.replace('真正起作用的是 <b>position mapping rotation</b>，不是 code，也不是“类位置”本身。','这些对照将较大降幅定位到位置映射切换造成的表示失配；尚不能说明持续适应后仍保持该降幅。')
        b=b.replace('  <p>这个结论也解释了为什么增加 trigger code 数量没有明显收益：攻击者学的是位置映射，而不是 code 内容。</p>','')
    if key=='stability':
        b=b.replace('所有设置保持相同 displacement、相同事件数、相同预算。','事件数与预算固定，并做位移分层及对齐审计；不能将位置随机化与其他协议下的Per-flow状态随机化等同。')
        b=b.replace('K 越大，防御越弱。原因是：当同一类有多个位置时，rotation 只是把“类级位置分布”整体置换；攻击者仍可学习分布形状，而不需要恢复精确映射。','本表中位置池增大伴随BA升高；分布形状可学习是一种解释，尚非该表单独识别的因果机制。')
        b=b.replace('类内位置稳定是防御资产。固定位置 + rotation，比“每 flow 随机位置”更强。','该块级映射切换协议中，固定类位置加切换更容易制造失配；不能外推为在线Full New5应取消逐流随机化。')
    early+=f'<h3>{title}</h3>'+b
parts.append(section('early','三、早期机制：组件效果与表示失配',early+'''
<h3>3.5 当时已出现的残余线索</h3><p>UNSW早期顺序无关包长直方图的clean→clean BA为0.9768、防御后训练→防御后测试为0.9636；真实包视图为0.9696。真实包视图使用内部角色标记，仅作诊断，全事件直方图才符合普通观测。早期padding审计约41 B/流、约4%真实事件变长；这些数值限于当时样本，不能直接替代后期在线结论。</p>
<div class="callout info">本阶段建立了“特定序列表示会失配”的证据，同时已提示包长边际未被充分隐藏。两条证据应同时保留，后续残余实验重新检验后者。</div>
'''+source('第一阶段实验结论.md')+source('第二阶段_StageA1_A2.md')+source('第二阶段_PositionPlacebo.md')))
# early state diagnostics
mechan=''
for key in ['reversible','online','perflow']:
    b=oldbody(key);b=re.sub(r'<h2>.*?</h2>','',b,count=1,flags=re.S)
    if key=='reversible':
        b=b.replace('Rotation 没有造成不可逆的身份信息损失。','这说明该实验中大量损失可恢复，不能证明所有身份信息均可逆。')
    if key=='perflow':
        b=b.replace('单流可观测特征几乎无法区分 state；监督分类器能学到一些规律，但 Top-1 只有 0.458。','早期无监督观测评分接近随机；另一个获得历史真实State监督的分类器达到Top-1 0.458。二者信息权限不同，不能表述成可观测流量一般不含State信号。')
        b=b.replace('Per-flow 的主要防御不是“已知 state 后的不可逆信息损失”，而是<b>逐流 state 推断的不确定性</b>。即使有监督历史数据，单流 state 仍难可靠恢复。','对本轮canonicalization攻击，逐流State不确定性限制了恢复收益；该诊断并未覆盖所有设备分类表示或所有攻击方法。')
        b='<p class="small">以下A–F采用历史真实State监督，属于特权攻击诊断；与后面的仅设备标签主比较分开。早期No-canon 0.546与本run 0.578受State RNG变化影响，不跨run计算增益。</p>'+b
    mechan+='<h3>'+{'reversible':'4.1 已知映射的可恢复性（特权参考）','online':'4.2 Periodic的块级适应与类别覆盖','perflow':'4.3 Per-flow监督推断与概率融合'}[key]+'</h3>'+b
parts.append(section('mapping','四、映射适应：可恢复性、标签权限与统一配对',mechan+'''
<h3>4.4 统一权限主比较：仅设备标签</h3><p>UNSW8单Seed，八个20观测日块；每块最多1024个Calibration和1000个后续Blind，共8000 Blind。三Policy共享原始流和参考LR；Fixed长期共享State、Periodic每两块共享State、Per-flow独立抽State。真实State不参与Adaptive训练，Oracle额外获得真实State。主BA对每块出现类等权后块均值，部分历史未见类计零召回。</p>'''+table(['Policy','No-canon','Adaptive','Oracle参考','Adaptive增益','Oracle−Adaptive','增益恢复比'],[
['Fixed Mapping','0.178','0.517','0.517','+0.339','0.000','100.0%'],['Periodic','0.291','0.550','0.550','+0.259','0.000','100.0%'],['Per-flow','0.312','0.442','0.539','+0.130','0.096','57.5%']])+'''
<p>Fixed/Periodic在当前标签预算下达到其Oracle参考，Per-flow仍有差距。这是策略结构与对应适应算法的联合结果，不能将差值全归因为State推断。表内由未四舍五入数计算增益与恢复比；不与上面的A–F绝对值混用。</p>'''+source('第二阶段_MappingPolicy配对.md')+source('第二阶段_FeatureCanonical.md')))
# Original dynamic 25-day table retained with explicit period.
dyn=sub('  <h3>固定状态库的调度能否持续压低识别率？</h3>','  <h3>整套新码本需要多频繁更新？</h3>')
dyn=dyn.replace('Deakin 外部短期','Deakin 跨数据集开发')
parts.append(section('dynamic','五、转向时间推进：固定状态库与新码本生成','''
<p class="lead">研究问题从“能否估计单流隐藏State”转为“过去学习能否持续迁移”。本节新码本更新周期为<b>25个观测日</b>；后文Full New5为5日，不能混作同一实验。</p>
'''+dyn+'''
<p>仅段内固定Mapping的新码本组，UNSW累计／滑动LR为0.8404／0.8089；加入逐流IID后为0.7939／0.7568。该控制支持两时间尺度结合的作用，但尚未严格估计所有因素的完整交互效应。冻结模型瞬时下降与持续学习后的收益分别评价。</p>'''+source('连续生成Trigger在线回放.md')+source('Deakin外部短期码本回放.md')))
parts.append(section('history','六、解释性诊断：历史标签是否失去价值',sub('  <h3>新码本怎样改变历史标签的价值？</h3>','  <h3>学习型生成器能否超过同池随机生成？</h3>')+'''
<p>替代实验比较等数量、等类别配额的新旧训练集；边际实验固定同一近期基础集，再分别加入同量旧标签和额外近期标签。旧池严格早于14日边界，资格由训练可用性决定，不能据Blind难度选择。共同类别边际评价覆盖147/150个重复日观测，缺少共同Blind的日期单列；全类别评分作为辅助。</p>
<div class="callout info">结果支持旧标签相对近期标签“折价”，不支持总体净负迁移。该证据来自Deakin RF及当时25日随机码本设置，不能直接归因到位置、长度、Gap某一组件，也未在5日条件下单独复验。</div>'''))
learn=sub('  <h3>学习型生成器能否超过同池随机生成？</h3>','</section>\r\n\r\n<section id="conclusion">') if False else oldbody('dynamic')
learn=learn[learn.index('  <h3>学习型生成器能否超过同池随机生成？</h3>'):]
parts.append(section('learning','七、学习型选择：依次检验质量、历史与长期价值',learn+'''
<p>这三关检验的是有限特征、候选池和回放设计；未发现可靠信号不等于证明所有学习控制无效。未训练PPO，也不能把预设规则实验称为强化学习算法对比。Deakin已用于研究判断，不作为该选择器的全新最终测试。</p>'''))
# Detailed short tables from markdown, no result recomputation.
def md_table(name,start=0):
    text=(root/'docs'/name).read_text(encoding='utf-8'); blocks=re.findall(r'(?m)^\|.*\n(?:\|.*\n)+',text)
    block=blocks[start]; rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in block.strip().splitlines()]
    return table(rows[0],rows[2:])
parts.append(section('refresh','八、简单更新规则：周期、复用与动态组件','''
<h3>8.1 初轮周期：5/10/25/50观测日</h3><p>固定新码本生成空间、逐流IID和标签协议。周期越短，部署码本越多，暴露频率与学习路径也同时改变。UNSW8三Seed结果如下：</p>'''+md_table('新码本更新周期.md',0)+'''
<p>初轮5日均值最低，成为后续工程参照；累计LR逐Seed最优为10/5/5日，不能称为普遍最优。各50日块沿用各自历史，分块排序不是自适应切换收益。</p>
<h3>8.2 持续新生成还是有限库循环</h3><p>每5日切换，New5与Reuse5前30日完全相同；随后Reuse5循环前六套。累计LR全期0.7683 vs 0.7833（差−0.0150，三Seed两降一升）；滑动LR为0.7405 vs 0.7404（差约零）。该比较同时改变多样性与重复暴露，不能独立识别新颖性。</p>
<h3>8.3 UNSW8单组件动态更新</h3>'''+md_table('新码本组件消融.md',0)+'''
<p>Position5保留Full相对固定码本约94%／96%的累计／滑动LR降幅。该比例是特定参照下的描述性结果，不是等效性结论；长度更新指Trigger码包，不是增加真实包padding。</p>
<h3>8.4 Deakin跨数据集开发复验与模型差异</h3>'''+table(['攻击者','固定码本+IID','Position5','Full New5','Full−Position5'],[['累计LR','0.3299','0.3197','0.3210','+0.0013'],['滑动LR','0.4412','0.4119','0.4073','−0.0047'],['累计RF','0.6295','0.5983','0.5730','−0.0252'],['滑动RF','0.5506','0.5295','0.5092','−0.0203'],['累计MLP','0.4740','0.4540','0.4483','−0.0058'],['滑动MLP','0.3618','0.3547','0.3529','−0.0018']])+'''
<p>Position5对四类LR/RF攻击的三个重复均低于固定码本。Full对RF仍有额外收益；MLP仅为固定小模型，其Fixed BA低于RF，不能认为已覆盖更强深度攻击。</p>
<h3>8.5 长度与Gap是否互补</h3>'''+table(['Deakin攻击','Position5','P+Length5','P+Gap5','Full','交互I'],[['累计RF','0.5983','0.5834','0.5858','0.5730','+0.0021'],['滑动RF','0.5295','0.5181','0.5174','0.5092','+0.0033']])+'''
<p>I = Full − P+L − P+G + P；BA越低越好，因此I&lt;0才表示超加和收益。本轮平均I略正且逐Seed变号，未形成稳定有利交互。保留Position5作机制基线，Full作完整参照。</p>'''+source('新码本更新周期.md')+source('新码本组件消融.md')+source('Deakin外部短期码本回放.md')))
parts.append(section('controls','九、有限优化空间：联合控制、State数量与历史差异','''
<h3>9.1 双时间尺度预设2×2控制</h3><p>A为Full New5；B每日轮转非均匀State概率；C码本持续5/10日交替；D合并二者。规则不读取攻击者BA，C/D同时减少部署码本数。</p>'''+md_table('双时间尺度联合控制可行性.md',0)+'''
<p>联合组平均交互虽为负，但全期BA仍高于A；本轮规则没有建立学习型联合控制的增量价值。</p>
<h3>9.2 State数量K=3/6/12</h3>'''+md_table('State数量消融.md',0)+'''
<p>各K从同一K=6热身历史开始，库为同一12-State序列的前缀。最低点依赖数据与更新规则，不能据两数据集分离设备类别数效应，也不能由三点证明极限饱和。保留K=6，不主张最优。</p>
<h3>9.3 历史差异性感知生成</h3><p>固定四候选池，按位置／长度／Gap归一化参数距离，最大化到最近三码本的最小距离。距离提高约13%，但八种数据×攻击组合的逐Seed差值均变号；累计LR/RF没有稳定收益。参数差异不等于有效观测分布差异。</p>'''+md_table('历史差异性码本生成.md',0)+'''
<p>仅否定本轮距离代理与候选池规则的稳定增量；不能推广为所有历史感知生成均无效。</p>'''+source('双时间尺度联合控制可行性.md')+source('State数量消融.md')+source('历史差异性码本生成.md')))
parts.append(section('lower','十、周期下界复核：缩短到1–2日是否继续改善','''
<p>后续仅补UNSW8，使用同一预定码本流前缀、共同热身、相同Blind与标签规则；1/2/5日部署150/75/30套码本。周期仍按观测日计，1日不是日内理论下界。</p>'''+md_table('新码本更新周期.md',5) if False else ''))
# Fill lower table by matching section explicitly, avoiding positional table assumptions.
periodtext=(root/'docs/新码本更新周期.md').read_text(encoding='utf-8').split('## 更新周期下界：')[1]
block=re.search(r'(?m)^\|.*\n(?:\|.*\n)+',periodtext).group(); cells=[[c.strip() for c in line.strip().strip('|').split('|')] for line in block.strip().splitlines()]
parts[-1]=section('lower','十、周期下界复核：缩短到1–2日是否继续改善','''<p>仅补UNSW8，使用同一预定码本流前缀、共同热身与配对标签账本。1/2/5日分别部署150/75/30套码本；1日是本轮日级回放分辨率，不是日内理论下界。</p>'''+table(cells[0],cells[2:])+'''<p>2日滑动RF相对5日的三个Seed均下降，均值−0.0076；其余差值存在变号，1日并非统一更优。保留5日既有基线，同时明确它不是所有已测周期的最优。</p>'''+source('新码本更新周期.md'))
# Latest diagnostic tables regenerated from source documentation.
parts.append(section('residual','十一、关键转折：Full New5下的可观测残余特征','''
<p class="lead">冻结防御，只切换攻击者的输入视图，直接检验位置扰动之外还剩多少可用信息。所有视图与D共用原始流、已到达设备标签、Blind、更新协议与配对Seed，不读取role、真实包索引或真实State。</p>'''+table(['视图','可观测信息','维数'],[['A','下行包长16箱归一化直方图 + 下行数量','17'],['B','&gt;256 B事件的16箱归一化直方图 + 有效事件总数','17'],['C','A+B；下行和大包长度的和／均值／标准差／最小／最大 + 大包数','45'],['D','原32维带方向长度 + 32维log Gap','64']])+md_table('FullNew5残余特征归因.md',0)+'''
<p>A–C通过事件顺序打乱不变性检查；各数据集三个Seed的逐日结果相同，应视为对当前Trigger生成序列不敏感，不当作三个独立时间复验。D复用同Seed已有结果。UNSW8全期150日，Deakin全期50日，不跨数据集直接比较绝对数值。</p>
<h3>11.1 哪些假设得到支持</h3><p><b>UNSW8：</b>下行包长与数量本身支持很强识别；C在四类攻击下超过D。<b>Deakin：</b>A约0.12，不能推广“下行主导”；B/C组合更强，C在LR下超过D，滑动RF接近D，累计RF低0.0169。</p>
<p>当前支持“存在跨码本稳定、无需精确位置与Gap的可用特征”。C高于D说明有限样本下的表示和归纳偏置会改变结果，D不构成性能上限。本实验是信息充分性测试，未识别原64维模型的因果依赖；也未拆分C中包长与数量各自的必要贡献，未测试时序残余。</p>
<h3>11.2 防御权限与预算如何限制这些信息</h3>'''+table(['信息','当前T1_W3可操作范围','尚不能断言'],[['下行包长','不做下行padding；只可能改变窗口覆盖','不能据此认定所有部署场景都不可修改下行'],['上行真实包长','允许只增padding；Weakener预算先支付三个dummy，早期均值约41 B/流用于padding','不等于630 B重新分配仍无法隐藏'],['短流规模','固定加五事件；输出有效数m&lt;32时可恢复输入数n=m−5','不代表独立包数攻击已单独测得高BA'],['方向与相对顺序','不翻转方向、不删除、不重排真实包；插包可混淆位置','尚无单独方向／顺序的当前在线因果归因']])+'''
<div class="callout warn"><b>中心认识更新：</b>Full New5对特定序列攻击器的收益依然成立，但无法作为设备身份已被隐藏的证据。现有变换保留的可观测信息构成明确的攻击面；动作权限是否形成不可突破瓶颈仍属假设。</div>'''+source('FullNew5残余特征归因.md')))
parts.append(section('branch','十二、最新小支线：T2_W1与T1_W3','''
<p>本节是残余诊断之后的探索，不改变此前冻结决定。两个数据集各单Seed=0；两臂共享原始流、标签和共同T1_W3热身历史。T2为四Trigger包，保持每对283 B／0.45 s，合计566 B／0.9 s；一个Weakener剩64 B／0.6 s，固定60 B dummy，真实包padding最多4 B。两臂均五个事件、同硬上限，<b>不保证实际成本相等</b>。</p>
<p>第二对独立随机生成，冲突锚点按最近未使用位置确定性消解；未搜索。每五观测日更新，均匀逐流IID。表内T1为同Seed值，不是前面三Seed均值。</p>'''+md_table('T2W1对比T1W3.md',0)+md_table('T2W1对比T1W3.md',1)+'''
<p>T2在UNSW8原64维下降，在Deakin原64维升高。UNSW8无序C累计RF仍约0.965，未关闭残余泄漏。T2实际均值约629.95 B／0.985 s（UNSW8）及629.79 B／0.963 s（Deakin）；实际成本与padding额度变化见原记录。</p>
<p>该比较同时改变对数、位置覆盖、资源分账、Weakener分布和真实包弱化额度，不能认定纯Trigger对数效应。单Seed不支持稳健性、显著性或最终泛化结论。</p>'''+source('T2W1对比T1W3.md')))
parts.append(section('conclusion','十三、可以提交审阅的主张、边界与冻结决定','''
'''+table(['主张','支持证据','允许的结论强度'],[
['动态更新有条件有效','25日新码本+IID、5日组件、LR/RF及小MLP回放','对已测试表示、模型与标签协议的开发收益；非普遍安全保证'],
['位置刷新重要但不足以代替完整更新','UNSW8 LR的Position5接近Full；Deakin RF仍有长度/Gap增量','模型与数据依赖；未做正式等效检验'],
['历史标签相对价值改变','Deakin RF等预算替代与边际增量实验','与历史适用性下降相一致；旧标签平均仍有正增益'],
['细调与学习型选择缺少可靠增量','排序、历史交叉、双阶段、2×2、K、Novelty与周期下界','冻结当前路线；不是学习型控制不可能的证明'],
['存在很强的跨码本观测残余','Full New5下A–C的配对在线识别，UNSW8 C累计RF=0.9647','可观测统计足以支持识别；非原模型必要性／因果归因'],
['T2_W1尚无统一替代价值','单Seed两数据集D/C结果','保留探索；不能视为配比最优或结构性泄漏的解决方案']])+'''
<h3>13.1 当前阶段表述</h3><div class="card"><p>在当前在线开发回放中，随机动态码本能够降低部分既定攻击器的识别表现，并改变历史标签相对近期标签的预测价值。然而，攻击者采用顺序无关的可观测统计后，仍能利用显著的跨码本稳定信息。结果表明，扰动特定序列表示并不等于充分隐藏设备身份；现有动态码本防御的安全收益受到残余泄漏的限制，且其组成具有数据集依赖性。</p></div>
<h3>13.2 审稿限制与尚未完成的检验</h3><ul>
<li><b>开发集反复使用：</b>UNSW8和Deakin均已参与选参和研究判断；capture-disjoint或换Seed不构成全新隔离测试。需新的多类长时间捕获验证泛化。</li>
<li><b>时间与类别：</b>每日类别集合变化、行为漂移、标签覆盖和滑动窗回退均影响评价；同日配对改善内部比较，不能消除所有跨日混杂。</li>
<li><b>统计证据：</b>多数在线表只有三随机重复的描述性均值，T2仅一个；时间自相关和重复样本限制有效样本量。未作显著性或等效性宣称。</li>
<li><b>攻击覆盖：</b>LR/RF与小MLP不覆盖任意强攻击者；残余视图已显示表示可绕开部分干扰，当前结果不能作为最坏情形安全保证。</li>
<li><b>归因范围：</b>窗口裁剪、预算分配和事件结构耦合；历史折价不等于总体净负迁移；C高BA不证明所有真实包特征主导，也不证明等预算防御不可能。</li>
<li><b>部署有效性：</b>类别身份来源、真实网络时序可实现性、带宽/延迟业务影响与端到端实现均待独立验证。当前逐流成本账本不能代替实网验证。</li></ul>
<h3>13.3 冻结与后续研究边界</h3><p>保留固定码本+IID作参照，Position5解释位置刷新，Full New5作完整随机动态基线。冻结调度、K、Novelty、码本选择和PPO路线；T2仅作为支线记录。后续若启动新研究，应围绕“哪些身份信息在允许变换下仍可观察、可否改变”提出可证伪假设，并先冻结隔离数据与攻击评价协议。</p>
'''+source('双时间尺度防御_时间泛化与训练隔离.md','未来隔离验证协议')))
# Audit appendix retained without overgeneral label requirement.
fail=oldbody('failures');fail=re.sub(r'<h2>.*?</h2>','',fail,count=1,flags=re.S)
fail=fail.replace('标签必须分层','标签抽样须符合各协议：机制实验可预设分层，在线实验按固定时间账本到达，不事后按Blind分层')
parts.append(section('audit','附录A：失败、修正与结论版本',fail+'''
<p>新增修正：5日“最优”仅适用于初轮已测周期；1/2日补测后不再宣称统一最优。Full New5“防御有效”须限定攻击表示，C视图已改变总体安全判断。T2仅等预算上限，不沿用其他实验的实际成本相等陈述。</p>'''))
parts.append(section('reproduce','附录B：复现与证据索引',table(['实验簇','协议／脚本','结果位置'],[
['早期组件与表示','<a href="第一阶段实验结论.md">第一阶段v2</a>；codebook_experiments.py、mechanism_decomp.py','artifacts/es/v2_*.json'],
['Mapping配对','paired_mapping_policies.py；<a href="第二阶段_MappingPolicy配对.md">配对协议</a>','artifacts/es/paired_mapping_policies.json'],
['在线与25日新码本','online_nonstationary.py、continual_new_trigger.py','artifacts/es/online_nonstationary_s*.json；continual_new_trigger_perflow_s*.json'],
['5日周期与组件','continual_new_trigger.py；<a href="新码本组件消融.md">组件协议</a>','artifacts/es/new_codebook_period5_s*.json；new_codebook_component_*.json'],
['Deakin与历史诊断','deakin_external_codebook.py；<a href="Deakin外部短期码本回放.md">完整账本</a>','artifacts/splits/deakin_external_short_time.json；artifacts/es/deakin_*.json'],
['历史选择与联合控制','learned_codebook_search.py、codebook_history_cross.py、codebook_two_stage_value.py','artifacts/es/；具体文件与命令见各节协议'],
['K／Novelty／更短周期','state_count_ablation.py、novelty_codebook.py','state_count_{family}_s*.json；novelty_new5_*.json；full_codebook_period{1,2}_unsw8_s*.json'],
['残余视图','residual_feature_views.py；入口state_count_ablation --residual','artifacts/es/residual_views_{family}_s{0,1,2}.json'],
['T2支线','two_trigger_pairs.py、two_pair_replay.py；--two-pairs','artifacts/es/two_pairs_{family}_s0.json']])+'''
<p>所有路径相对于项目根目录；原始日志保留逐日／逐块支持、标签数量、配对Seed及协议所需的成本或审计字段。复现应使用协议记录的配置、清单与SHA绑定数据，不能混用同名但不同时间窗结果。本文整理没有新增实验或重算研究结果。</p>
'''))
head=old[:old.index('<body>')+len('<body>')]
head=head.replace('<title>FADE 设备指纹防御：实验汇报</title>','<title>FADE：动态码本的收益、残余泄漏与安全边界</title>')
header='''
<header><div class="wrap"><div class="kicker">阶段性研究汇报 · 按研究演进整理 · 开发证据与安全边界</div>
<h1>动态码本设备指纹防御：收益与残余信息</h1>
<div class="sub">从等预算扰动、映射适应到双时间尺度随机化：为什么既定攻击器会失配，为什么设备身份仍可识别？</div>
<div class="meta">整理：2026-11-05 · UNSW8 150观测日 / Deakin10 50观测日为在线开发回放 · 早期CIC/UNSW2018/Mon为机制补充</div></div></header>
'''
nav='<nav><div class="wrap">'+''.join(f'<a href="#{i}">{t}</a>' for i,t in [('summary','摘要'),('roadmap','演进顺序'),('protocol','协议与权限'),('early','早期机制'),('mapping','映射适应'),('dynamic','动态生成'),('history','历史价值'),('learning','学习型选择'),('refresh','周期与组件'),('controls','控制与否证'),('lower','日级下界'),('residual','残余信息'),('branch','T2支线'),('conclusion','结论边界'),('reproduce','复现索引')])+'</div></nav>'
result=head+header+nav+'\n<main class="wrap">\n'+''.join(parts)+'</main>\n<footer>FADE阶段性研究汇报 · 2026-11-05 · 在线主表为描述性开发结果；原始证据保留于 docs/ 与 artifacts/es/。重排前版本保留于 artifacts/reports/。</footer>\n</body>\n</html>\n'
m=re.search(r'<section id="branch">\s*<h2>.*?</h2>(.*?)</section>',result,re.S)
assert m
branch='<div id="branch"><h3>T2_W1与T1_W3：固定五事件的后续在线对照</h3><p class="small">后续支线，按主题集中展示：以下使用Full New5在线协议，与上方早期Stage A划分、训练及统计口径不同，不能直接比较绝对BA。</p>'+m.group(1)+'</div>'
result=result[:m.start()]+result[m.end():]
pos=result.index('<h3>是 trigger code 本身有用吗？</h3>')
result=result[:pos]+branch+result[pos:]
result=result.replace('十三、可以提交审阅的主张、边界与冻结决定','十二、可以提交审阅的主张、边界与冻结决定').replace('十一至十二为最新残余诊断及配比支线；十三给出最终主张与审稿边界','十一为最新残余诊断；T2支线按主题并入三；十二给出最终主张与审稿边界')
for a,b in [('>unsw8<','>UNSW8<'),('>deakin10<','>Deakin10<'),('>lr<','>LR<'),('>rf<','>RF<'),('>cumulative<','>累计<'),('>sliding<','>滑动<')]:result=result.replace(a,b)
result=re.sub(r'(<table>.*?</table>)',r'<div style="max-width:100%;overflow-x:auto">\1</div>',result,flags=re.S)
assert re.search(r'<style>.*?</style>',result,re.S).group()==style
p.write_bytes(result.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
print('Reorganized report:',len(result),'characters; sections',len(parts))

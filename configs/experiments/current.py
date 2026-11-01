"""FADE setting: CIC Active and UNSW8, total wire delay <= 1.5 s."""

# =============================================================================
# 中文阅读导图：主实验（FADE 单次轮换）的 case 列表
# =============================================================================
# 这个文件是 experiments.run_experiment 的默认配置。它不实现算法，只回答：
#   * 用哪份 NPZ（dataset + data_protocol 双重校验，防止拿错版本）；
#   * 用哪组物理预算（shared_budget 来自 configs/experiments/fade_budget.json，
#     即 630 B / 1.5 s / 6 事件；预算名写进 budget_profile 便于结果表分组）；
#   * 怎么切分（split.unit='capture'；UNSW 额外使用冻结分区协议与 1 万/类上限）；
#   * 调哪个适配器、传什么参数（methods[0].key 决定 ADAPTERS 里的类）。
#
# 两个 case 的区别只在数据集与类别数：
#   cic_active_session_v2 → CIC Active 10 类，group 是整份 PCAP；
#   unsw_8class_main      → UNSW 8 类，group 是 UTC 日，且带 partition_protocol。
#
# FADE 的 train_mapping='old' / test_mapping='next' 是论文的核心协议：
# 攻击者在旧映射下训练，测试时防御方已经切换到新映射（无固定点错排）。
# 想改实验规模时，优先复制本文件、改 repeat_count/attacker_models，
# 而不是直接改这里——默认配置应与论文表格一一对应。
from copy import deepcopy
from configs.experiments.common import ROOT, EVALUATION
from experiment_support.budget import load_budget

DATASETS = (
    ('cic_active_session_v2', 'cic_v2/flows_cic_active_10class.npz', 'cic_active', 10,
     'cic-tcp-sessions-v2-first-packet-origin', 'whole raw PCAP'),
    ('unsw_8class_main', 'unsw_main8_v1/flows_unsw_8class.npz', 'unsw8', 8,
     'unsw-tcp-session-day-v2', 'frozen global UTC days; training capped at 10000 per class'),
)
EXPERIMENTS = []
for name, filename, label, classes, protocol, groups in DATASETS:
    evaluation = deepcopy(EVALUATION)
    evaluation['repeat_count'] = 1
    case = dict(name=name, dataset=ROOT/'data/processed'/filename,
        data_protocol=protocol, expected_classes=classes,
        shared_budget=load_budget(ROOT/'configs/experiments/fade_budget.json').to_dict(),
        budget_profile='FADE-630B-1500ms-6events',
        output_dir=ROOT/'artifacts/runs/fade'/name,
        feature={'max_frame_length':1514., 'max_gap_seconds':10.},
        split={'unit':'capture','group_semantics':groups,'train_ratio':.75,'seed':42},
        evaluation=evaluation,
        methods=[dict(key='method33_shared_budget_v1',name='FADE-old-to-next',enabled=True,
            params={'config':ROOT/f'configs/methods/fade_{label}.json',
                    'train_mapping':'old','test_mapping':'next',
                    'include_weakener':True,'include_trigger':True})])
    if label == 'unsw8':
        case['split'].update(partition_protocol='unsw-8class-train-cap10000-v1',train_cap_per_class=10000)
    EXPERIMENTS.append(case)

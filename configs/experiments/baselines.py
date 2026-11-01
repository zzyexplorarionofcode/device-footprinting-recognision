"""Baseline comparison: 630 B, 3.5 s, 6 added events per flow."""

# =============================================================================
# 中文阅读导图：把 FADE 的两个 case “翻转”成六基线对比 case
# =============================================================================
# 做法：deepcopy current.EXPERIMENTS（保持数据集、切分、特征完全一致），
# 只替换三样东西：
#   1. shared_budget → baseline_budget.json（630 B / 3.5 s / 6 事件）；
#      注意基线的时间上限更宽（3.5 s vs FADE 1.5 s），这是论文的设定，
#      不是笔误：基线需要更多时延才能达到相近的混淆强度。
#   2. methods → common.BASELINES，并把 key 改写为 *_shared_budget_v1，
#      从而走 adapters.SharedBudgetMixin 的约束路径；
#   3. output_dir → artifacts/runs/baselines/<case>。
#
# 这样同一份数据上，FADE 与 6 个基线只有“方法”不同，其余协议逐字相同。
# 运行：python -B -m experiments.run_experiment --config configs/experiments/baselines.py \
#         --case cic_active_session_v2 --dry-run
from copy import deepcopy
from configs.experiments.current import EXPERIMENTS as FADE, ROOT
from configs.experiments.common import BASELINES, EVALUATION
from experiment_support.budget import load_budget
KEYS = {'traditional':'traditional_shared_budget_v1', 'ipet_v2':'ipet_shared_budget_v1',
        'iotgan':'iotgan_shared_budget_v1', 'glimpse_oracle':'glimpse_shared_budget_v1'}
EXPERIMENTS = deepcopy(FADE)
for case in EXPERIMENTS:
    case['shared_budget'] = load_budget(ROOT/'configs/experiments/baseline_budget.json').to_dict()
    case['budget_profile'] = 'baseline-630B-3.5s-6events'
    case['evaluation'] = deepcopy(EVALUATION)
    case['methods'] = deepcopy(BASELINES)
    for method in case['methods']:
        method['key'] = KEYS[method['key']]
    case['output_dir'] = ROOT/'artifacts/runs/baselines'/case['name']

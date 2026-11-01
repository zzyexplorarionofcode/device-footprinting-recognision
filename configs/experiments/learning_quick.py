"""学习/冒烟用的小配置：不是论文数字，只用于第一次跑通全流程。

与 ``current.py`` 的区别只有三处，目的是把一次运行压到可接受的时间：
  * 攻击者只保留 MLP / RF / SVM，不训练 LSTM（CPU 上 LSTM 最慢）；
  * 重训练比例只取 0 / 50% / 100% 三个点（论文曲线是 0/10/25/50/75/100）；
  * 输出目录写到 ``artifacts/runs/learning/<case>``，与正式结果分开。

数据、切分、特征、预算、Method33 配置都与 ``current.py`` 完全相同，
所以这里得到的结论方向可以用来看流程，但**不能当作论文结果引用**。

用法：
    python -B -m experiments.run_experiment \
        --config configs/experiments/learning_quick.py \
        --case cic_active_session_v2 --dry-run
    python -B -m experiments.run_experiment \
        --config configs/experiments/learning_quick.py \
        --case cic_active_session_v2
"""

from copy import deepcopy

from configs.experiments.current import EXPERIMENTS as FADE, ROOT

# deepcopy 很重要：current.EXPERIMENTS 里的 case 会在 import 时被构造一次，
# 就地修改会污染同一进程里后续对 current.py 的使用。
EXPERIMENTS = deepcopy(FADE)

for case in EXPERIMENTS:
    # 1) 攻击者集合：去掉 LSTM-full。
    case['evaluation']['attacker_models'] = ('MLP-full', 'RF-full', 'SVM-full')
    # 2) 重训练比例：只保留三个有代表性的点。
    case['evaluation']['retraining_ratios'] = (0.0, 0.5, 1.0)
    # 3) 单次重复；seed_step 逻辑仍在，只是不会用到第二轮。
    case['evaluation']['repeat_count'] = 1
    # 4) 独立输出目录，避免和 artifacts/runs/fade 混在一起。
    case['output_dir'] = ROOT / 'artifacts/runs/learning' / case['name']

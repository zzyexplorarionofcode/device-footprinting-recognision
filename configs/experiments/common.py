"""Current exploratory protocol. Choose a case; all methods share the same PCAP split.
UNSW main uses eight classes and an immutable train-only cap of 10,000 per class.

Templates inherit historical hyperparameters; reselect on development captures before
claiming an unbiased final paper result. No historical result/policy is reused.
"""

# =============================================================================
# 中文阅读导图：所有实验共享的协议常量与 6 个基线声明
# =============================================================================
# EVALUATION 里的字段含义（改这些会影响结果可比性，务必谨慎）：
#   retraining_ratios  攻击者训练集中被“受保护版本”替换的比例，0..1 六档；
#   repeat_count       重复次数；每轮用 seed_step 偏移全部种子；
#   attacker_models    启用的识别器；LSTM-full 会额外导入 PyTorch；
#   mask_seed/model_seed/apply_seed/policy_seed  四类随机性的显式种子；
#   query_model_seed   IoTGAN 查询分类器的种子；
#   test_apply_seed_offset  train/test 应用种子差，防止攻击者记住测试实现。
#
# BASELINES 每项的结构：key（ADAPTERS 注册名）、name（CSV 分组名，必须唯一）、
# enabled（可临时关掉重型依赖）、params（透传给适配器/原生方法）。
# 注意：这里声明的是“原生方法”，统一预算版本由 baselines.py 把 key 改写为
# *_shared_budget_v1 后再注入 shared_budget。
#
# Glimpse 通过 TensorFlow/TF-Agents 训练 DDPG，与本包 PyTorch 依赖存在版本耦合；
# 只学 FADE 或其他基线时把它 enabled=False。
from pathlib import Path
from copy import deepcopy
from experiment_support.budget import load_budget
ROOT=Path(__file__).resolve().parents[2]
SHARED_BUDGET=load_budget(ROOT/'configs/experiments/baseline_budget.json').to_dict()
EVALUATION=dict(retraining_ratios=(0.,.1,.25,.5,.75,1.),repeat_count=2,
    lstm={'device':'auto','batch_size':256,'max_epochs':100},
    attacker_models=('MLP-full','RF-full','SVM-full','LSTM-full'),
    mask_seed=731100,model_seed=42,apply_seed=731800,policy_seed=734000,
    query_model_seed=42,seed_step=10000,test_apply_seed_offset=500)
# Requested comparison: six baselines plus Method33 old-to-next.
BASELINES=[
    dict(key='traditional',name='RandomRate-adapted',enabled=True,params={'variant':'random','apply':{}}),
    dict(key='traditional',name='ConstantRate-sequence-surrogate',enabled=True,params={'variant':'constant_rate','apply':{}}),
    dict(key='traditional',name='TrafficMorphing-adapted',enabled=True,params={'variant':'classic_traffic_morphing','fit':{},'apply':{}}),
    dict(key='ipet_v2',name='iPET-physical-slots-v5-outbound',enabled=True,params={'mode':'outbound','policy_seed':1733000,'fit':{'device':'auto'}}),
    dict(key='iotgan',name='IoTGAN-aligned-outbound',enabled=True,params={'mode':'outbound','policy_seed':1915000,'fit':{}}),
    dict(key='glimpse_oracle',name='Glimpse-Seq64-oracle-adapted',enabled=True,params={'config':{
        'gamma':.99,'rho':.995,'reward_threshold':.65,'training_iterations':300,
        'max_delay_seconds':.1,'max_fragments':4,'target_update_period':1,
        'replay_buffer_capacity':10000,'exploration_noise_stddev':.2,
        'exploration_noise_damping':.15,'seed':1734000}}),
]

"""FADE multi-rotation scenarios used by the paper, under the 1500 ms policy."""

# =============================================================================
# 中文阅读导图：多次轮换的 case 配置（3 次映射切换）
# =============================================================================
# 在 current.py 的两个 case 上只增加一项 rotation：
#   stages=3                  轮换次数，即映射序列长度 = stages + 1（old/next + 2）
#   application_batch_size    变换时的分块大小，控制完整事件 trace 的内存峰值
# 输出目录切到 artifacts/runs/rotation/persistent/<case>；
# 新攻击者场景由 run_rotation.py 在运行时改到 artifacts/runs/rotation/new/<case>。
#
# 三个版本之外的映射由 rotation_history.mapping_schedule 生成：
# 每版都要求“每个类别换一个从未用过的码字”，并与上一版距离尽量大。
# stages 必须小于等于类别数（码字数量上限）。
from copy import deepcopy
from configs.experiments.current import EXPERIMENTS as FADE, ROOT
EXPERIMENTS = deepcopy(FADE)
for case in EXPERIMENTS:
    case['rotation'] = {'stages':3,'application_batch_size':2048}
    case['output_dir'] = ROOT/'artifacts/runs/rotation/persistent'/case['name']

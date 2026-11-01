"""History-constrained mapping extension; never uses classifier/test scores."""

# =============================================================================
# 中文阅读导图：第 3 版及以后映射是怎么“选”出来的
# =============================================================================
# 约束来自论文的长期对抗设定：每个类别在历次轮换中用过哪些码字是公开的，
# 因此新映射必须做到
#   1) 是 0..K-1 的一个排列（每个码字恰好用一次）；
#   2) 每个类别拿到一个它在 history 里从未用过的码字；
#   3) 与上一版映射的码字距离尽量大（用 _trigger_feature 的欧氏距离）。
#
# 实现要点：
#   * 距离先转成 as_integer_ratio() 的整数权重，避免浮点并列时结果不稳定；
#   * 状态压缩 DP：states[已用码字位掩码] = (累计权重, 前缀排列)，
#     逐类别扩展，最后取掩码 (1<<K)-1 的最优解；K<=10 时规模很小；
#   * 不使用任何分类器分数或测试集信息，纯几何 + 历史约束。
#
# mapping_schedule(config, stages) 以 old/next 为前两版，逐版调用上面的函数，
# 返回 stages+1 个排列，供 run_cumulative_rotation / run_new_attacker_rotation
# 在每一轮里取对应版本。
import numpy as np

from methods.shared_tools import _trigger_feature

PROTOCOL = 'cumulative-history-no-class-code-reuse-v2'


def next_distinct_mapping(config, history):
    """Maximize distance from the latest mapping without reusing any class-code pair.

    Each label must receive a code it has never used in ``history``. The
    distance remains the existing length/gap metric, not a position metric.
    """
    k = config.class_count
    history = tuple(tuple(int(v) for v in row) for row in history)
    if not history or any(sorted(row) != list(range(k)) for row in history):
        raise ValueError('history must contain valid class-to-code permutations')
    if any(len({row[label] for row in history}) != len(history) for label in range(k)):
        raise ValueError('history already reuses a class-code pair')
    if len(history) >= k:
        raise ValueError('no unused code remains for every class')  
    old = history[-1]
    features = np.stack([_trigger_feature(code, config) for code in config.trigger_codes])
    distances = np.linalg.norm(features[:, None] - features[None, :], axis=2)
    ratios = [[float(v).as_integer_ratio() for v in row] for row in distances]
    denominator = max(d for row in ratios for _, d in row)
    weights = [[n * (denominator // d) for n, d in row] for row in ratios]
    states = {0: (0, ())}
    for label in range(k):
        following = {}
        forbidden = {row[label] for row in history}
        for used, (score, prefix) in states.items():
            for code in range(k):
                bit = 1 << code
                if used & bit or code in forbidden:
                    continue
                key = used | bit
                candidate = score + weights[old[label]][code], prefix + (code,)
                if key not in following or candidate > following[key]:
                    following[key] = candidate
        states = following
    result = states.get((1 << k) - 1)
    if result is None:
        raise ValueError('no full permutation assigns a new code to every class')
    return result[1]


def mapping_schedule(config, stages=3):
    """Preserve old/next; extend with a new code for every class at each version."""
    if isinstance(stages, bool) or not isinstance(stages, int) or stages < 1:
        raise ValueError('stages must be a positive integer')
    if stages + 1 > config.class_count:
        raise ValueError('more versions requested than available codes per class')
    history = [tuple(config.old_mapping), tuple(config.next_mapping)]
    if any(sorted(row) != list(range(config.class_count)) for row in history):
        raise ValueError('the configured initial mappings must be permutations')
    if any(a == b for a, b in zip(*history)):
        raise ValueError('the configured initial rotation must change every class')
    while len(history) < stages + 1:
        history.append(next_distinct_mapping(config, history))
    return history

"""Run the packaged synthetic regression suite without real datasets."""
# =============================================================================
# 中文阅读导图：首次接触本包时最安全的一条命令
# =============================================================================
# 作用：发现 tests/ 下的全部测试并串行执行，不需要任何真实 PCAP/NPZ。
# 覆盖内容大致包括：FADE 行为与预算、32 事件窗口边界、iPET 物理投影、
# Glimpse 分片约束、UNSW 会话切分、LSTM 攻击者接口、v2 协议。
#
# 为什么先 torch.set_num_threads(1)：LSTM 相关测试用极小批量的 CPU 训练，
# 多线程只会增加调度噪声和 CI 时间，不影响数值结论。
#
# 用法（在包根目录）：python -B scripts/run_checks.py
# 退出码非 0 表示有测试失败，可直接用于 CI。
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

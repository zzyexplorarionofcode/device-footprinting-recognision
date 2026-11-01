"""Run one Table 3 case under the existing 1500 ms total-delay FADE policy.

Both attacker scenarios use the same frozen config and three mapping rotations.
Only one repeat is run; CIC Idle is omitted.
"""

# =============================================================================
# 中文阅读导图：多次轮换的统一入口（只做分发，不实现实验）
# =============================================================================
# 两个场景对应两种攻击者假设，结果不能混用：
#   --scenario new        每轮换一个新攻击者，只用“当前映射”的受保护数据训练，
#                         测试它在当前版本 before 与下一版本 after 上的表现；
#                         → experiments/run_new_attacker_rotation.py
#   --scenario persistent 攻击者持续累积历史：训练集包含干净数据 + 历次映射的
#                         受保护数据，评估同一分类器在边界前后的表现；
#                         → experiments/run_cumulative_rotation.py
#
# 映射序列来自 configs/experiments/cumulative_rotation.py 的 rotation.stages
# （默认 3 次轮换），由 experiment_support.rotation_history.mapping_schedule 生成。
# --dry-run 只校验数据/分区/预算/映射，不训练；--resume 复用已有 RUN_DIR。
from __future__ import annotations

import argparse
from pathlib import Path

from configs.experiments.cumulative_rotation import EXPERIMENTS, ROOT
from experiments import run_cumulative_rotation as persistent
from experiments import run_new_attacker_rotation as new_attacker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=("new", "persistent"))
    parser.add_argument("--case", required=True, choices=tuple(case["name"] for case in EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="existing RUN_DIR for this scenario and case")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume cannot be combined")

    case = next(case for case in EXPERIMENTS if case["name"] == args.case)
    if args.scenario == "persistent":
        persistent.run_case(case, dry_run=args.dry_run, resume=args.resume)
    else:
        output_parent = ROOT / "artifacts/runs/rotation/new" / args.case
        new_attacker.run_case(
            case,
            transitions=case["rotation"]["stages"],
            repeats=case["evaluation"]["repeat_count"],
            dry_run=args.dry_run,
            resume=args.resume,
            output_parent=output_parent,
        )


if __name__ == "__main__":
    main()

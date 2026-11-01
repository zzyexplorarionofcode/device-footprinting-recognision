"""Traffic-defense baselines used by the unified comparison experiment."""

# =============================================================================
# 中文阅读导图：基线的模块加载方式
# =============================================================================
# 这里用 importlib.util 手动加载两个 IoTGAN 实现，而不是普通 import：
#   iotgan_aligned.py              单向模式（默认实验使用）
#   iotgan_aligned_bidirectional.py 双向模式（保留，由本包初始化时一并加载）
# 原因：文件名带下划线后缀以标记“对齐版”，但实验代码希望统一通过
# `from baselines import iotgan` 使用；手动注册到 sys.modules 后，
# 后续 import 会命中同一对象，不会重复执行模块级代码。
#
# 其余基线（traditional / common / ipet / glimpse / securitas）按常规包导入，
# 重型依赖（TensorFlow、TF-Agents）只在真正启用时由 adapters 延迟导入。
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_aligned_iotgan():
    """Expose the requested hyphenated file through the stable name iotgan."""

    module_name = f"{__name__}.iotgan_aligned"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("iotgan_aligned.py")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载aligned IoTGAN模块: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _load_bidirectional_iotgan():
    """Expose the separately named bidirectional aligned IoTGAN variant."""

    module_name = f"{__name__}.iotgan_aligned_bidirectional"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("iotgan_aligned_bidirectional.py")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载bidirectional aligned IoTGAN模块: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


# Existing experiment code can continue to use `from baselines import
# iotgan`, but the object now comes exclusively from iotgan_aligned.py.
iotgan = _load_aligned_iotgan()
iotgan_bidirectional = _load_bidirectional_iotgan()

__all__ = [
    "common",
    "glimpse",
    "iotgan",
    "iotgan_bidirectional",
    "ipet",
    "traditional",
]

"""Shared project interfaces, with no defense implementation imports."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / 'artifacts'
RUNS_ROOT = ARTIFACTS_ROOT / 'runs' / 'securitas'
VALIDATION_ROOT = ARTIFACTS_ROOT / 'validation' / 'securitas'
LEGACY_ROOT = ARTIFACTS_ROOT / 'legacy' / 'securitas'
# dataset_unsw_iotraffic itself uses the historical top-level dataset_pcap name.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_pipeline import pcap as dataset_pcap  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

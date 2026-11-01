"""Small, explicit provenance helpers for fitted experiment policies.

The helpers in this module deliberately do not discover files or inspect a
project directory.  A caller supplies the dataset path, context values and
source paths that form a policy's provenance record.  The resulting record is
stored next to a policy artifact as ``<artifact>.provenance.json``.
"""

# =============================================================================
# 中文阅读导图：给“已训练策略”发一张可核验的身份证
# =============================================================================
# 问题：缓存复用时，怎么证明磁盘上的策略确实是用这份数据、这份配置、这版代码
# 训练出来的？本文件用“规范化 JSON + 摘要”回答。
#
# 分层：
#   _normalise/canonical_json   把 Path、NumPy 标量/数组、集合、NaN 等转成
#                               确定性表示（Path 统一用 /；NaN 直接拒绝），
#                               再按 key 排序序列化；
#   digest_json / sha256_file / array_digest
#                               分别给“结构化描述 / 文件 / 数组内容”算摘要；
#   runtime_versions            记录 Python/NumPy/SciPy/sklearn/PyTorch 版本；
#   policy_descriptor           汇总 dataset、配置、源码路径、数组摘要、环境；
#   write_provenance/read_provenance
#                               原子写入 <artifact>.provenance.json 并读回；
#   validate_reuse              逐字段比较期望值与实际值，列出所有差异。
#
# 复杂语法提示：
#   * json.dumps(..., sort_keys=True, separators=(',',':'), allow_nan=False)
#     是“规范化”的关键：任何空格/键序/NaN 差异都会改变摘要；
#   * _normalise 对复数和 longdouble 做了显式分支，避免 np.generic.item()
#     在不同平台上返回不同类型而导致摘要不稳定；
#   * 摘要只依赖内容，不依赖时间戳或绝对路径，因此同一份数据换目录仍可复用。
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROVENANCE_SCHEMA = "experiment-policy-provenance-v1"


def _finite_float(value: float) -> float:
    """Return a JSON-safe float, rejecting NaN and infinities."""

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("canonical JSON does not permit NaN or infinite values")
    return result


def _normalise(value: Any) -> Any:
    """Convert supported values into deterministic, JSON-safe Python values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return _finite_float(value)

    if isinstance(value, os.PathLike):
        # JSON records use forward slashes so the same relative Path and POSIX
        # spelling do not hash differently merely because the caller is on
        # Windows.
        return Path(value).as_posix()

    if isinstance(value, np.generic):
        # ``item`` turns most NumPy scalar strings/numbers/bools into their
        # Python counterparts.  Long-double scalars intentionally remain NumPy
        # scalars on some platforms, so handle dtype categories before calling
        # it to avoid recursive conversion.
        if np.issubdtype(value.dtype, np.complexfloating):
            real = _finite_float(value.real)
            imag = _finite_float(value.imag)
            return {"__complex__": [real, imag]}
        if np.issubdtype(value.dtype, np.floating):
            return _finite_float(value)
        if np.issubdtype(value.dtype, np.integer):
            return int(value)
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value)
        if np.issubdtype(value.dtype, np.str_):
            return str(value)
        if np.issubdtype(value.dtype, np.bytes_):
            return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
        item = value.item()
        if isinstance(item, (bytes, bytearray, memoryview)):
            return {
                "__bytes__": base64.b64encode(bytes(item)).decode("ascii")
            }
        if item is value:
            raise TypeError(f"unsupported NumPy scalar dtype: {value.dtype}")
        return _normalise(item)

    if isinstance(value, np.ndarray):
        # Keeping dtype and shape in the JSON representation prevents, for
        # example, int32 and int64 arrays with equal values from becoming the
        # same provenance input.  ``tolist`` also exposes NaN to the recursive
        # finite-value check.
        return {
            "__ndarray__": {
                "dtype": value.dtype.str,
                "shape": [int(item) for item in value.shape],
                "data": _normalise(value.tolist()),
            }
        }

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normal_key = _normalise(key)
            if not isinstance(normal_key, (str, int, float, bool)):
                raise TypeError(
                    "canonical JSON object keys must be strings or scalar values"
                )
            # JSON object keys are strings.  Converting explicitly also makes
            # sorting independent of insertion order and Python key types.
            key_text = str(normal_key)
            if key_text in result:
                raise ValueError(f"duplicate canonical JSON key: {key_text!r}")
            result[key_text] = _normalise(item)
        return {key: result[key] for key in sorted(result)}

    if isinstance(value, (set, frozenset)):
        items = [_normalise(item) for item in value]
        # Sort by the canonical encoding so heterogeneous sets remain stable.
        return {
            "__set__": sorted(
                items,
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            )
        }

    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]

    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}

    if isinstance(value, complex):
        return {
            "__complex__": [_finite_float(value.real), _finite_float(value.imag)]
        }

    raise TypeError(
        f"unsupported value for canonical JSON: {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def canonical_json(value: Any) -> str:
    """Return a stable JSON encoding for supported provenance values.

    Paths and NumPy values are normalised recursively.  Object keys are sorted,
    insignificant whitespace is omitted, Unicode is kept readable, and all
    non-finite floating-point values are rejected.
    """

    normalised = _normalise(value)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_json(value: Any) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_json`."""

    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: os.PathLike[str] | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one explicitly supplied file without traversing any directory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file required for provenance: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(value: Any) -> str:
    """Hash an array including its dtype and shape.

    The byte representation is C-contiguous and therefore independent of the
    input view's strides.  Object arrays use the canonical JSON representation
    of their values because object-pointer bytes are process-dependent.
    """

    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError("array digest does not permit NaN or infinite values")

    header = canonical_json(
        {"dtype": array.dtype.str, "shape": [int(item) for item in array.shape]}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    if array.dtype.hasobject:
        digest.update(canonical_json(array.tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _lookup(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _context_value(context: Mapping[str, Any], *keys: str) -> Any:
    """Read one of a small set of explicit context locations."""

    direct = _lookup(context, *keys)
    if direct is not None:
        return direct
    dataset = context.get("dataset")
    if isinstance(dataset, Mapping):
        nested = _lookup(dataset, *keys)
        if nested is not None:
            return nested
    spec = context.get("spec")
    if isinstance(spec, Mapping):
        nested = _lookup(spec, *keys)
        if nested is not None:
            return nested
    return None


def _evaluation_value(context: Mapping[str, Any], key: str) -> Any:
    value = _lookup(context, key)
    if value is not None:
        return value
    evaluation = context.get("evaluation")
    if isinstance(evaluation, Mapping):
        value = evaluation.get(key)
        if value is not None:
            return value
    spec = context.get("spec")
    if isinstance(spec, Mapping):
        evaluation = spec.get("evaluation")
        if isinstance(evaluation, Mapping):
            return evaluation.get(key)
    return None


def _as_train_target_digest(
    prediction: Any, train_indices: np.ndarray, sample_count: int | None
) -> str:
    array = np.asarray(prediction)
    # The normal runner stores predictions for every flow.  Accepting an
    # already-sliced train array makes this helper useful for callers that keep
    # only the fit subset, while still rejecting ambiguous lengths.
    if array.ndim >= 1 and sample_count is not None and array.shape[0] == sample_count:
        array = array[train_indices]
    elif array.ndim >= 1 and array.shape[0] != len(train_indices):
        raise ValueError(
            "query target predictions must have either all-flow or train-flow length"
        )
    return array_digest(array)


def _target_digests(
    context: Mapping[str, Any], train_indices: np.ndarray, targets: Any
) -> dict[str, str]:
    if targets is None:
        targets = _lookup(
            context, "black_box_outputs", "query_target_predictions", "targets"
        )
    if targets is None:
        return {}
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping from target name to predictions")

    labels = _context_value(context, "labels")
    sample_count = None if labels is None else int(np.asarray(labels).shape[0])
    result: dict[str, str] = {}
    for target, prediction in targets.items():
        name = str(target)
        if name in result:
            raise ValueError(f"duplicate query target name: {name!r}")
        result[name] = _as_train_target_digest(prediction, train_indices, sample_count)
    return {key: result[key] for key in sorted(result)}


def _path_hashes(source_paths: Iterable[os.PathLike[str] | str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for source in source_paths:
        path = Path(source)
        key = str(path.resolve())
        if key in result:
            raise ValueError(f"duplicate provenance source path: {key}")
        result[key] = sha256_file(path)
    return {key: result[key] for key in sorted(result)}


def _array_metadata(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    return {
        "sha256": array_digest(array),
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
    }


def runtime_versions() -> dict[str, Any]:
    """Record training-library versions without importing optional frameworks."""
    import platform
    from importlib.metadata import version, PackageNotFoundError
    versions: dict[str, Any] = {'python': platform.python_version()}
    for package in ('numpy','scikit-learn','torch','tensorflow','tf-agents'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def policy_descriptor(
    context: Mapping[str, Any],
    method_key: str,
    params: Mapping[str, Any] | None,
    source_paths: Iterable[os.PathLike[str] | str],
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete, caller-scoped descriptor for one fitted policy.

    ``context`` follows ``universal_experiment.run_experiment.prepare_context``:
    ``dataset_path``, ``train_indices``, ``labels``, ``dataset.device_names``,
    ``feature``, ``evaluation.policy_seed`` and (when applicable)
    ``black_box_outputs`` are read from explicit keys.  No directory search or
    implicit source discovery is performed.
    """

    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")
    fit_params = {} if params is None else dict(params)

    dataset_path = _context_value(context, "dataset_path", "dataset_file")
    dataset_sha = _lookup(context, "dataset_sha256")
    if dataset_sha is None:
        if dataset_path is None:
            raise ValueError(
                "policy provenance requires context['dataset_path'] or an explicit dataset_sha256"
            )
        dataset_sha = sha256_file(dataset_path)
    dataset_sha = str(dataset_sha)

    train_indices_value = _context_value(context, "train_indices")
    if train_indices_value is None:
        raise ValueError("policy provenance requires context['train_indices']")
    train_indices = np.asarray(train_indices_value)
    if train_indices.ndim != 1:
        raise ValueError("train_indices must be one-dimensional")

    labels = _context_value(context, "labels")
    device_names = _context_value(context, "device_names", "devices")
    feature_protocol = _lookup(context, "feature_protocol")
    if feature_protocol is None:
        feature_protocol = _context_value(context, "feature")
    if feature_protocol is None:
        feature_protocol = {}

    mode = fit_params.get("mode")
    if mode is None:
        mode = _context_value(context, "mode")
    policy_seed = fit_params.get("policy_seed")
    if policy_seed is None:
        policy_seed = _evaluation_value(context, "policy_seed")

    labels_meta = _array_metadata(labels)
    device_meta = _array_metadata(device_names)
    source_hashes = _path_hashes(source_paths)
    target_hashes = _target_digests(context, train_indices, targets)

    descriptor: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "software": runtime_versions(),
        "method_key": str(method_key),
        "dataset_sha256": dataset_sha,
        "train_indices_sha256": array_digest(train_indices),
        "labels": labels_meta,
        "labels_sha256": None if labels_meta is None else labels_meta["sha256"],
        "device_names": device_meta,
        "device_names_sha256": None if device_meta is None else device_meta["sha256"],
        "mode": mode,
        "policy_seed": policy_seed,
        "params": fit_params,
        "fit_params": fit_params.get("fit", {}),
        "feature_protocol": feature_protocol,
        # This is an explicit part of the shared experiment contract.  Keeping
        # it in the descriptor prevents a policy fitted under a different
        # first-packet timing assumption from being silently reused.
        "protocol": {"first_packet_wait_observable": False},
        "query_target_predictions": target_hashes,
        "source_hashes": source_hashes,
    }
    # Normalising once here both validates the descriptor and ensures callers
    # receive a JSON-safe object before it is placed into a sidecar.
    return _normalise(descriptor)


def provenance_path(artifact_path: os.PathLike[str] | str) -> Path:
    """Return the conventional sidecar path for an artifact."""

    artifact = Path(artifact_path)
    return Path(f"{artifact}.provenance.json")


def atomic_json(
    path: os.PathLike[str] | str,
    value: Any,
    *,
    indent: int | None = 2,
) -> Path:
    """Write JSON through a same-directory temporary file and ``replace``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalised = _normalise(value)
    payload = json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )
    if indent is not None:
        payload += "\n"

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return destination


def write_provenance(
    artifact_path: os.PathLike[str] | str,
    descriptor: Mapping[str, Any],
    *,
    sidecar_path: os.PathLike[str] | str | None = None,
) -> Path:
    """Write a policy sidecar containing descriptor and artifact hashes."""

    artifact = Path(artifact_path)
    artifact_sha = sha256_file(artifact)
    normalised_descriptor = _normalise(descriptor)
    record = {
        "schema": PROVENANCE_SCHEMA,
        "descriptor": normalised_descriptor,
        "descriptor_sha256": digest_json(normalised_descriptor),
        "artifact_sha256": artifact_sha,
    }
    destination = provenance_path(artifact) if sidecar_path is None else Path(sidecar_path)
    return atomic_json(destination, record)


def read_provenance(
    artifact_path: os.PathLike[str] | str,
    *,
    sidecar_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Read one sidecar, refusing an absent or malformed provenance record."""

    artifact = Path(artifact_path)
    sidecar = provenance_path(artifact) if sidecar_path is None else Path(sidecar_path)
    if not sidecar.is_file():
        raise ValueError(
            f"拒绝复用策略：缺少 provenance sidecar {sidecar}; "
            "请使用新的 output 或禁用 reuse"
        )
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"拒绝复用策略：无法读取 provenance sidecar {sidecar}; "
            "请使用新的 output 或禁用 reuse"
        ) from exc
    if not isinstance(record, dict):
        raise ValueError(
            "拒绝复用策略：provenance sidecar 不是 JSON object; "
            "请使用新的 output 或禁用 reuse"
        )
    if record.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(
            "拒绝复用策略：provenance schema 不匹配; 请使用新的 output 或禁用 reuse"
        )
    if not artifact.is_file():
        raise ValueError(
            f"拒绝复用策略：artifact 不存在 {artifact}; 请使用新的 output 或禁用 reuse"
        )
    if not isinstance(record.get("descriptor"), dict):
        raise ValueError(
            "拒绝复用策略：sidecar 缺少 descriptor; 请使用新的 output 或禁用 reuse"
        )
    if record.get("descriptor_sha256") != digest_json(record["descriptor"]):
        raise ValueError(
            "拒绝复用策略：descriptor hash 与 sidecar 内容不匹配; "
            "请使用新的 output 或禁用 reuse"
        )
    actual_artifact_sha = sha256_file(artifact)
    if record.get("artifact_sha256") != actual_artifact_sha:
        raise ValueError(
            "拒绝复用策略：artifact SHA256 不匹配; 请使用新的 output 或禁用 reuse"
        )
    return record


def _differences(expected: Any, actual: Any, prefix: str = "descriptor") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{prefix}.{key}"
            if key not in expected:
                differences.append(f"{child} unexpected")
            elif key not in actual:
                differences.append(f"{child} missing")
            else:
                differences.extend(_differences(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{prefix} length {len(actual)} != {len(expected)}"]
        differences: list[str] = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(_differences(left, right, f"{prefix}[{index}]"))
        return differences
    if expected != actual:
        return [f"{prefix} differs"]
    return []


def validate_reuse(
    artifact_path: os.PathLike[str] | str,
    expected_descriptor: Mapping[str, Any],
    *,
    sidecar_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate all sidecar fields before allowing an artifact to be reused."""

    record = read_provenance(artifact_path, sidecar_path=sidecar_path)
    expected = _normalise(expected_descriptor)
    actual = record["descriptor"]
    differences = _differences(expected, actual)
    if differences:
        details = ", ".join(differences[:8])
        if len(differences) > 8:
            details += f", ... ({len(differences)} differences)"
        raise ValueError(
            f"拒绝复用策略：descriptor 字段不匹配 ({details}); "
            "请使用新的 output 或禁用 reuse"
        )
    expected_hash = digest_json(expected)
    if record.get("descriptor_sha256") != expected_hash:
        raise ValueError(
            "拒绝复用策略：descriptor hash 不匹配; 请使用新的 output 或禁用 reuse"
        )
    return record


# Explicit aliases keep the small API convenient for callers that prefer a
# policy-specific name while retaining one implementation of each operation.
write_policy_provenance = write_provenance
validate_policy_reuse = validate_reuse
check_provenance = validate_reuse


__all__ = [
    "PROVENANCE_SCHEMA",
    "array_digest",
    "atomic_json",
    "canonical_json",
    "check_provenance",
    "digest_json",
    "policy_descriptor",
    "provenance_path",
    "read_provenance",
    "sha256_file",
    "validate_policy_reuse",
    "validate_reuse",
    "write_policy_provenance",
    "write_provenance",
]

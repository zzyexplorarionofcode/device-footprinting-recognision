"""Chunk/order/parallel invariance for Method33's per-flow RNG."""

from __future__ import annotations

import concurrent.futures
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33 as core  # noqa: E402
from methods import method33_tools as api  # noqa: E402


def project():
    return api.load_config(ROOT / "configs/methods/fade_unsw8.json")


def synthetic(valid: int = 20):
    p = project()
    count = p.method33.class_count
    lengths = np.zeros((count, 32), dtype=np.float32)
    gaps = np.zeros_like(lengths)
    directions = np.zeros_like(lengths, dtype=np.int8)
    mask = np.zeros_like(lengths, dtype=np.uint8)
    lengths[:, :valid] = 300 + np.arange(valid)[None, :] * 11 + np.arange(count)[:, None] * 7
    gaps[:, :valid] = 0.002 + np.arange(valid)[None, :] * 0.0001
    gaps[:, 0] = 0
    directions[:, :valid] = np.where(np.arange(valid) % 2, -1, 1)
    mask[:, :valid] = 1
    return {
        "lengths": lengths,
        "gaps": gaps,
        "directions": directions,
        "mask": mask,
        "labels": np.arange(count),
    }


def run(data, p, indices):
    return core.apply_method33(
        data["lengths"][indices],
        data["gaps"][indices],
        data["directions"][indices],
        data["mask"][indices],
        data["labels"][indices],
        p.method33,
        731800,
        mapping="old",
        jitter_config=p.trigger_jitter,
    )


class Method33InvarianceTests(unittest.TestCase):
    def setUp(self):
        self.p = project()
        self.data = synthetic()

    def assert_arrays_equal(self, left, right):
        for index in range(4):
            np.testing.assert_array_equal(left[index], right[index])

    def test_chunk_invariance(self):
        full = run(self.data, self.p, np.arange(8))
        chunks = []
        for start, stop in ((0, 3), (3, 6), (6, 8)):
            chunks.append(run(self.data, self.p, np.arange(start, stop)))
        for index in range(4):
            np.testing.assert_array_equal(full[index], np.concatenate([chunk[index] for chunk in chunks]))

    def test_order_invariance(self):
        full = run(self.data, self.p, np.arange(8))
        order = np.array([5, 2, 7, 0, 3, 6, 1, 4])
        permuted = run(self.data, self.p, order)
        for index in range(4):
            np.testing.assert_array_equal(permuted[index], full[index][order])

    def test_parallel_invariance(self):
        full = run(self.data, self.p, np.arange(8))
        ranges = [np.arange(0, 3), np.arange(3, 6), np.arange(6, 8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            chunks = list(pool.map(lambda ids: run(self.data, self.p, ids), ranges))
        for index in range(4):
            np.testing.assert_array_equal(full[index], np.concatenate([chunk[index] for chunk in chunks]))

    def test_identical_flow_content_is_stable_across_positions(self):
        data = synthetic()
        data["lengths"][1] = data["lengths"][0]
        data["gaps"][1] = data["gaps"][0]
        data["directions"][1] = data["directions"][0]
        data["mask"][1] = data["mask"][0]
        data["labels"][1] = data["labels"][0]
        one = run(data, self.p, np.asarray([0]))
        two = run(data, self.p, np.asarray([1]))
        for index in range(4):
            np.testing.assert_array_equal(one[index], two[index])


if __name__ == "__main__":
    unittest.main()

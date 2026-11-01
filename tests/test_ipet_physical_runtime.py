"""Small physical-v5 iPET runtime regressions.

The tests use hand-built projected rows and never fit a generator.  They check
the contract at the boundary between the model's slot projection and the
runtime's packet injector, including the shared pool cursor.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.ipet.ipet_v2aligned import baseline, traffic
from baselines.ipet.ipet_v2aligned.model import physical_slot_projection


class IpetPhysicalRuntimeTests(unittest.TestCase):
    def test_projected_rows_inject_with_the_same_counts_and_bytes(self):
        raw = torch.tensor(
            [
                [0.6, 118.0, 1.6, 152.0],
                [1.2, 180.0, 2.6, 305.0],
                [0.0, 500.0, 0.2, 89.0],
            ],
            dtype=torch.float32,
        )
        projected = physical_slot_projection(raw).detach().cpu().numpy()

        events, info = traffic.inject_official_ipet_trace(
            projected,
            np.asarray([512.0]),
            np.asarray([0.0]),
            np.asarray([1], dtype=np.int8),
            omega=0.1,
            silence_threshold=0.3,
            dummy_directions=(-1, 1),
        )
        observed = {}
        for event in events:
            if not event.is_dummy:
                continue
            slot = traffic._slot_index(event.timestamp, 0.1)
            observed.setdefault((slot, event.direction), []).append(event.length)

        expected_packets = 0
        expected_bytes = 0.0
        for slot in range(len(projected)):
            for direction, offset in ((-1, 0), (1, 2)):
                count = int(projected[slot, offset])
                total = float(projected[slot, offset + 1])
                packets = observed.get((slot, direction), [])
                self.assertEqual(len(packets), count)
                self.assertAlmostEqual(sum(packets), total)
                expected_packets += count
                expected_bytes += total

        self.assertEqual(info["dummy_packets"], expected_packets)
        self.assertAlmostEqual(info["dummy_bytes"], expected_bytes)
        # The count-zero row is still consumed, but emits no placeholder frame.
        self.assertEqual(info["perturbation_rows_consumed"], 3)

    def test_omega_boundaries_match_training_slotization(self):
        lengths = np.asarray([[101.0, 202.0, 303.0, 404.0]])
        gaps = np.asarray([[0.0, 0.1, 0.2, 0.1]])
        directions = np.asarray([[1, -1, 1, -1]], dtype=np.int8)
        mask = np.ones_like(directions, dtype=bool)
        features, _ = traffic.slotize_flows(
            lengths, gaps, directions, mask, total_time=0.6, omega=0.1
        )

        # cumsum([0, .1, .2, .1]) is [0, .1, .3, .4].  The .3 event must be
        # in slot 3 even though its binary representation is just below .3.
        self.assertEqual(float(features[0, 0, 2]), 1.0)
        self.assertEqual(float(features[0, 1, 0]), 1.0)
        self.assertEqual(float(features[0, 3, 2]), 1.0)
        self.assertEqual(float(features[0, 4, 0]), 1.0)
        self.assertEqual(float(features[0, 2].sum()), 0.0)

        ticks = traffic._make_ticks(
            np.asarray([0.0, 0.1, 0.3, 0.4]),
            0.1,
            10.0,
            (-1, 1),
        )
        boundary_ticks = [tick for tick in ticks if tick[0] <= 0.4 + 1e-12]
        self.assertIn((3, -1), {(slot, direction) for _, direction, slot in boundary_ticks})
        self.assertIn((3, 1), {(slot, direction) for _, direction, slot in boundary_ticks})
        self.assertIn((4, -1), {(slot, direction) for _, direction, slot in boundary_ticks})

    def test_long_silence_keeps_absolute_slot_rows(self):
        timestamps = np.asarray([0.0, 0.1, 6.0])
        ticks = traffic._make_ticks(timestamps, 0.1, 5.0, (-1, 1))
        slots = {slot for _, _, slot in ticks}
        self.assertIn(60, slots)
        self.assertNotIn(51, slots)

        _, info = traffic.inject_official_ipet_trace(
            np.zeros((200, 4), dtype=np.float32),
            np.asarray([100.0, 120.0, 140.0]),
            np.asarray([0.0, 0.1, 5.9]),
            np.asarray([1, -1, 1], dtype=np.int8),
        )
        # The second segment begins at slot 60; skipped silent rows are part of
        # the absolute pool cursor even though they produce no packets.
        self.assertEqual(info["perturbation_rows_consumed"], 110)
        self.assertFalse(info["perturbation_pool_wrapped"])

    def test_plan_start_is_a_pool_offset_and_both_directions_share_rows(self):
        values = np.zeros((120, 4), dtype=np.float64)
        values[40] = [1.0, 100.0, 2.0, 150.0]
        values[41] = [1.0, 180.0, 3.0, 306.0]
        plan = traffic.PerturbationPlan(
            values=values,
            start_index=40,
            vector_count=3,
            slot_count=40,
        )
        events, info = traffic.inject_official_ipet_trace(
            plan,
            np.asarray([300.0]),
            np.asarray([0.0]),
            np.asarray([1], dtype=np.int8),
            omega=0.1,
            silence_threshold=0.3,
            dummy_directions=(-1, 1),
        )

        by_slot_direction = {}
        for event in events:
            if event.is_dummy:
                key = (traffic._slot_index(event.timestamp, 0.1), event.direction)
                by_slot_direction.setdefault(key, []).append(event.length)
        self.assertEqual(by_slot_direction[(0, -1)], [100.0])
        self.assertEqual(by_slot_direction[(0, 1)], [75.0, 75.0])
        self.assertEqual(by_slot_direction[(1, -1)], [180.0])
        self.assertEqual(by_slot_direction[(1, 1)], [102.0, 102.0, 102.0])
        self.assertNotIn((2, -1), by_slot_direction)
        self.assertNotIn((2, 1), by_slot_direction)
        self.assertEqual(info["perturbation_start_index"], 40)
        self.assertEqual(info["perturbation_rows_consumed"], 3)
        self.assertFalse(info["perturbation_pool_wrapped"])

    def test_schema_five_without_physical_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-v5.npz"
            metadata = {
                "schema_version": 5,
                "method": baseline.IPET_METHOD,
                "class_labels": [0],
            }
            np.savez_compressed(
                path,
                metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
            )
            with self.assertRaisesRegex(ValueError, "physical-v5|重新训练"):
                baseline.load_policy(path)


if __name__ == "__main__":
    unittest.main()

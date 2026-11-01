"""Lightweight regression tests for the versioned protocol changes.

These tests exercise the public data, sessionization, metric, and iPET
projection APIs directly.  They deliberately use small in-memory examples so
that a protocol regression can be checked without reading the full CIC PCAP
corpus or fitting a model.
"""

from __future__ import annotations

import unittest

import numpy as np

from data_pipeline.cic import Sessionizer
from data_pipeline.pcap import ParsedPacket, make_features, split_by_capture, validate_dataset
from experiment_support.metrics import timing_metrics


class DatasetProtocolTests(unittest.TestCase):
    @staticmethod
    def _dataset(*, gaps: np.ndarray | None = None, padding_length: float = 0.0):
        lengths = np.asarray([[400.0, 600.0, padding_length]], dtype=np.float32)
        if gaps is None:
            gaps = np.asarray([[0.0, 0.25, 0.0]], dtype=np.float32)
        directions = np.asarray([[1, -1, 0]], dtype=np.int8)
        mask = np.asarray([[1, 1, 0]], dtype=np.uint8)
        return {
            "lengths": lengths,
            "gaps": np.asarray(gaps, dtype=np.float32),
            "directions": directions,
            "mask": mask,
            "labels": np.asarray([0], dtype=np.int16),
            "groups": np.asarray(["capture-0"]),
        }

    def test_validate_dataset_rejects_nonzero_values_outside_mask(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_dataset(self._dataset(padding_length=900.0))

        padded_gap = self._dataset()
        padded_gap["gaps"][0, 2] = 0.1
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_dataset(padded_gap)

    def test_make_features_always_starts_observer_clock_at_first_packet(self):
        first_gap_a = np.asarray([[0.0, 0.25, 0.0]], dtype=np.float32)
        first_gap_b = np.asarray([[7.5, 0.25, 0.0]], dtype=np.float32)
        data = self._dataset(gaps=first_gap_a)
        feature_a = make_features(
            data["lengths"], data["gaps"], data["directions"], data["mask"]
        )
        feature_b = make_features(
            data["lengths"], first_gap_b, data["directions"], data["mask"]
        )
        np.testing.assert_array_equal(feature_a, feature_b)
        self.assertEqual(float(feature_a[0, data["lengths"].shape[1]]), 0.0)

    def test_mixed_capture_split_keeps_each_capture_whole(self):
        # Each raw capture contains both labels.  This is the leakage pattern
        # that per-label splitting must not divide across train and test.
        groups = np.asarray(["cap-a", "cap-a", "cap-b", "cap-b", "cap-c", "cap-c", "cap-d", "cap-d"])
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int16)
        train, test = split_by_capture(labels, groups, train_ratio=0.5, seed=11)

        self.assertTrue(np.any(train))
        self.assertTrue(np.any(test))
        self.assertEqual(set(groups[train]) & set(groups[test]), set())
        for label in np.unique(labels):
            self.assertTrue(np.any(train & (labels == label)))
            self.assertTrue(np.any(test & (labels == label)))


class SessionizerProtocolTests(unittest.TestCase):
    @staticmethod
    def _packet(timestamp: float, *, direction: int = 1, flags: int = 0, sequence: int = 0):
        return ParsedPacket(
            flow_key=(6, "10.0.0.1", 443, "10.0.0.2", 50000),
            timestamp=timestamp,
            direction=direction,
            frame_length=100,
            source_port=50000 if direction == 1 else 443,
            destination_port=443 if direction == 1 else 50000,
            is_tls_record=True,
            tcp_flags=flags,
            tcp_sequence=sequence,
        )

    def test_retransmitted_syn_does_not_split_but_new_syn_does(self):
        sessionizer = Sessionizer(timeout=120.0, width=32)
        key = "flow"
        self.assertEqual(sessionizer.push(key, self._packet(0.0, flags=0x02, sequence=100)), [])
        self.assertEqual(sessionizer.push(key, self._packet(0.1, flags=0x02, sequence=100)), [])
        completed = sessionizer.push(key, self._packet(0.2, flags=0x02, sequence=200))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].count, 2)
        self.assertEqual(sessionizer.active[key].count, 1)

    def test_rst_fin_and_timeout_close_sessions(self):
        rst = Sessionizer(timeout=120.0, width=32)
        key = "rst"
        rst.push(key, self._packet(0.0))
        completed = rst.push(key, self._packet(0.1, flags=0x04))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].count, 2)

        fin = Sessionizer(timeout=120.0, width=32)
        key = "fin"
        fin.push(key, self._packet(0.0, direction=1, flags=0x01))
        self.assertIn(key, fin.active)
        completed = fin.push(key, self._packet(0.1, direction=-1, flags=0x01))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].fins, {1, -1})

        idle = Sessionizer(timeout=120.0, width=32)
        key = "idle"
        idle.push(key, self._packet(0.0))
        completed = idle.push(key, self._packet(121.0))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].count, 1)
        self.assertEqual(idle.active[key].count, 1)

    def test_prefix_is_bounded_while_count_remains_physical(self):
        sessionizer = Sessionizer(timeout=120.0, width=32)
        key = "long"
        for index in range(40):
            sessionizer.push(key, self._packet(float(index)))
        session = sessionizer.active[key]
        self.assertEqual(session.count, 40)
        self.assertEqual(len(session.prefix), 32)


class TimingMetricTests(unittest.TestCase):
    def test_real_wire_and_cover_tail_are_separate_quantities(self):
        values = timing_metrics(
            np.asarray([0.0, 1.0, 1.0]),
            np.asarray([0.0, 2.25, 3.0]),
            np.asarray([True, True, False]),
        )
        self.assertAlmostEqual(values["real_completion_delay_seconds"], 0.25)
        self.assertAlmostEqual(values["wire_completion_delay_seconds"], 1.0)
        self.assertAlmostEqual(values["cover_tail_seconds"], 0.75)
        self.assertNotEqual(values["real_completion_delay_seconds"], values["wire_completion_delay_seconds"])


class IPETProjectionTests(unittest.TestCase):
    def test_physical_slot_projection_is_integral_and_zero_count_is_noop(self):
        torch = __import__("torch")
        from baselines.ipet.ipet_v2aligned.model import physical_slot_projection

        raw = torch.tensor([[0.2, 500.0, 2.4, 4000.0]], dtype=torch.float32)
        projected = physical_slot_projection(raw)
        actual = projected.detach().cpu().numpy().reshape(-1, 2)

        self.assertEqual(actual[0].tolist(), [0.0, 0.0])
        self.assertEqual(actual[1, 0], 2.0)
        self.assertEqual(actual[1, 1] % actual[1, 0], 0.0)
        frame_length = actual[1, 1] / actual[1, 0]
        self.assertGreaterEqual(frame_length, 60.0)
        self.assertLessEqual(frame_length, 1514.0)
        self.assertEqual(frame_length, round(frame_length))

    def test_physical_slot_projection_ste_gradient_is_finite(self):
        torch = __import__("torch")
        from baselines.ipet.ipet_v2aligned.model import physical_slot_projection

        raw = torch.tensor([[0.2, 500.0, 2.4, 4000.0]], dtype=torch.float32, requires_grad=True)
        physical = physical_slot_projection(raw, straight_through=True)
        physical.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all().item())


if __name__ == "__main__":
    unittest.main()

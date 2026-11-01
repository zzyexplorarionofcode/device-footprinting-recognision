"""Focused LSTM integration tests; no formal experiment results are overwritten.

Run: python -B tools/test_lstm_attacker.py
"""
from __future__ import annotations

import copy
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments import run_experiment as runner
from experiments.lstm_attacker import LSTMAttacker, LSTMConfig, sequence_features

(ROOT / "tmp").mkdir(exist_ok=True)

SMALL = {"hidden_size": 8, "max_epochs": 2, "batch_size": 16}
CLASSICAL = ("MLP-full", "RF-full", "SVM-full")


def training_data(classes=3, per_class=12, length=8):
    rng = np.random.default_rng(708)
    labels = np.repeat(np.arange(classes), per_class)
    values = rng.normal(0, 0.1, (len(labels), length * 2))
    values[:, :length] += labels[:, None]
    values[:, length:] += labels[:, None] * 0.5
    return values, labels


class LSTMTests(unittest.TestCase):
    def test_pairing_and_first_delay(self):
        lengths = np.array([[100.0, 200.0, 300.0]])
        gaps = np.array([[0.005, 0.01, 0.02]])
        directions = np.array([[1, -1, 1]])
        flat = runner.dataset_pcap.make_features(lengths, gaps, directions)
        before = flat.copy()
        sequence = sequence_features(flat)
        np.testing.assert_allclose(sequence[0, :, 0], [100/1514, -200/1514, 300/1514])
        np.testing.assert_allclose(sequence[0, :, 1], np.log1p([0, 10, 20]))
        self.assertEqual(sequence[0, 0, 1], 0)
        np.testing.assert_array_equal(flat, before)
        for invalid in (np.zeros((2, 3)), np.array([[0., np.nan]]), np.zeros(4)):
            with self.assertRaises(ValueError):
                sequence_features(invalid)

    def test_learns_order_and_restores_random_state(self):
        # Same values/histogram, opposite order: a sequence classification task.
        labels = np.repeat(np.array([2, 9]), 32)
        lengths = np.tile(np.array([1., 1., 1., 1., -1., -1., -1., -1.]), (64, 1))
        lengths[32:] *= -1
        features = np.concatenate((lengths, np.zeros_like(lengths)), axis=1)
        config = LSTMConfig(hidden_size=8, max_epochs=30, learning_rate=0.02, batch_size=32)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        model = LSTMAttacker(config=config, seed=23).fit(features, labels)
        self.assertGreaterEqual(accuracy_score(labels, model.predict(features)), 0.98)
        self.assertLess(model.loss_curve_[-1], model.loss_curve_[0] * 0.2)
        self.assertEqual(model.n_samples_seen_, len(labels))
        np.testing.assert_array_equal(model.classes_, [2, 9])
        np.testing.assert_array_equal(torch.get_rng_state(), torch_state)
        self.assertEqual(torch.get_num_threads(), threads)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertEqual(python_state, random.getstate())
        second = LSTMAttacker(config=config, seed=23).fit(features, labels)
        np.testing.assert_array_equal(model.predict_proba(features), second.predict_proba(features))

    def test_class_counts_and_validation(self):
        for classes in (6, 10):
            features, labels = training_data(classes=classes, per_class=2, length=32)
            model = LSTMAttacker(config=LSTMConfig(**SMALL)).fit(features, labels)
            probabilities = model.predict_proba(features)
            self.assertEqual(probabilities.shape, (len(labels), classes))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            self.assertEqual(model.predict(features[:0]).shape, (0,))
            with self.assertRaises(ValueError):
                model.predict(np.zeros((1, 2)))
        for options in ({"max_epochs": 0}, {"hidden_size": 2.5}, {"learning_rate": float("nan")}, {"class_weight": "bad"}):
            with self.assertRaises(ValueError):
                LSTMConfig(**options)
        with self.assertRaises(ValueError):
            LSTMAttacker(config=LSTMConfig(**SMALL)).fit(np.ones((4, 4)), np.zeros(4, dtype=int))

    def test_scaler_and_saved_bundle(self):
        features, labels = training_data()
        before = features.copy()
        scaler, models = runner.train_attackers(
            features, labels, 31, model_names=("LSTM-full",), lstm_params=SMALL,
        )
        reference = StandardScaler().fit(features)
        np.testing.assert_array_equal(scaler.mean_, reference.mean_)
        np.testing.assert_array_equal(scaler.scale_, reference.scale_)
        shifted = features + 500
        mean_before = scaler.mean_.copy()
        prediction = models["LSTM-full"].predict(scaler.transform(shifted))
        with tempfile.TemporaryDirectory(prefix="lstm_weights_") as folder:
            path = Path(folder) / "bundle.joblib"
            joblib.dump((scaler, models), path)
            restored_scaler, restored_models = joblib.load(path)
            np.testing.assert_array_equal(
                restored_models["LSTM-full"].predict(restored_scaler.transform(shifted)), prediction,
            )
            for key, value in models["LSTM-full"].model_.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), restored_models["LSTM-full"].model_.state_dict()[key].numpy())
        np.testing.assert_array_equal(scaler.mean_, mean_before)
        np.testing.assert_array_equal(features, before)

    def test_full_models_unchanged_when_lstm_is_added(self):
        features, labels = training_data()
        base_scaler, base = runner.shared_tools._train_quick_attackers(features, labels, seed=73)
        self.assertEqual(runner.MODEL_NAMES, CLASSICAL)
        self.assertEqual(runner.resolve_attacker_models(), CLASSICAL)
        scaler, combined = runner.train_attackers(
            features, labels, 73, model_names=runner.AVAILABLE_MODEL_NAMES, lstm_params=SMALL,
        )
        self.assertEqual(tuple(combined), CLASSICAL + ("LSTM-full",))
        np.testing.assert_array_equal(scaler.mean_, base_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, base_scaler.scale_)
        for name in CLASSICAL:
            expected = base[name].predict(base_scaler.transform(features))
            np.testing.assert_array_equal(expected, combined[name].predict(scaler.transform(features)))
            self.assertEqual(base[name].get_params(), combined[name].get_params())
        expected_metrics = {
            name: {"accuracy": float(accuracy_score(labels, model.predict(base_scaler.transform(features)))),
                   "balanced_accuracy": float(balanced_accuracy_score(labels, model.predict(base_scaler.transform(features))))}
            for name, model in base.items()
        }
        self.assertEqual(runner.evaluate(features, labels, features, labels, 73), expected_metrics)
        for selection in ((), ("bad",), ("LSTM-full", "LSTM-full"), "LSTM-full"):
            with self.assertRaises(ValueError):
                runner.resolve_attacker_models(selection)

    def test_real_method33_shared_adapter(self):
        # Use the active CIC protocol declaration, but replace its dataset with
        # a tiny in-memory-shaped NPZ.  This keeps the test on the real
        # Method33 adapter and runner without waiting for the full PCAP build.
        current = runner.load_config(ROOT / "configs/experiments/current.py")
        base = next(spec for spec in current if spec["name"] == "cic_active_session_v2")
        # Method33 remains supported even though it is no longer a default comparison.
        method = dict(key='method33_shared_budget_v1', name='Method33-old-to-old', enabled=True, params={
            'config': ROOT/'configs/methods/fade_cic_active.json',
            'train_mapping':'old', 'test_mapping':'old', 'include_weakener':True, 'include_trigger':True})
        device_names = np.asarray(
            json.loads(
                (ROOT / "configs/methods/fade_cic_active.json")
                .read_text(encoding="utf-8")
            )["dataset"]["devices"]
        )

        with tempfile.TemporaryDirectory(prefix="lstm_method33_") as folder:
            folder = Path(folder)
            flow_count, width = 10 * 4 * 2, 32
            labels = np.repeat(np.arange(10, dtype=np.int16), 4 * 2)
            groups = np.tile(np.repeat([f"capture-{i}" for i in range(4)], 2), 10)
            # Four complete events per flow; capture names are shared between
            # classes to exercise the mixed-label whole-capture splitter.
            lengths = np.zeros((flow_count, width), dtype=np.float32)
            gaps = np.zeros_like(lengths)
            directions = np.zeros((flow_count, width), dtype=np.int8)
            mask = np.zeros((flow_count, width), dtype=np.uint8)
            lengths[:, :4] = np.asarray([300.0, 420.0, 510.0, 640.0])
            gaps[:, :4] = np.asarray([0.0, 0.01, 0.02, 0.03])
            directions[:, :4] = np.asarray([1, -1, 1, -1])
            mask[:, :4] = 1
            dataset_path = folder / "cic_fixture.npz"
            np.savez_compressed(
                dataset_path,
                lengths=lengths,
                gaps=gaps,
                directions=directions,
                mask=mask,
                labels=labels,
                groups=groups,
                capture_ids=groups,
                device_names=device_names,
                data_protocol=np.asarray("cic-tcp-sessions-v2-first-packet-origin"),
            )

            spec = copy.deepcopy(base)
            spec.update(
                name="lstm-method33-smoke",
                dataset=dataset_path,
                output_dir=folder / "results",
                methods=[copy.deepcopy(method)],
            )
            spec["evaluation"].update(
                attacker_models=("LSTM-full",),
                lstm=SMALL,
                retraining_ratios=(0.0, 1.0),
                repeat_count=2,
            )
            rotated = copy.deepcopy(method)
            rotated['name'] = 'Method33-cache-rotation-test'
            rotated['params']['test_mapping'] = 'next'
            spec['methods'].append(rotated)
            context = runner.prepare_context(spec)
            np.testing.assert_array_equal(context["clean_all"][:, width], 0.0)
            self.assertEqual(set(context["groups"][context["train_indices"]]) &
                             set(context["groups"][context["test_indices"]]), set())
            with patch.object(runner, 'fit_one_model', wraps=runner.fit_one_model) as fit_calls:
                manifest = runner.run_case(spec)
            # Same training mapping: share both clean and protected models across rotations.
            self.assertEqual(fit_calls.call_count, 4)
            with Path(manifest['files']['overhead_summary']).open(encoding='utf-8-sig') as stream:
                costs = list(csv.DictReader(stream))
            for row in costs:
                self.assertEqual(float(row['budget_max_added_bytes']), spec['shared_budget']['max_added_bytes'])
                self.assertEqual(float(row['budget_max_added_packets']), spec['shared_budget']['max_added_packets'])
                self.assertLessEqual(float(row['max_added_bytes']), 630. + 1e-7)
                self.assertLessEqual(float(row['max_added_packets']), 6.)
                self.assertEqual(float(row['budget_violation_count']), 0.)
            with Path(manifest['files']['budget_audit']).open(encoding='utf-8-sig') as stream:
                audits = list(csv.DictReader(stream))
            self.assertEqual({row['split'] for row in audits}, {'train', 'test'})
            self.assertEqual(len(audits), 8)
            clean_records = [r for r in manifest['performance']
                             if r['stage'] == 'attacker_evaluation' and r['ratio'] == 0.]
            self.assertEqual([r['clean_model_cache_hit'] for r in clean_records],
                             [False, True, False, True])
            with Path(manifest["files"]["observations"]).open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["policy_target"] for r in rows}, {"shared"})
            self.assertEqual({r["model"] for r in rows}, {"LSTM-full"})
            self.assertEqual(manifest["outer_split"]["capture_intersection"], [])
            self.assertEqual(manifest["protocol"]["lstm"]["sequence_length"], 32)
            self.assertEqual(manifest["protocol"]["lstm"]["first_gap"], "zero_relative_to_first_observed_packet")
            self.assertEqual(manifest["protocol"]["first_packet_wait_observable"], False)
            self.assertEqual(
                manifest["protocol"]["timing_metrics"],
                ["real_completion_delay_seconds", "wire_completion_delay_seconds", "cover_tail_seconds"],
            )
            overhead = Path(manifest["files"]["overhead_summary"]).read_text(encoding="utf-8-sig")
            for field in ("real_completion_delay_seconds", "wire_completion_delay_seconds", "cover_tail_seconds"):
                self.assertIn(field, overhead)

    def test_target_routing_and_case_outputs(self):
        # A tiny target-specific adapter exercises query -> fit -> apply -> score
        # without retraining or changing any existing defense implementation.
        features, labels = training_data(classes=3, per_class=8)
        lengths = np.abs(features[:, :8]) * 100 + 1
        gaps = np.abs(features[:, 8:]) / 100
        arrays = (lengths, gaps, np.ones_like(lengths), np.ones_like(lengths, dtype=bool))
        clean = runner.dataset_pcap.make_features(*arrays[:3])
        train = np.concatenate([np.arange(k * 8, k * 8 + 6) for k in range(3)])
        test = np.setdiff1d(np.arange(len(labels)), train)
        target_calls = []

        class TargetAdapter:
            params = {}
            target_specific = True
            name = "test-target-adapter"
            seed_offset = 0
            metadata = {}

            def check(self, context):
                pass

            def fit(self, context, output):
                self.outputs = context["black_box_outputs"]
                self.metadata = {"targets": list(self.outputs)}

            def apply(self, context, part, *, seed, target=None):
                self_outer.assertIn(target, self.outputs)
                target_calls.append((part, target, seed))
                indices = context[f"{part}_indices"]
                return (*[a[indices].copy() for a in arrays], {})

        self_outer = self
        with tempfile.TemporaryDirectory(prefix="lstm_case_") as folder:
            dataset = Path(folder) / "input.fixture"
            dataset.write_bytes(b"synthetic fixture for manifest hashing")
            evaluation = {
                "attacker_models": ("RF-full", "LSTM-full"), "lstm": SMALL,
                "retraining_ratios": (0., 0.25, 0.5, 1.), "repeat_count": 1,
                "mask_seed": 1, "model_seed": 2, "apply_seed": 200,
                "query_model_seed": 3, "seed_step": 10000, "test_apply_seed_offset": 500,
            }
            spec = {"name": "lstm-integration-test", "dataset": dataset, "output_dir": Path(folder)/"results",
                    "methods": [], "split": {"seed": 42, "unit": "sample"}, "evaluation": evaluation}
            context = {
                "spec": spec, "dataset_path": dataset, "evaluation": evaluation, "arrays": arrays,
                "labels": labels, "groups": np.arange(len(labels)).astype(str), "class_count": 3,
                "train_indices": train, "test_indices": test, "feature": {"max_frame_length":1514., "max_gap_seconds":10.},
                "clean_all": clean, "clean_train": clean[train], "clean_test": clean[test],
                "train_labels": labels[train], "test_labels": labels[test],
            }
            adapter = TargetAdapter()
            with patch.object(runner, "prepare_context", return_value=context), patch.object(
                runner, "build_adapters", return_value=[adapter],
            ), patch.object(runner, "overhead_arrays", side_effect=lambda app: {"added_bytes": np.zeros(len(app[0]))}):
                manifest = runner.run_case(spec)
            self.assertEqual(manifest["protocol"]["attacker_models"], ["RF-full", "LSTM-full"])
            self.assertNotIn("clean_control", manifest["files"])
            self.assertFalse((Path(spec["output_dir"])/"clean_control.csv").exists())
            self.assertEqual(manifest["protocol"]["lstm"]["max_epochs"], 2)
            self.assertEqual(manifest["protocol"]["lstm"]["first_gap"], "zero_relative_to_first_observed_packet")
            self.assertIn("torch", manifest["software"])
            self.assertEqual(list(adapter.outputs), ["RF-full", "LSTM-full"])
            self.assertEqual(target_calls, [
                ("train", "RF-full", 3200), ("test", "RF-full", 3700),
                ("train", "LSTM-full", 4200), ("test", "LSTM-full", 4700),
            ])
            with Path(manifest["files"]["observations"]).open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual({row["model"] for row in rows}, {"RF-full", "LSTM-full"})
            self.assertTrue(all(row["model"] == row["policy_target"] for row in rows))
            actual_output = Path(manifest['experiment']['output_dir'])
            self.assertNotEqual(actual_output, Path(spec['output_dir']))
            self.assertEqual(json.loads((actual_output/"manifest.json").read_text(encoding="utf-8"))["status"], "complete")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    (ROOT / "tmp").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)

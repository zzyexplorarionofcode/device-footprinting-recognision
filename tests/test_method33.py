"""Behavioral and isolated-runtime tests for independent Method33."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods import method33 as core, method33_tools as api


def project(classes=10):
    return api.load_config(ROOT / ("configs/methods/fade_cic_active.json" if classes == 10 else "configs/methods/fade_unsw8.json"))


def synthetic(p, valid=32):
    count = p.method33.class_count
    lengths = np.zeros((count, 32), dtype=np.float32)
    gaps = np.zeros_like(lengths)
    directions = np.zeros_like(lengths, dtype=np.int8)
    mask = np.zeros_like(lengths, dtype=np.uint8)
    lengths[:, :valid] = 300 + np.arange(valid) * 11
    gaps[:, :valid] = .002 + np.arange(valid) * .0001
    gaps[:, 0] = 0
    directions[:, :valid] = np.where(np.arange(valid) % 2, -1, 1)
    mask[:, :valid] = 1
    return dict(lengths=lengths, gaps=gaps, directions=directions, mask=mask,
                labels=np.arange(count), groups=np.array([f"{d}/synthetic" for d in p.dataset.devices]),
                device_names=np.array(p.dataset.devices))


def apply(data, p, mapping="old", seed=731800, **kw):
    return api.apply_configured_method(data, p, mapping=mapping, seed=seed,
                                      include_event_trace=True, **kw)


def positions(trace):
    real_count = 0
    result = []
    for event in trace:
        if event["role"] == "real":
            real_count += 1
        elif event["role"] == "trigger_dummy":
            result.append(real_count)
    return result


class Method33Tests(unittest.TestCase):
    def test_current_configurations_and_hidden_first_wait(self):
        for c in (8,10):
            p=project(c)
            self.assertFalse(p.method33.require_all_dummies_in_window)
            app=apply(synthetic(p),p)
            np.testing.assert_array_equal(app[1][:,0],0)
            self.assertFalse(app[5]['first_packet_wait_observable'])

    def test_rotation_moves_positions_and_preserves_weakener_and_cost(self):
        p = project(); data = synthetic(p)
        old, new = apply(data, p), apply(data, p, "next")
        for i, (a, b) in enumerate(zip(old[6], new[6])):
            ta, tb = a["complete_event_trace"], b["complete_event_trace"]
            expected_a = list(p.method33.trigger_positions[p.method33.old_mapping[i]])
            expected_b = list(p.method33.trigger_positions[p.method33.next_mapping[i]])
            self.assertEqual(positions(ta), expected_a)
            self.assertEqual(positions(tb), expected_b)
            self.assertNotEqual(expected_a, expected_b)
            for role in ("real", "weakener_dummy"):
                va = [(e['length_bytes'], e['gap_seconds'], e['direction']) for e in ta if e['role'] == role]
                vb = [(e['length_bytes'], e['gap_seconds'], e['direction']) for e in tb if e['role'] == role]
                self.assertEqual(va, vb)
            self.assertEqual(a['retained_trigger_dummy_events'], 2)
            self.assertEqual(a['retained_weakener_dummy_events'], 3)
        for key in old[4]:
            np.testing.assert_allclose(old[4][key], new[4][key], atol=1e-10, rtol=0)

    def test_jitter_does_not_change_positions_or_zero_sum_budgets(self):
        p = project(); data = synthetic(p)
        base = core.apply_method33_base(*(data[k] for k in ('lengths','gaps','directions','mask','labels')),
                                       p.method33, 42, include_event_trace=True)
        jitter = apply(data, p, seed=42)
        self.assertTrue(any(a['length_bytes'] != b['length_bytes'] or a['gap_seconds'] != b['gap_seconds']
            for ra, rb in zip(base[6], jitter[6])
            for a, b in zip(ra['complete_event_trace'], rb['complete_event_trace']) if a['role']=='trigger_dummy'))
        for a, b in zip(base[6], jitter[6]):
            ta, tb = a['complete_event_trace'], b['complete_event_trace']
            self.assertEqual([(e['role'], e['source_index']) for e in ta], [(e['role'], e['source_index']) for e in tb])
            self.assertEqual(positions(ta), positions(tb))
            for field, total in (('length_bytes', p.method33.trigger_budget.bytes), ('gap_seconds', p.method33.trigger_budget.delay_seconds)):
                self.assertAlmostEqual(sum(e[field] for e in tb if e['role']=='trigger_dummy'), total, places=12)

    def test_short_flow_fallback_and_full_event_accounting(self):
        p = project(); data = synthetic(p, valid=4)
        original = {k: v.copy() for k, v in data.items()}
        app = apply(data, p)
        for i, row in enumerate(app[6]):
            trace = row['complete_event_trace']
            expected = [min(v, 4) for v in p.method33.trigger_positions[i]]
            self.assertEqual(positions(trace), expected)
            self.assertEqual(row['trigger_realized_real_slots'], expected)
            self.assertEqual(row['trigger_short_flow_fallback_used'], any(v>4 for v in p.method33.trigger_positions[i]))
            self.assertEqual(len(trace), 9)
            self.assertEqual([e['source_index'] for e in trace if e['role']=='real'], list(range(4)))
            self.assertEqual(sum(e['role']=='weakener_dummy' for e in trace), 3)
            self.assertEqual(sum(e['role']=='trigger_dummy' for e in trace), 2)
            self.assertAlmostEqual(sum(e['length_bytes'] for e in trace)-data['lengths'][i].sum(),
                                   app[4]['real_added_bytes'][i]+app[4]['dummy_bytes'][i], places=8)
            self.assertAlmostEqual(sum(e['gap_seconds'] for e in trace)-data['gaps'][i].astype(float).sum(),
                                   app[4]['real_added_delay'][i]+app[4]['dummy_delay'][i], places=10)
        for k in data:
            np.testing.assert_array_equal(data[k], original[k])

    def test_repeatability_and_weakener_label_independence(self):
        p = project(); data = synthetic(p)
        a, b = apply(data, p, include_trigger=False), apply(data, p, include_trigger=False)
        for x,y in zip(a[:4], b[:4]): np.testing.assert_array_equal(x,y)
        shuffled = dict(data, labels=data['labels'][::-1].copy())
        c = apply(shuffled, p, include_trigger=False)
        for x,y in zip(a[:4], c[:4]): np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(a[1][:,0], 0)

    def test_future_suffix_does_not_change_emitted_prefix(self):
        p = project(); first = synthetic(p); second = {k:v.copy() for k,v in first.items()}
        second['lengths'][:,20:] += 100
        second['gaps'][:,20:] += .5
        a,b = apply(first,p),apply(second,p)
        for ra,rb in zip(a[6],b[6]):
            def prefix(row):
                result=[]
                for e in row['complete_event_trace']:
                    if e['role']=='real' and e['source_index']==20: break
                    result.append((e['role'],e['length_bytes'],e['gap_seconds'],e['direction']))
                return result
            self.assertEqual(prefix(ra),prefix(rb))

    def test_invalid_position_tables_rejected(self):
        p = project(); positions = list(p.method33.trigger_positions)
        for bad in ((1,1),(0,4),(4,1),(1.5,4),(True,4),positions[1]):
            candidate = replace(p, method33=replace(p.method33, trigger_positions=(bad,*positions[1:])))
            with self.assertRaises(ValueError): api.validate_config(candidate)

    def test_module_cli_and_trace_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            np.savez_compressed(d/'input.npz',**synthetic(project(8),valid=4))
            command=[sys.executable,'-B','-m','methods.method33_tools','--config',str(ROOT/'configs/methods/fade_unsw8.json')]
            inspected=subprocess.run(command+['inspect'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
            self.assertEqual(len(json.loads(inspected.stdout)['effective_trigger_positions']),8)
            subprocess.run(command+['apply','--dataset',str(d/'input.npz'),'--output',str(d/'output.npz'),'--trace'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
            with np.load(d/'output.npz',allow_pickle=False) as data:
                self.assertEqual(data['lengths'].shape,(8,32))
                np.testing.assert_array_equal(data['gaps'][:,0],0)
                np.testing.assert_array_equal(data['stats__dummy_count'],5)

    def test_sparse_uplink_full_window_keeps_all_physical_events(self):
        p=project();data=synthetic(p)
        data['directions'][:]=-1
        data['directions'][:,[0,10,20]]=1
        for seed in range(10):
            app=apply(data,p,seed=seed)
            for row in app[6]:
                trace=row['complete_event_trace']
                self.assertEqual(sum(e['role']=='real' for e in trace),32)
                self.assertEqual(sum(e['role']!='real' for e in trace),5)
            np.testing.assert_array_equal(app[4]['dummy_count'],5)

if __name__=='__main__':
    unittest.main(verbosity=2)

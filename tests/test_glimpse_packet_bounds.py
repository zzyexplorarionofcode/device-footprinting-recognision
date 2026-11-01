"""Preserve captured inbound VLAN frames without expanding outbound actions."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from baselines.glimpse.glimpse_seq64_oracle_v2 import core
from experiments.adapters import GlimpseAdapter


def config():
    return core.GlimpseConfig(
        gamma=.9, rho=.95, reward_threshold=.7, training_iterations=2,
        max_delay_seconds=.1, max_fragments=4, target_update_period=1,
        replay_buffer_capacity=256, exploration_noise_stddev=.1,
        exploration_noise_damping=.15, seed=7,
    )


def arrays():
    lengths = np.zeros((1, 32)); lengths[0, :3] = [500, 1518, 600]
    gaps = np.zeros_like(lengths); gaps[0, 1:3] = [.02, .03]
    directions = np.zeros((1, 32), dtype=np.int8); directions[0, :3] = [1, -1, 1]
    mask = np.zeros((1, 32), dtype=bool); mask[0, :3] = True
    return lengths, gaps, directions, mask


class GlimpsePacketBoundsTest(unittest.TestCase):
    def test_inbound_vlan_roundtrip_and_feature_scale(self):
        original = arrays()
        checked = core.validate_packet_inputs(*original, config=config())
        flow = core._flow_from_row(*(a[0] for a in checked[:5]))
        for expected, actual in zip(original, core._flow_to_row(flow)):
            np.testing.assert_array_equal(expected[0], actual)
        self.assertEqual(core.make_features(*original)[0, 1], -1518 / 1514)

    def test_all_actions_preserve_inbound_frame_and_nonnegative_cost(self):
        checked = core.validate_packet_inputs(*arrays(), config=config())
        flow = core._flow_from_row(*(a[0] for a in checked[:5]))
        for chosen in range(3):
            action = np.full(4, -1.); action[chosen] = 1.; action[3] = 1.
            result, cost, _ = core.TrafficManipulator(config()).apply(flow, 0, action)
            inbound = next(p for p in result.packets if p.source_index == 1)
            # Delay can reorder events and thus change relative gaps, but must
            # retain the inbound frame bytes and its observed absolute time.
            self.assertEqual({k:v for k,v in vars(inbound).items() if k != 'gap'},
                             {k:v for k,v in vars(flow.packets[1]).items() if k != 'gap'})
            inbound_position = next(i for i,p in enumerate(result.packets) if p.source_index == 1)
            self.assertAlmostEqual(result.start_time + sum(p.gap for p in result.packets[:inbound_position+1]), .02)
            self.assertTrue(all(p.length <= 1514 for p in result.packets if p.direction == 1))
            self.assertGreaterEqual(cost.added_bytes, 0)
            self.assertGreaterEqual(cost.added_payload_bytes, 0)
            self.assertEqual(cost.added_bytes, sum(p.length for p in result.packets) - sum(p.length for p in flow.packets))
            with self.assertRaisesRegex(ValueError, 'outbound'):
                core.TrafficManipulator(config()).apply(flow, 1, action)

    def test_outbound_oversize_rejected_before_training(self):
        values = arrays(); values[2][0, 1] = 1
        with patch.object(core.IsolationForestRewardModel, 'fit') as fit:
            with self.assertRaisesRegex(ValueError, 'row=0, packet=1'):
                core.GlimpseSeq64OracleV2.fit(*values, np.array([0]), config=config())
            fit.assert_not_called()
        with self.assertRaisesRegex(ValueError, '1514'):
            core.Packet(1518, 0., 1, 0)

    def test_invalid_headers_remain_rejected(self):
        values = arrays(); values[0][0, 1] = 53
        with self.assertRaisesRegex(ValueError, '头长'):
            core.validate_packet_inputs(*values, config=config())
        with self.assertRaises(ValueError):
            core.Packet(53, 0., -1, 0)

    def test_normal_full_frame_padding_stays_zero(self):
        flow = core.Flow([core.Packet(1514, 0., 1, 0)])
        result, cost, _ = core.TrafficManipulator(config()).apply(flow, 0, np.array([1., -1., -1., 1.]))
        self.assertEqual(vars(result.packets[0]), vars(flow.packets[0]))
        self.assertEqual(cost.added_payload_bytes, 0)
        self.assertEqual(cost.added_bytes, 0)

    def test_adapter_preflight_checks_packet_inputs(self):
        from dataclasses import asdict
        adapter = GlimpseAdapter({'key': 'glimpse_oracle', 'params': {'config': asdict(config())}})
        context = {'arrays': arrays(), 'evaluation': {'policy_seed': 7}}
        # Dependency imports are irrelevant to the preflight contract under test.
        with patch.dict('sys.modules', {'tensorflow': SimpleNamespace(__version__='2.15.0'),
                                        'tf_agents': SimpleNamespace(__version__='0.19.0')}):
            adapter.check(context)
            context['arrays'][2][0, 1] = 1
            with self.assertRaisesRegex(ValueError, 'row=0, packet=1'):
                adapter.check(context)


if __name__ == '__main__':
    unittest.main()

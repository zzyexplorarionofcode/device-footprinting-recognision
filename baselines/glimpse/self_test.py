"""无需外部数据的独立自测试；这里的数值只用于软件测试，不是论文参数。"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace
import sys
import tempfile

import numpy as np

# Windows嵌入式Python可能不把脚本目录放入sys.path；仅补充本独立目录。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import make_synthetic_data, stratified_split
from glimpse_seq64_oracle_v2 import (
    ACTION_NAMES,
    ACTION_DIMENSION,
    DDPGMorphAgent,
    Flow,
    GlimpseConfig,
    GlimpseSeq64OracleV2,
    IsolationForestRewardModel,
    Packet,
    TrafficManipulator,
    apply_policy,
    make_features,
)


def test_features():
    lengths = np.zeros((1, 32)); lengths[0, :2] = [1514, 757]
    gaps = np.zeros((1, 32)); gaps[0, :2] = [0.0, 20.0]
    directions = np.zeros((1, 32)); directions[0, :2] = [1, -1]
    mask = np.zeros((1, 32), dtype=bool); mask[0, :2] = True
    features = make_features(lengths, gaps, directions, mask)
    assert features.shape == (1, 64)
    np.testing.assert_allclose(features[0, :2], [1.0, -0.5])
    np.testing.assert_allclose(features[0, 32:34], np.log1p([0.0, 10000.0]))
    assert np.count_nonzero(features[0, 2:32]) == 0
    gaps[0, 0] = 0.001
    try:
        make_features(lengths, gaps, directions, mask)
    except ValueError as exc:
        assert "首包" in str(exc)
    else:
        raise AssertionError("首包等待不能作为流内包间隔输入")


def test_reward_and_actions(config):
    assert ACTION_NAMES == ("payload_padding", "ip_tcp_fragmentation", "packet_delay")
    assert ACTION_DIMENSION == 4
    rng = np.random.default_rng(4)
    reward_model = IsolationForestRewardModel(seed=4)
    reward_model.fit(rng.normal(size=(20, 64)))
    vector = np.zeros(64)
    reward = reward_model.reward(vector)
    assert 0.0 < reward <= 1.0
    assert len(reward_model.forest.estimators_) == 100
    # 快速树遍历必须与sklearn公开decision_path得到的逐树边数严格一致。
    depths = [
        int(tree.decision_path(vector.reshape(1, -1)).getnnz(axis=1)[0]) - 1
        for tree in reward_model.forest.estimators_
    ]
    expected = 2.0 ** (-np.mean(depths) / reward_model._normalizer())
    np.testing.assert_allclose(reward, expected, rtol=0.0, atol=1e-15)

    base = Flow([
        Packet(500.0, 0.0, 1, 0, destination_port=443),
        Packet(420.0, 0.02, -1, 1, destination_port=443),
        Packet(600.0, 0.03, 1, 2, destination_port=443),
    ])
    manipulator = TrafficManipulator(config)
    for chosen in range(len(ACTION_NAMES)):
        # 前五维唯一最大值选择动作；强度0对应Tanh输出的中点。
        action = np.full(ACTION_DIMENSION, -1.0); action[chosen] = 1.0; action[3] = 0.0
        changed, cost, name = manipulator.apply(base, 0, action)
        assert name == ACTION_NAMES[chosen]
        assert changed.original_source_indices() == base.original_source_indices()
        assert changed.packets[0].destination_port == 443
        assert cost.action_counts[name] == 1


def test_fragment_accounting(config):
    """载荷字节守恒；新增头部及链路填充均反映在帧长与开销中。"""

    manipulator = TrafficManipulator(config)
    action = np.array([-1.0, 1.0, -1.0, 0.0])  # 3段
    base = Flow([Packet(501, 0.0, 1, 0)])  # 54头部+447载荷
    result, cost, _ = manipulator.apply(base, 0, action)
    assert [p.length for p in result.packets] == [203, 203, 203]
    assert [p.payload_bytes for p in result.packets] == [149, 149, 149]
    assert [p.payload_offset for p in result.packets] == [0, 149, 298]
    assert cost.added_header_bytes == cost.added_bytes == 108
    assert cost.added_link_padding_bytes == 0

    # 原来只有一个字节的以太网填充；三个新帧各需补足60字节。
    short = Flow([Packet(60, 0.0, 1, 0, payload_bytes=5)])
    result, cost, _ = manipulator.apply(short, 0, action)
    assert [p.payload_bytes for p in result.packets] == [2, 2, 1]
    assert [p.length for p in result.packets] == [60, 60, 60]
    assert cost.added_header_bytes == 108
    assert cost.added_link_padding_bytes == 12
    assert cost.added_bytes == 120

    # 真实头长不同（如TCP选项），每个额外头部也必须按该值计费。
    options = Flow([Packet(502, 0.0, 1, 0, header_bytes=62)])
    result, cost, _ = manipulator.apply(options, 0, action)
    assert cost.added_header_bytes == cost.added_bytes == 124
    assert sum(p.payload_bytes for p in result.packets) == 440

    for payload in (0, 1):
        no_split = Flow([Packet(60, 0.0, 1, 0, payload_bytes=payload)])
        result, cost, _ = manipulator.apply(no_split, 0, action)
        assert len(result.packets) == 1 and cost.added_bytes == 0
        assert cost.added_fragments == 0

    # 覆盖不能整除的载荷和最小帧边界；由字节区间重组证明没有丢字节/重叠。
    for length in (54, 55, 56, 59, 60, 61, 499, 500, 501, 1514):
        for strength in (-1.0, 0.0, 1.0):
            action[-1] = strength
            base = Flow([Packet(length, 0.0, 1, 0)])
            result, cost, _ = manipulator.apply(base, 0, action)
            pieces = result.packets
            assembled = [i for p in pieces for i in range(p.payload_offset, p.payload_offset + p.payload_bytes)]
            assert assembled == list(range(length - 54))
            assert all(p.length == int(p.length) for p in pieces)
            assert cost.added_bytes == sum(p.length for p in pieces) - length
            assert cost.added_bytes == cost.added_header_bytes + cost.added_link_padding_bytes

    # payload padding先消耗原有链路填充；分片不能再次把padding算作新增载荷。
    padded, pad_cost, _ = manipulator.apply(short, 0, np.array([1.0, -1.0, -1.0, -1.0]))
    assert pad_cost.added_payload_bytes == 200 and pad_cost.added_bytes == 199
    split, split_cost, _ = manipulator.apply(padded, 0, action)
    assert sum(p.padding_bytes for p in split.packets) == 200
    assert split_cost.added_payload_bytes == 0
    assert sum(p.length for p in split.packets) - 60 == pad_cost.added_bytes + split_cost.added_bytes


def test_full_flow_cost_and_metadata(config):
    """批量API传递真实头长，且开销计算覆盖被挤出32事件窗口的后续分段。"""

    from glimpse_seq64_oracle_v2.core import _DevicePolicy, _flow_from_row, _flow_to_row

    config = replace(config, reward_threshold=1.0)
    lengths = np.full((2, 32), 500.0)
    gaps = np.full((2, 32), 0.01); gaps[:, 0] = 0.0
    directions = np.ones((2, 32), dtype=int)
    mask = np.ones((2, 32), dtype=bool)
    labels = np.zeros(2, dtype=int)
    headers = np.full((2, 32), 62)
    payloads = lengths - headers
    model = IsolationForestRewardModel(config.seed)
    model.fit(make_features(lengths, gaps, directions, mask))
    policy = GlimpseSeq64OracleV2(config)
    policy.device_policies[0] = _DevicePolicy(model.forest, ())

    class FragmentPolicy:
        def policy_actions(self, states):
            return np.tile([-1.0, 1.0, -1.0, 1.0], (len(states), 1))

        def policy_action(self, state):
            return self.policy_actions([state])[0]

    policy._runtime_agents[0] = FragmentPolicy()
    output = apply_policy(policy, lengths, gaps, directions, mask, labels,
                          header_lengths=headers, payload_lengths=payloads)
    flow = _flow_from_row(lengths[0], gaps[0], directions[0], mask[0], headers[0], payloads[0])
    transformed, cost = policy.transform_flow(flow, 0)
    assert len(transformed.packets) == 62
    assert cost.added_bytes == cost.added_header_bytes == 10 * 3 * 62
    assert cost.added_bytes == sum(p.length for p in transformed.packets) - sum(lengths[0])
    assert output[4][0]["added_bytes"] == cost.added_bytes
    assert output[4][0]["action_counts"] == {"payload_padding": 0, "ip_tcp_fragmentation": 10, "packet_delay": 0}
    for field, row in enumerate(_flow_to_row(transformed)):
        np.testing.assert_allclose(output[field][0], row)
    assert output[5]["deployed_actor"] == "target"
    assert output[5]["adapter_choices"]["header_lengths_source"] == "provided"
    assert output[5]["adapter_choices"]["payload_lengths_source"] == "provided"


def test_delay_timestamps(config):
    """用可手算的发送时刻验证延迟、FIFO、双向重排及开销。"""

    from glimpse_seq64_oracle_v2.core import _flow_to_row

    manipulator = TrafficManipulator(config)

    def delay(flow, source, seconds):
        action = np.full(ACTION_DIMENSION, -1.0)
        action[2] = 1.0
        action[3] = 2.0 * seconds / config.max_delay_seconds - 1.0
        result, cost, _ = manipulator.apply(flow, source, action)
        np.testing.assert_allclose(cost.added_delay, seconds, atol=1e-15)
        assert result.packets[0].gap == 0.0
        assert _flow_to_row(result)[1][0] == 0.0
        assert result.original_source_indices() == flow.original_source_indices()
        return result

    def times_by_source(flow):
        times = flow.start_time + np.cumsum([p.gap for p in flow.packets])
        return {p.source_index: t for p, t in zip(flow.packets, times)}

    base = Flow([
        Packet(500.0, 0.0, 1, 0),
        Packet(420.0, 0.02, -1, 1),
        Packet(600.0, 0.03, 1, 2),
    ])
    changed = delay(base, 0, 0.01)
    np.testing.assert_allclose([p.gap for p in changed.packets], [0.0, 0.01, 0.03])
    np.testing.assert_allclose(changed.start_time, 0.01)
    np.testing.assert_allclose([times_by_source(changed)[i] for i in range(3)], [0.01, 0.02, 0.05])
    assert base.start_time == 0.0
    np.testing.assert_allclose([p.gap for p in base.packets], [0.0, 0.02, 0.03])

    # 首包被扣留时，原有下行事件可先出现；之后上行包只能排在队首之后。
    interleaved = Flow([
        Packet(500.0, 0.0, 1, 0),
        Packet(420.0, 0.02, -1, 1),
        Packet(600.0, 0.01, 1, 2),
        Packet(430.0, 0.01, -1, 3),
        Packet(610.0, 0.02, 1, 4),
    ])
    changed = delay(interleaved, 0, 0.05)
    assert [p.source_index for p in changed.packets] == [1, 3, 0, 2, 4]
    np.testing.assert_allclose([times_by_source(changed)[i] for i in range(5)], [0.05, 0.02, 0.05, 0.04, 0.06])
    np.testing.assert_allclose([p.gap for p in changed.packets], [0.0, 0.02, 0.01, 0.0, 0.01])

    # 连续动作必须保留同一时间原点；只影响当前及后续的上行包。
    changed = delay(changed, 2, 0.02)
    np.testing.assert_allclose([times_by_source(changed)[i] for i in range(5)], [0.05, 0.02, 0.07, 0.04, 0.07])
    assert [p.source_index for p in changed.packets if p.direction == 1] == [0, 2, 4]

    # 单包或整体平移的流：等待仍有成本，流内间隔不凭空增加。
    single = delay(Flow([Packet(500.0, 0.0, 1, 0)]), 0, 0.05)
    np.testing.assert_allclose(single.start_time, 0.05)
    np.testing.assert_array_equal(_flow_to_row(single)[1], np.zeros(32))
    unchanged = delay(interleaved, 0, 0.0)
    np.testing.assert_allclose([times_by_source(unchanged)[i] for i in range(5)], [0.0, 0.02, 0.03, 0.04, 0.06])

    # 分片后的相同时刻事件仍保持FIFO与长度总量。
    fragment_action = np.full(ACTION_DIMENSION, -1.0)
    fragment_action[1] = 1.0
    fragment_action[3] = 0.0
    fragmented, _, _ = manipulator.apply(base, 0, fragment_action)
    changed = delay(fragmented, 0, 0.06)
    assert [p.source_index for p in changed.packets] == [1, 0, 0, 0, 2]
    np.testing.assert_allclose(sum(p.length for p in changed.packets), sum(p.length for p in fragmented.packets))
    np.testing.assert_allclose(changed.start_time + np.cumsum([p.gap for p in changed.packets]), [0.02, 0.06, 0.06, 0.06, 0.06])


def test_legacy_delay_policy(config):
    """不能把按旧奖励/状态转移训练的策略静默用于新实验。"""

    legacy = GlimpseSeq64OracleV2(config)
    del legacy.policy_format
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.pkl"
        legacy.save(path)
        try:
            GlimpseSeq64OracleV2.load(path)
        except ValueError as exc:
            assert "重新训练" in str(exc)
        else:
            raise AssertionError("旧版延迟策略应要求重新训练")


def test_ddpg_update(config):
    """显式执行一次batch=32的MSBE更新，覆盖Actor/Critic和target网络路径。"""

    rng = np.random.default_rng(9)
    agent = DDPGMorphAgent(config)
    main_before = agent.main_actor_weights()
    target_before = agent.target_actor_weights()
    for main, target in zip(main_before, target_before):
        np.testing.assert_array_equal(main, target)
    # TF经验池按连续的两个time steps采样，因此建立10条、每条10步的短轨迹。
    for _ in range(10):
        state = rng.normal(size=64).astype(np.float32)
        current = agent.start(state)
        for ordinal in range(10):
            action, action_step = agent.collect_action(current)
            next_state = rng.normal(size=64).astype(np.float32)
            following = agent.next_time_step(
                next_state, reward=0.5, terminal=ordinal == 9
            )
            agent.remember(current, action_step, following)
            current = following
    loss = agent.update()
    assert np.isfinite(loss) and loss >= 0.0
    main_after = agent.main_actor_weights()
    target_after = agent.target_actor_weights()
    assert any(not np.array_equal(a, b) for a, b in zip(main_before, main_after))
    assert any(not np.array_equal(a, b) for a, b in zip(main_after, target_after))
    for before, main, target in zip(target_before, main_after, target_after):
        np.testing.assert_allclose(target, config.rho * before + (1.0 - config.rho) * main, atol=1e-7)

    # 故意让主Actor与目标Actor输出不同，证明部署路径确实使用theta_targ。
    states = rng.normal(size=(3, 64)).astype(np.float32)
    target_action = agent.policy_actions(states)
    main_zero = [np.zeros_like(w) for w in main_after]
    main_zero[-1].fill(0.8)
    agent._actor.set_weights(main_zero)
    np.testing.assert_allclose(agent.policy_actions(states), target_action)
    ts = agent._ts.restart(agent._observations(states), batch_size=3)
    assert not np.allclose(agent._agent.policy.action(ts).action.numpy(), target_action)
    restored = DDPGMorphAgent(config, training=False)
    restored.set_target_actor_weights(target_after)
    np.testing.assert_allclose(restored.policy_actions(states), target_action)


def test_end_to_end(config):
    # 至少120交互/设备，确保经过真实梯度与target软更新，而非只保存随机网络。
    config = replace(config, training_iterations=12, reward_threshold=1.0)
    arrays = make_synthetic_data(samples_per_class=8, seed=config.seed)
    lengths, gaps, directions, mask, labels = arrays
    train, test = stratified_split(labels, config.seed)
    policy = GlimpseSeq64OracleV2.fit(
        lengths, gaps, directions, mask, labels,
        config=config, train_indices=train,
    )
    output = apply_policy(
        policy, lengths[test], gaps[test], directions[test], mask[test], labels[test]
    )
    assert output[0].shape == lengths[test].shape
    assert len(output[4]) == len(test)
    assert output[5]["oracle_full_32_event_window"] is True
    assert output[5]["arima_used"] is False
    assert output[5]["adapter_choices"]["delay_model"] == "outbound_fifo_timestamps_v1"
    np.testing.assert_array_equal(output[1][:, 0], np.zeros(len(test)))
    assert set(policy.device_policies) == {0, 1, 2, 3}
    assert all(log["network_updates"] >= 1 for log in policy.training_log.values())
    assert all(log["deployed_actor"] == "target" for log in policy.training_log.values())
    from glimpse_seq64_oracle_v2.core import _flow_from_row, _flow_to_row
    single_rows = []
    for index in test:
        flow = _flow_from_row(lengths[index], gaps[index], directions[index], mask[index])
        transformed, _ = policy.transform_flow(flow, int(labels[index]))
        single_rows.append(_flow_to_row(transformed))
    for field in range(4):
        # Float32 actor batch kernels can differ by sub-nanosecond holds.
        np.testing.assert_allclose(output[field], np.stack([row[field] for row in single_rows]),
                                   rtol=0, atol=1e-9 if field == 1 else 0)

    # 序列化结果必须能在不访问原项目代码的前提下恢复并继续推理。
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.pkl"
        policy.save(path)
        restored = GlimpseSeq64OracleV2.load(path)
        restored_output = apply_policy(
            restored,
            lengths[test[:1]], gaps[test[:1]], directions[test[:1]],
            mask[test[:1]], labels[test[:1]],
        )
        for field in range(4):
            np.testing.assert_allclose(restored_output[field], output[field][:1],
                                       rtol=0, atol=1e-9 if field == 1 else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="只测特征、Isolation Forest与三动作，不加载TensorFlow-Agents",
    )
    args = parser.parse_args()
    # 以下数值只为了让自测试快速跑通，不代表论文推荐或本文实验设置。
    config = GlimpseConfig(
        gamma=0.9,
        rho=0.95,
        reward_threshold=0.7,
        training_iterations=2,
        max_delay_seconds=0.1,
        max_fragments=4,
        target_update_period=1,
        replay_buffer_capacity=256,
        exploration_noise_stddev=0.1,
        exploration_noise_damping=0.15,
        seed=7,
    )
    test_features()
    test_reward_and_actions(config)
    test_fragment_accounting(config)
    test_full_flow_cost_and_metadata(config)
    test_delay_timestamps(config)
    test_legacy_delay_policy(config)
    if args.core_only:
        print("Glimpse-Seq64-Oracle-v2 核心自测试通过（未测试DDPG）")
        return
    try:
        test_ddpg_update(config)
        test_end_to_end(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print("Glimpse-Seq64-Oracle-v2 完整自测试全部通过")


if __name__ == "__main__":
    main()

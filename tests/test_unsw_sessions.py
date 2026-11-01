"""Focused UNSW session-boundary and protocol regression tests."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

import numpy as np

from data_pipeline import pcap
from data_pipeline.unsw import (
    FlowAccumulator,
    UNSW_DATA_PROTOCOL,
    _date_group,
    _dataset_from_samples,
    _session_boundary_reason,
    _session_terminal_reason,
)


FLOW_KEY = (pcap._TCP, (b"\x0a\x00\x00\x01", 12345), (b"\x0a\x00\x00\x02", 443))


def _packet(
    timestamp: float,
    *,
    direction: int = 1,
    flags: int = 0,
    sequence: int = 1,
    frame_id: int = 1,
) -> pcap.ParsedPacket:
    return pcap.ParsedPacket(
        flow_key=FLOW_KEY,
        timestamp=float(timestamp),
        direction=int(direction),
        frame_length=100,
        source_port=12345,
        destination_port=443,
        is_tls_record=True,
        frame_id=int(frame_id),
        tcp_flags=int(flags),
        tcp_sequence=int(sequence),
    )


class UnswSessionTests(unittest.TestCase):
    def test_same_sequence_syn_retransmission_is_not_a_boundary(self):
        flow = FlowAccumulator.start(_packet(100.0, flags=0x02, sequence=700), 32)
        retransmission = _packet(101.0, flags=0x02, sequence=700, frame_id=2)
        self.assertIsNone(_session_boundary_reason(flow, retransmission, 120.0))
        flow.add(retransmission, 32)

        syn_ack = _packet(
            102.0, direction=-1, flags=0x12, sequence=900, frame_id=3
        )
        self.assertIsNone(_session_boundary_reason(flow, syn_ack, 120.0))
        flow.add(syn_ack, 32)
        syn_ack_retransmission = _packet(
            103.0, direction=-1, flags=0x12, sequence=900, frame_id=4
        )
        self.assertIsNone(
            _session_boundary_reason(flow, syn_ack_retransmission, 120.0)
        )
        flow.add(syn_ack_retransmission, 32)

        new_syn = _packet(104.0, flags=0x02, sequence=701, frame_id=5)
        self.assertEqual(_session_boundary_reason(flow, new_syn, 120.0), "syn")

    def test_timeout_timestamp_regression_and_utc_day_are_boundaries(self):
        flow = FlowAccumulator.start(_packet(100.0), 32)
        self.assertEqual(
            _session_boundary_reason(flow, _packet(106.0, frame_id=2), 5.0),
            "inactivity",
        )
        self.assertEqual(
            _session_boundary_reason(flow, _packet(99.5, frame_id=3), 120.0),
            "timestamp_regression",
        )

        late = datetime(2018, 1, 1, 23, 59, 59, tzinfo=timezone.utc).timestamp()
        day_flow = FlowAccumulator.start(_packet(late, frame_id=4), 32)
        next_day = _packet(late + 2.0, frame_id=5)
        self.assertEqual(
            _session_boundary_reason(day_flow, next_day, 120.0), "utc_day"
        )
        self.assertEqual(_date_group("iHome", late), "2018-01-01")

    def test_rst_and_bidirectional_fin_close_after_terminal_packet(self):
        rst_flow = FlowAccumulator.start(_packet(100.0), 32)
        rst = _packet(101.0, flags=0x04, frame_id=2)
        rst_flow.add(rst, 32)
        self.assertEqual(_session_terminal_reason(rst_flow, rst), "rst")

        fin_flow = FlowAccumulator.start(_packet(200.0), 32)
        first_fin = _packet(201.0, direction=1, flags=0x01, frame_id=2)
        fin_flow.add(first_fin, 32)
        self.assertIsNone(_session_terminal_reason(fin_flow, first_fin))
        second_fin = _packet(202.0, direction=-1, flags=0x11, frame_id=3)
        fin_flow.add(second_fin, 32)
        self.assertEqual(_session_terminal_reason(fin_flow, second_fin), "double_fin")

    def test_dataset_uses_global_day_capture_ids_and_protocol(self):
        width = 4
        values = np.asarray([100.0, 101.0, 102.0, 103.0])
        sample = pcap.FlowSample(
            lengths=np.full(width, 100.0, dtype=np.float32),
            timestamps=values,
            frame_ids=np.arange(1, width + 1, dtype=np.int32),
            source_ips=np.full(width, "10.0.0.1", dtype="<U15"),
            destination_ips=np.full(width, "10.0.0.2", dtype="<U15"),
            gaps=np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            directions=np.asarray([1, -1, 1, -1], dtype=np.int8),
            mask=np.ones(width, dtype=np.int8),
            label=0,
            group="2018-01-01",
        )
        manifest = {"data_protocol": UNSW_DATA_PROTOCOL}
        dataset = _dataset_from_samples([sample], ["iHome"], manifest)
        self.assertEqual(dataset["capture_ids"].tolist(), ["2018-01-01"])
        self.assertEqual(str(dataset["data_protocol"]), UNSW_DATA_PROTOCOL)
        self.assertEqual(
            json.loads(str(dataset["unsw_extraction_manifest_json"]))["data_protocol"],
            UNSW_DATA_PROTOCOL,
        )


if __name__ == "__main__":
    unittest.main()

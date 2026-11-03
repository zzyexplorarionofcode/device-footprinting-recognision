import unittest

from scripts.event_canonical import Event, forward, inverse


def make_events(n=32, trigger_slots=(3, 9)):
    events = []
    for i in range(n):
        events.append(Event(f"real-{i}", "real", source_index=i, attrs={"length": 100 + i}))
        if (i + 1) in trigger_slots:
            events.append(Event(f"trigger-{i + 1}", "trigger", slot=i + 1, attrs={"code": i + 1}))
        if i in (4, 12):
            events.append(Event(f"weakener-{i}", "weakener", attrs={"length": 80}))
    return events


class SyntheticRoundTripTests(unittest.TestCase):
    def test_no_truncation_round_trip(self):
        reference = make_events(24, trigger_slots=(3, 9))
        self.assertLessEqual(len(reference), 32)
        actual = forward(reference, actual_slots=[7, 15])
        self.assertEqual(actual.lost, [])
        back = inverse(actual.events, reference_slots=[3, 9])
        self.assertEqual([e.event_id for e in back.events], [e.event_id for e in reference])
        self.assertEqual([e.role for e in back.events], [e.role for e in reference])
        self.assertEqual([e.attrs for e in back.events], [e.attrs for e in reference])
        self.assertEqual(back.lost, [])

    def test_truncation_marks_missing_events(self):
        reference = make_events(32, trigger_slots=(3, 9))
        actual = forward(reference, actual_slots=[30, 31])
        self.assertTrue(actual.lost)
        lost_ids = {e.event_id for e in actual.lost}
        self.assertTrue(any(e.role == "trigger" for e in actual.lost))
        recovered_ids = {e.event_id for e in actual.events}
        self.assertTrue(lost_ids.isdisjoint(recovered_ids))
        # If a trigger itself is truncated, inverse cannot be complete.
        retained_triggers = [e for e in actual.events if e.role == "trigger"]
        if len(retained_triggers) == 2:
            back = inverse(actual.events, reference_slots=[3, 9])
            self.assertTrue(any(e.role == "trigger" for e in back.events))
            self.assertLessEqual(len(back.events), 32)
        else:
            self.assertLess(len(retained_triggers), 2)

    def test_short_flow_fallback_is_explicit(self):
        short = [Event(f"real-{i}", "real", source_index=i) for i in range(5)]
        short.extend([
            Event("trigger-a", "trigger", slot=1),
            Event("trigger-b", "trigger", slot=10),
        ])
        actual = forward(short, actual_slots=[1, 10])
        self.assertEqual(actual.fallback_slots, [10])
        self.assertEqual([e.event_id for e in actual.events if e.role == "trigger"], ["trigger-a", "trigger-b"])


if __name__ == "__main__":
    unittest.main()

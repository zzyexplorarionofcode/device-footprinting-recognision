import unittest

import numpy as np

from methods import method33_tools


class CanonicalizationRoundTripTests(unittest.TestCase):
    def test_mapping_inverse_round_trip(self):
        project = method33_tools.load_config("configs/methods/fade_unsw8_pos_v2.json")
        method = project.method33
        mappings = [list(method.old_mapping), list(method.next_mapping)]
        for mapping in mappings:
            self.assertEqual(sorted(mapping), list(range(method.class_count)))
            inverse = {code: cls for cls, code in enumerate(mapping)}
            self.assertEqual(len(inverse), method.class_count)
            for cls in range(method.class_count):
                self.assertEqual(inverse[mapping[cls]], cls)

    def test_code_targets_are_state_dependent_but_invertible(self):
        project = method33_tools.load_config("configs/methods/fade_unsw8_pos_v2.json")
        method = project.method33
        old = list(method.old_mapping)
        nxt = list(method.next_mapping)
        for cls in range(method.class_count):
            self.assertNotEqual(old[cls], nxt[cls])
            self.assertEqual({old[cls]: cls for cls in range(method.class_count)}[old[cls]], cls)
            self.assertEqual({nxt[cls]: cls for cls in range(method.class_count)}[nxt[cls]], cls)


if __name__ == "__main__":
    unittest.main()

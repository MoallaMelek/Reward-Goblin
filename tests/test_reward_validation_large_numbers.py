"""JSON integers can exceed float range; validation must return errors safely."""
import unittest
from backend.env.rewards import RewardConfig, validate


class LargeNumberTests(unittest.TestCase):
    def test_huge_integer_weights_and_step_limits_are_rejected(self):
        for value in (10 ** 1000, -(10 ** 1000)):
            for payload in ({'components': {'goal_contact': value}},
                            {'goal_contact': 1, 'max_steps': value}):
                with self.subTest(payload=payload):
                    self.assertTrue(validate(payload)[0])
                    with self.assertRaises(ValueError):
                        RewardConfig.from_dict(payload)

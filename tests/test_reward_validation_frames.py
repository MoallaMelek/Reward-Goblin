"""A duration in simulation frames cannot silently discard a fractional frame."""
import unittest
from backend.env.rewards import RewardConfig, validate


class RewardFrameTests(unittest.TestCase):
    def test_fractional_frames_rejected(self):
        for value in (1.5, 30.9, 299.99):
            for structured in (True, False):
                with self.subTest(value=value, structured=structured):
                    payload = {'components': {'goal_contact': 1}}
                    payload.update({'params': {'stable_goal_frames': value}} if structured else {'stable_goal_frames': value})
                    self.assertTrue(any('must be an integer' in e for e in validate(payload)[0]))
                    with self.assertRaisesRegex(ValueError, 'must be an integer'):
                        RewardConfig.from_dict(payload)

    def test_whole_float_frames_remain_accepted(self):
        self.assertEqual(RewardConfig.from_dict({'goal_contact': 1, 'stable_goal_frames': 30.0}).stable_frames, 30)

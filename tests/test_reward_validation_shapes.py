"""Reward validation must reject malformed JSON without throwing internal errors."""
import unittest

from backend.env.rewards import RewardConfig, validate


class RewardShapeTests(unittest.TestCase):
    def test_nested_sections_must_be_objects(self):
        for field in ('components', 'params', 'termination'):
            for value in (None, 1, 'invalid', [], [['goal_contact', 1]]):
                with self.subTest(field=field, value=value):
                    payload = {'components': {'goal_contact': 1}, field: value}
                    errors, warnings = validate(payload)
                    self.assertIn(f'{field} must be a JSON object', errors)
                    self.assertEqual(warnings, [])
                    with self.assertRaisesRegex(ValueError, f'{field} must be a JSON object'):
                        RewardConfig.from_dict(payload)

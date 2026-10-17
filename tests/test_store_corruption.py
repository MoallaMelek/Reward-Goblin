"""A corrupt record should not prevent browsing healthy training runs."""
import tempfile
import unittest
from pathlib import Path

from backend.api.store import Store


class StoreCorruptionTests(unittest.TestCase):
    def test_unreadable_and_non_object_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Store(root)
            for run_id, content in [('binary', b'\xff\xfe'), ('scalar', b'42'), ('array', b'[1]'),
                                    ('broken', b'{'), ('good', b'{"run_id":"good"}')]:
                path = root / 'runs' / run_id
                path.mkdir(parents=True)
                (path / 'meta.json').write_bytes(content)
            self.assertEqual([r['run_id'] for r in store.list_runs()], ['good'])
            for run_id in ('binary', 'scalar', 'array', 'broken'):
                self.assertIsNone(store.meta(run_id))

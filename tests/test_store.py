from backend.api.store import Store, SAFE_ID


def test_ids_reject_trailing_newlines(tmp_path):
    assert SAFE_ID.match("run_1")
    assert not SAFE_ID.match("run_1\n")
    store = Store(tmp_path)
    assert store.meta("run_1\n") is None
    assert store.recording("run_1", "best\n") is None

from backend.api.store import Store, SAFE_ID


def test_ids_reject_trailing_newlines(tmp_path):
    assert SAFE_ID.match("run_1")
    assert not SAFE_ID.match("run_1\n")
    store = Store(tmp_path)
    assert store.meta("run_1\n") is None
    assert store.recording("run_1", "best\n") is None


def test_invalid_utf8_run_is_skipped(tmp_path):
    run = tmp_path / "runs" / "broken"
    run.mkdir(parents=True)
    (run / "meta.json").write_bytes(b"\xff")
    store = Store(tmp_path)
    assert store.meta("broken") is None
    assert store.list_runs() == []

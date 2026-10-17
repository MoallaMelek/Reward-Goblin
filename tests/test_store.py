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


def test_pooled_outscores_reference_rate_uses_only_paired_episodes():
    from backend.api.store import _pool
    episodes = [
        {"return": 10, "reference_return": 5, "exploits": [], "true_success": True},
        {"return": 1, "reference_return": 5, "exploits": [], "true_success": False},
        {"return": 100, "exploits": [], "true_success": True},
    ]
    assert _pool(episodes)["outscores_reference_rate"] == 0.5
    assert "outscores_reference_rate" not in _pool(episodes[2:])

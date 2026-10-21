from backend.api.store import Store

def test_recording_kinds_exclude_unreadable_names_and_directories(tmp_path):
    folder=tmp_path/'recordings'/'run';folder.mkdir(parents=True)
    (folder/'highlight.json').write_text('{}')
    (folder/'bad name.json').write_text('{}')
    (folder/'directory.json').mkdir()
    assert Store(tmp_path).recording_kinds('run')==['highlight']
    assert Store(tmp_path).recording_kinds('../run')==[]

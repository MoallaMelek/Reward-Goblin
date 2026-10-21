from backend.replay.recorder import save_recording, load_recording

def test_utf8_recording_from_external_tools_is_readable(tmp_path):
    path=tmp_path/'replay.json'
    path.write_bytes('{"label":"Déclic 🎵"}'.encode('utf-8'))
    assert load_recording(path)=={'label':'Déclic 🎵'}
    save_recording({'label':'Déclic 🎵'},path)
    assert load_recording(path)=={'label':'Déclic 🎵'}

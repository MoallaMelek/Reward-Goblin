from backend.api.store import Store

def test_listing_does_not_advertise_unreadable_run_ids(tmp_path):
    for name in ['valid-run_1','space name','_private']:
        (tmp_path/'runs'/name).mkdir(parents=True)
    assert Store(tmp_path).run_ids()==['valid-run_1']

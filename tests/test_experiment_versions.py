import pytest
from backend.training.train import normalize_config

@pytest.mark.parametrize('version',[True,False,0,-1,1.5,'2',None])
def test_invalid_versions_are_not_coerced(version):
    with pytest.raises(ValueError,match='version'):
        normalize_config({'reward':{'components':{'distance_progress':1.0}},'version':version})

def test_valid_version_is_preserved():
    assert normalize_config({'reward':{'components':{'distance_progress':1.0}},'version':3})['version']==3

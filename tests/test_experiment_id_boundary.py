import pytest
from backend.training.train import normalize_config

@pytest.mark.parametrize('identifier', ['../escape','a/b','a\\b','',None,1,'x'*81])
def test_cli_config_rejects_unsafe_experiment_identifiers(identifier):
    with pytest.raises(ValueError,match='experiment'):
        normalize_config({'reward':{'components':{'distance_progress':1.0}},'experiment':identifier})

def test_safe_experiment_id_is_preserved():
    assert normalize_config({'reward':{'components':{'distance_progress':1.0}},'experiment':'my_test-2'})['experiment']=='my_test-2'

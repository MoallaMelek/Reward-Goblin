from backend.training.train import normalize_config, ALGO_DEFAULTS

def test_normalized_default_does_not_mutate_global_defaults():
    expected=list(ALGO_DEFAULTS['PPO']['net_arch'])
    config=normalize_config({'components':{'distance_progress':1.0}})
    config['algo']['net_arch'].append(32)
    assert ALGO_DEFAULTS['PPO']['net_arch']==expected
    assert normalize_config({'components':{'distance_progress':1.0}})['algo']['net_arch']==expected

def test_normalized_custom_architecture_does_not_alias_input():
    layers=[128,32]
    config=normalize_config({'reward':{'components':{'distance_progress':1.0}},'algo':{'net_arch':layers}})
    config['algo']['net_arch'].append(16)
    assert layers==[128,32]

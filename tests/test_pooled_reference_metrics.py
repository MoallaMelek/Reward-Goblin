import pytest
from backend.api.store import _pool

def test_only_paired_episodes_contribute_to_reference_comparisons():
    episodes=[
        {'return':10,'true_success':True,'exploits':[],'reference_return':5,'reference_success':True},
        {'return':0,'true_success':False,'exploits':['stalling'],'reference_return':5,'reference_success':False},
        {'return':100,'true_success':False,'exploits':['stalling']},
    ]
    result=_pool(episodes)
    assert result['episodes']==3
    assert result['mean_reference_return']==5
    assert result['outscores_reference_rate']==.5
    assert result['reference_success_rate']==.5
    assert result['true_success_rate']==pytest.approx(1/3)
    assert result['exploit_rate']==pytest.approx(2/3)

def test_unpaired_evaluation_does_not_invent_reference_scores():
    result=_pool([{'return':0,'true_success':False,'exploits':[]}])
    assert result['mean_reference_return'] is None
    assert 'outscores_reference_rate' not in result
    assert _pool([])=={'episodes':0}

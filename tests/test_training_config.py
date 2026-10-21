import pytest

from backend.training.train import normalize_config

REWARD = {"components": {"distance_progress": 1.0}}


@pytest.mark.parametrize("steps", [1000.5, "1000", True, None])
def test_steps_must_be_integer(steps):
    with pytest.raises(ValueError, match="steps must be an integer"):
        normalize_config({"reward": REWARD, "steps": steps})


@pytest.mark.parametrize("key,value", [("n_steps", 8.5), ("batch_size", 16.5), ("n_envs", 1.5), ("n_epochs", 1.5)])
def test_integer_hyperparameters_are_not_truncated(key, value):
    with pytest.raises(ValueError, match="must be an integer"):
        normalize_config({"reward": REWARD, "algo": {key: value}})


@pytest.mark.parametrize("layers", [None, "64", [], [0], [-1], [True], [64.5], [4097], [64] * 9])
def test_invalid_network_architecture_rejected_before_worker_start(layers):
    with pytest.raises(ValueError, match="net_arch"):
        normalize_config({"reward": REWARD, "algo": {"net_arch": layers}})


def test_valid_custom_network_architecture():
    assert normalize_config({"reward": REWARD, "algo": {"net_arch": [128, 32]}})["algo"]["net_arch"] == [128, 32]


@pytest.mark.parametrize("field,value", [("steps", True), ("steps", "1000"), ("seeds", [True]), ("seeds", ["2"])])
def test_training_request_does_not_coerce_counts(field, value):
    from backend.api.main import TrainBody
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrainBody.model_validate({"config": {}, field: value})

@pytest.mark.parametrize("episodes", [True, "3", 0, 51])
def test_evaluation_request_rejects_invalid_episode_counts(episodes):
    from backend.api.main import EvalBody
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvalBody(run_id="run", episodes=episodes)


@pytest.mark.parametrize('suites',[[],{},[None],[{'name':'x','layout':'arena','episodes':0}],[{'name':'x','layout':'arena','episodes':True}],[{'name':'x','layout':'unknown','episodes':2}],[{'name':'x','layout':'arena','episodes':2,'offset':-1}],[{'name':'x','layout':'arena','episodes':2}]*2])
def test_invalid_evaluation_plan_is_rejected_before_training(suites):
    with pytest.raises(ValueError):normalize_config({'reward':REWARD,'eval':suites})

def test_custom_evaluation_plan_is_preserved_without_aliasing():
    suites=[{'name':'held_out','layout':'medium','episodes':10,'offset':5000}]
    normalized=normalize_config({'reward':REWARD,'eval':suites})
    assert normalized['eval']==suites
    normalized['eval'][0]['episodes']=2
    assert suites[0]['episodes']==10

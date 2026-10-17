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

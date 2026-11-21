"""The gymnasium VectorEnv (skipped without gymnasium)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from wingbatch.envs import LevelFlightVectorEnv  # noqa: E402


def test_vector_env_api_and_autoreset():
    env = LevelFlightVectorEnv(num_envs=8, max_episode_steps=20)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,) + env.single_observation_space.shape and obs.dtype == np.float32
    assert env.action_space.shape == (8, 5)  # 4 control surfaces + 1 motor
    ended = np.zeros(8, bool)
    for k in range(25):
        a = np.zeros((8, 5), np.float32)
        a[:, 4] = -0.4  # about trim throttle
        obs, r, term, trunc, info = env.step(a)
        assert obs.shape[0] == 8 and r.shape == (8,) and np.all(np.isfinite(obs))
        if k == 19:
            assert np.all(trunc | term)
            ended = trunc | term
        if k == 20:  # NEXT_STEP: the ended envs restart, reward 0, not done
            assert np.all(r[ended] == 0) and not np.any(term[ended] | trunc[ended])
    env2 = LevelFlightVectorEnv(num_envs=8)
    o1, _ = env2.reset(seed=3)
    o2, _ = env2.reset(seed=3)
    np.testing.assert_array_equal(o1, o2)


def test_full_aileron_ends_the_episode():
    env = LevelFlightVectorEnv(num_envs=4, max_episode_steps=2000)
    env.reset(seed=1)
    a = np.zeros((4, 5), np.float32)
    a[:, 0], a[:, 1] = 1.0, -1.0  # left elevon down... full roll command
    done = np.zeros(4, bool)
    for _ in range(400):
        _, _, term, trunc, _ = env.step(a)
        done |= term
    assert np.all(done)

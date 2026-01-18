import pytest
from overcooked_ai_py.mdp.overcooked_env import Overcooked, OvercookedEnv, OvercookedGridworld


@pytest.fixture
def env_setup():
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(mdp, info_level=1, horizon=400)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    obs = env.reset()
    overcooked_state = obs["overcooked_state"]

    return mdp, base_env, overcooked_state


# ---------------------------------------------------------
# FEATURE SHAPE CALCULATION
# ---------------------------------------------------------
# Formula: Total = (Features * N_players) + (2 * (N_players - 1)) + 2
# Where:   Features = 26 + (10 * N_pots)
#
# Constants:
#    26 = Base Features (Orientation, Held Obj, Walls, etc.)
#    10 = Per Pot Features
#
# Example (2 Players, 1 Pot):
#    Features = 26 + 10 = 36
#    Total = (36 * 2) + 2 + 2 = 76
# ---------------------------------------------------------
def test_featurize_state_shape_num_pots_1(env_setup):
    mdp, base_env, state = env_setup

    features = base_env.featurize_state_mdp(state, num_pots=1)
    p1_features, p2_features = features

    assert p1_features.shape[0] == 76
    assert p2_features.shape[0] == 76


def test_featurize_state_shape_num_pots_2(env_setup):
    mdp, base_env, state = env_setup

    features = base_env.featurize_state_mdp(state, num_pots=2)
    p1_features, p2_features = features

    assert p1_features.shape[0] == 96
    assert p2_features.shape[0] == 96


def test_lossless_state_encoding_shape(env_setup):
    mdp, base_env, state = env_setup

    grid_encoding = mdp.lossless_state_encoding(state)

    # Cramped Room dimensions:
    # Width = 5, Height = 4
    # Channels = 26 (features per tile)
    expected_shape = (5, 4, 26)
    assert grid_encoding[0].shape == expected_shape

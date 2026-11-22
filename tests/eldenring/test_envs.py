"""Tests for the Margit environment.

The live scenarios reset the game, teleport the player, press keys and kill the player when the
environment closes. They are marked ``integration`` and ``manual_destructive`` and only run on
request. They are skipped if the game is not running or the player is not loaded into the game. The
skip condition is evaluated lazily when a test runs, so collecting or deselecting the tests never
attaches to a running game.

Run the scenarios one at a time with Elden Ring in offline mode without EAC, the player loaded into
the game and the game window at 800x450. Do not use the keyboard or the mouse while a scenario runs
and check the game afterwards.

The remaining tests run without a game. They replace the game, the game input and the game window
by fakes.
"""

import logging
import time
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

import soulsgym  # noqa: F401, registers the environments
from soulsgym.envs.eldenring.margit import MargitEnv, MargitState
from soulsgym.envs.utils import RetriesExceededError
from soulsgym.exception import (
    GameStateError,
    LockOnFailure,
    MemoryReadError,
    ResetError,
    ResetNeeded,
)
from soulsgym.games.eldenring import EldenRing
from soulsgym.games.game import StaticGameData
from tests.eldenring.utils import (
    BOSS_MAX_HP,
    COORDS,
    PLAYER_IDLE_ANIMATION,
    PLAYER_MAX_HP,
    FakeGame,
    FakeInput,
    game_not_open,  # noqa: F401, used by the skipif conditions
    simulated_game,
)

NO_OP_ACTION = 19
ACTION_SEQUENCE = (16, 17, 8)
ACTIONS = StaticGameData("EldenRing").actions
RESPAWN_TIMEOUT_S = 30.0


def test_margit_expected_action_ids_exist():
    """Ensure the action IDs used in smoke checks exist in the action map."""
    assert all(action in ACTIONS for action in (*ACTION_SEQUENCE, NO_OP_ACTION))


def _wait_for_respawn(game: EldenRing):
    """Wait until the player is back in the game and idle after closing the environment.

    The environment does not wait for the reload on close. Without this wait, the next scenario
    would attach to a game that is still reloading.
    """
    t_start = time.time()
    while time.time() - t_start < RESPAWN_TIMEOUT_S:
        game.clear_cache()  # Death reloads invalidate the cached addresses
        try:
            idle = game.player_animation == PLAYER_IDLE_ANIMATION
            if game.is_ingame and game.player_hp > 0 and idle:
                return
        except MemoryReadError:  # Reads during the reload might fail
            pass
        time.sleep(0.1)
    raise RuntimeError("Elden Ring did not respawn the player after the environment was closed")


@pytest.fixture
def margit_env():
    """Create a live Margit env for bounded integration checks."""
    try:
        env = gym.make("SoulsGymMargit-v0", game_speed=3.0, random_player_pose=True)
    except (GameStateError, MemoryReadError, RuntimeError) as exc:
        pytest.skip(f"Elden Ring is not in a stable in-game state: {exc}")
    try:
        yield env
    finally:
        env.close()
        _wait_for_respawn(env.unwrapped.game)


@pytest.mark.manual_destructive
@pytest.mark.integration
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
def test_margit_env_creation_integration(margit_env):
    """Destructively verify live environment creation and teardown spaces."""
    assert margit_env.action_space is not None
    assert margit_env.observation_space is not None


@pytest.mark.manual_destructive
@pytest.mark.integration
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
def test_margit_reset_only_profile(margit_env):
    """Run two reset-only cycles and verify reset contract output shape."""
    for _ in range(2):
        obs, info = margit_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "player_hp" in obs
        assert "boss_hp" in obs


@pytest.mark.manual_destructive
@pytest.mark.integration
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
def test_margit_one_step_profile(margit_env):
    """Run one bounded no-op step after reset and verify Gymnasium return tuple."""
    margit_env.reset()
    obs, reward, terminated, truncated, info = margit_env.step(NO_OP_ACTION)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def _is_action_ready(info: dict, action: int) -> bool:
    """Return whether an action is currently valid according to the action mask."""
    mask = info.get("action_mask")
    if mask is None:
        return True
    return bool(mask[action])


def _wait_until_action_ready(env: Any, action: int, max_wait_steps: int = 24) -> tuple:
    """Advance with safe no-op steps until an action is valid again."""
    obs: dict | None = None
    terminated = False
    truncated = False
    for _ in range(max_wait_steps):
        obs, _, terminated, truncated, info = env.step(NO_OP_ACTION)
        if terminated or truncated:
            return obs, terminated, truncated, False
        if _is_action_ready(info, action):
            return obs, False, False, True
    return obs, False, False, False


@pytest.mark.manual_destructive
@pytest.mark.integration
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
def test_margit_action_sequence_profile(margit_env):
    """Run one bounded action sequence cycle and assert it does not error."""
    obs, _ = margit_env.reset()
    assert "player_hp" in obs

    unwrapped = margit_env.unwrapped
    if hasattr(unwrapped, "_lock_on"):
        unwrapped._lock_on()
        unwrapped._game_input.update_input()

    for action in ACTION_SEQUENCE:
        _, _, terminated, truncated, _ = margit_env.step(action)
        if terminated or truncated:
            pytest.fail(
                f"Unexpected terminal state during action sequence: action={action}, "
                f"terminated={terminated}, truncated={truncated}"
            )
        _, terminated, truncated, ready = _wait_until_action_ready(margit_env, action)
        if terminated or truncated:
            pytest.fail(
                f"Unexpected terminal state while settling action {action}: "
                f"terminated={terminated}, truncated={truncated}"
            )
        assert ready, f"Action {action} did not become ready within settle timeout"


@pytest.mark.manual_destructive
@pytest.mark.integration
@pytest.mark.skipif("game_not_open()", reason="Elden Ring is not running.")
def test_margit_random_agent_episodes(margit_env):
    """Run three consecutive random agent episodes until they terminate or are truncated.

    A random agent is expected to die within a short time. At the latest the time limit of the
    registered environment ends an episode after 3000 steps.
    """
    for _ in range(3):
        margit_env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = margit_env.step(margit_env.action_space.sample())


# The tests without a game

# Debug flags of the Dark Souls III environments that are not mapped for Elden Ring
UNMAPPED_DEBUG_FLAGS = set(
    "allow_attacks allow_hits allow_moves allow_deaths allow_weapon_durability_dmg "
    "lock_on_bonus_range los_lock_on_deactivate_time".split()
)
POST_FOG_WALL = COORDS["post_fog_wall"]


@pytest.fixture
def fakes(monkeypatch):
    """Replace the game, the game input and the game window of the environments by fakes."""
    game = FakeGame()
    monkeypatch.setattr("soulsgym.envs.eldenring.margit.time", game.clock)
    mocks = {
        "game_factory": Mock(return_value=game),
        "GameInput": Mock(return_value=FakeInput()),
        "GameWindow": Mock(return_value=Mock(focused=True)),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(f"soulsgym.envs.soulsenv.{name}", mock)
    return mocks


@pytest.fixture
def env(fakes):
    return MargitEnv()


def _reset(env: gym.Env) -> tuple[dict, dict]:
    """Reset the environment. The fake game reacts to the fog gate interaction and the lock on."""
    unwrapped = env.unwrapped
    game = unwrapped.game
    unwrapped._game_input.on_action["interact"] = lambda: vars(game).update(
        player_pose=POST_FOG_WALL.copy()
    )
    unwrapped._lock_on = lambda: vars(game).update(lock_on=True)
    return env.reset()


def test_elden_ring_does_not_fake_unmapped_debug_flags():
    # The weapon durability is the exception. Its property exists, but its setter does nothing
    flags = UNMAPPED_DEBUG_FLAGS - {"allow_weapon_durability_dmg"}
    assert not {flag for flag in flags if hasattr(EldenRing, flag)}


def test_environment_only_sets_mapped_game_properties(env):
    game = env.game
    assert game.writes == ["last_bonfire", "player_stats"]
    assert game.calls == ["resume"]
    assert game.last_bonfire == MargitEnv.BONFIRE
    assert game.player_stats == game.data.player_stats["margit"]


@pytest.mark.parametrize("resolution", [None, (1024, 576)])
def test_environment_forwards_the_game_resolution_to_the_game_and_the_window(fakes, resolution):
    MargitEnv(game_resolution=resolution)
    fakes["game_factory"].assert_called_once_with("EldenRing", game_resolution=resolution)
    fakes["GameWindow"].assert_called_once_with("EldenRing", game_resolution=resolution)


@pytest.mark.parametrize("phase", [2, 3])
def test_only_phase_1_is_supported(fakes, phase):
    with pytest.raises(NotImplementedError, match="phase 1"):
        MargitEnv(phase=phase)

    fakes["game_factory"].assert_not_called()


def test_spaces_match_the_static_data(env):
    data = env.game.data
    animations = data.boss_animations["margit"]["all"]

    assert set(env.observation_space.spaces) == set(
        "phase player_hp player_max_hp player_sp player_max_sp player_mp player_max_mp boss_hp "
        "boss_max_hp player_pose boss_pose camera_pose player_animation player_animation_duration "
        "boss_animation boss_animation_duration lock_on".split()
    )
    assert env.action_space.n == len(data.actions)
    assert env.observation_space["player_animation"].n == len(data.player_animations) + 1
    assert env.observation_space["boss_animation"].n == len(animations) + 1
    # The indices of the boss animations are contiguous. -1 denotes an unknown animation
    indices = sorted(animation["ID"] for animation in animations.values())
    assert indices == list(range(len(animations)))
    space = env.observation_space["boss_animation"]
    assert all(space.contains(index) for index in [-1, *indices])
    assert not space.contains(len(animations))


@pytest.mark.filterwarnings("error::UserWarning")  # The passive environment checker warns
def test_margit_env_is_registered_and_truncated_by_the_time_limit(fakes):
    spec = gym.spec("SoulsGymMargit-v0")
    assert spec.nondeterministic
    env = gym.make(spec.id)
    assert isinstance(env, gym.wrappers.TimeLimit)
    assert isinstance(env.unwrapped, MargitEnv)
    assert env.spec.max_episode_steps == spec.max_episode_steps == 3000

    env = gym.make(spec.id, max_episode_steps=3)
    _reset(env)
    results = [env.step(NO_OP_ACTION) for _ in range(3)]
    env.close()

    assert [terminated for _, _, terminated, _, _ in results] == [False, False, False]
    assert [truncated for _, _, _, truncated, _ in results] == [False, False, True]


def test_reset_enters_the_arena_without_touching_debug_flags(env):
    game, game_input = env.game, env._game_input

    obs, info = _reset(env)

    assert "reload" not in game.calls
    assert game.calls[-1] == "pause"
    assert not UNMAPPED_DEBUG_FLAGS & set(game.writes)
    pressed = [name for name in game_input.calls if name not in ("reset", "update_input")]
    assert pressed == ["interact"]  # The fog gate is the only key press of a reset
    env._game_window.focus.assert_called_once_with()
    assert not env.terminated
    assert obs["boss_animation"] == game.data.boss_animations["margit"]["all"]["Idle"]["ID"]
    assert obs["player_animation"] == game.data.player_animations[PLAYER_IDLE_ANIMATION]["ID"]
    assert info["action_mask"].shape == (env.action_space.n,)


def test_reset_sets_up_the_arena_in_phase_1_and_a_reload_requires_a_new_setup(env):
    assert env._arena_setup_required()

    obs, _ = _reset(env)

    assert env._arena_init
    assert obs["phase"] == 1
    assert env.step(NO_OP_ACTION)[0]["phase"] == 1
    env._reload()
    assert env.game.calls[-1] == "reload"
    assert env._arena_setup_required()


def test_reload_is_required_if_margit_is_dead_or_after_the_hard_reset_interval(env):
    clock, interval = env.game.clock, MargitEnv.HARD_RESET_INTERVAL
    assert not env._reload_required()
    clock.now = interval - 1
    assert not env._reload_required()
    clock.now = interval + 1
    assert env._reload_required()
    clock.now = 0
    env.game.margit_hp = 0
    assert env._reload_required()


@pytest.mark.parametrize(
    "method, state, error, message",
    [
        ("_reload_required", {"is_ingame": False}, GameStateError, "ingame"),
        ("_enter_fog_gate", {"player_animation": 60060}, ResetError, "idle.*60060"),
        ("_enter_fog_gate", {}, ResetError, "not entered"),
        ("_enter_fog_gate", {"player_pose": POST_FOG_WALL, "player_hp": 0}, ResetError, "died"),
        ("_camera_reset", {}, LockOnFailure, "Lock on"),
    ],
    ids=["not ingame", "no idle", "outside", "died", "no lock on"],
)
def test_reset_steps_fail_with_a_descriptive_error(env, method, state, error, message):
    vars(env.game).update(state)

    with pytest.raises(error, match=message):
        getattr(env, method)()

    assert not env._arena_init


def test_arena_setup_fails_if_the_player_cannot_reach_the_fog_wall(env, monkeypatch):
    # The teleports have no effect
    pose = property(lambda self: COORDS["bonfire"], lambda self, pose: None)
    monkeypatch.setattr(FakeGame, "player_pose", pose, raising=False)

    with pytest.raises(RetriesExceededError):
        env._arena_setup()


def test_fog_gate_is_entered_after_the_fall_animation_of_the_teleport(env, monkeypatch):
    # Teleports can trigger a fall animation that ignores the interaction with the fog gate
    fall = property(lambda game: 29032000 if game.clock.now < 0.5 else PLAYER_IDLE_ANIMATION)
    monkeypatch.setattr(FakeGame, "player_animation", fall, raising=False)
    game, animations = env.game, []
    env._game_input.on_action["interact"] = lambda: (
        animations.append(game.player_animation),
        vars(game).update(player_pose=POST_FOG_WALL.copy()),
    )

    env._enter_fog_gate()

    assert animations == [PLAYER_IDLE_ANIMATION]


def test_fog_gate_is_entered_if_the_player_passes_the_fog_gate_late(env, monkeypatch):
    # The player turns towards the fog gate before the fog gate animation starts
    game, t_passed = env.game, []
    env._game_input.on_action["interact"] = lambda: t_passed.append(game.clock.now + 0.5)
    pose = property(
        lambda game: (
            POST_FOG_WALL if t_passed and game.clock.now >= t_passed[0] else COORDS["fog_wall"]
        ),
        lambda game, pose: None,
    )
    monkeypatch.setattr(FakeGame, "player_pose", pose, raising=False)

    env._enter_fog_gate()

    assert game.clock.now >= t_passed[0]


def test_entity_reset_does_not_move_the_player_past_the_target_while_the_position_lags(
    env, monkeypatch
):
    # Elden Ring updates the player position one tick after a pose write, and the pose setter moves
    # the player relative to that position. Writes before the update overshoot the target
    position = {"actual": COORDS["bonfire"].copy(), "reported": COORDS["bonfire"].copy()}
    written = []

    def write_pose(game: FakeGame, pose: np.ndarray):
        position["actual"] = position["actual"] + np.asarray(pose) - position["reported"]
        written.append(position["actual"][:3])

    def sleep(game: FakeGame, t: float):
        game.clock.now += t
        position["reported"] = position["actual"].copy()

    lagging_pose = property(lambda game: position["reported"].copy(), write_pose)
    monkeypatch.setattr(FakeGame, "player_pose", lagging_pose, raising=False)
    monkeypatch.setattr(FakeGame, "sleep", sleep)

    env._entity_reset()

    np.testing.assert_allclose(written, [COORDS["player_init_pose"][:3]] * len(written))


def test_entity_reset_failure_requests_an_arena_setup(env):
    env._arena_init = True
    env.game.player_animation = 60060

    with pytest.raises(ResetError, match="could not be reset"):
        env._entity_reset()

    assert env._arena_setup_required()


def test_step_returns_the_gymnasium_step_tuple_and_pauses_the_game_at_the_configured_speed(fakes):
    env = MargitEnv(game_speed=2.0)
    obs, _ = _reset(env)
    assert env.observation_space.contains(obs)
    env.game.calls.clear()

    obs, reward, terminated, truncated, info = env.step(NO_OP_ACTION)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)
    assert info["action_mask"].dtype == bool
    assert info["action_mask"].shape == (env.action_space.n,)
    assert env.game.game_speed == 2.0
    assert env.game.calls == ["pause"]


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_step_executes_the_keys_of_the_action(env, action):
    _reset(env)
    obs, *_ = env.step(action)
    assert env._game_input.updates[-1] == ACTIONS[action]
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("action", [-1, len(ACTIONS), 1000])
def test_step_rejects_out_of_range_actions_without_pressing_keys(env, action):
    _reset(env)

    obs, reward, terminated, truncated, _ = env.step(action)

    assert env._game_input.updates[-1] == []
    assert env.observation_space.contains(obs)
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_step_rejects_actions_that_need_stamina_without_stamina(env):
    _reset(env)
    env.game.player_sp = 0

    _, _, _, _, info = env.step(NO_OP_ACTION)
    # Only the movement (0 - 7) and the no-op (19) remain valid
    assert info["action_mask"].tolist() == [True] * 8 + [False] * 11 + [True]

    env.step(8)  # Roll forward
    assert env._game_input.updates[-1] == []

    env.step(0)  # Walk forward
    assert env._game_input.updates[-1] == ACTIONS[0]


def test_random_actions_produce_valid_observations(env):
    env.action_space.seed(0)
    _reset(env)

    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert env.observation_space.contains(obs)
        assert not (terminated or truncated)


def _state(player_hp: int = PLAYER_MAX_HP, boss_hp: int = BOSS_MAX_HP) -> MargitState:
    return MargitState(
        player_hp=player_hp, player_max_hp=PLAYER_MAX_HP, boss_hp=boss_hp, boss_max_hp=BOSS_MAX_HP
    )


# The damage of the player and the boss in a step and the resulting reward
REWARDS = {
    "nothing happens": (0, 0, 0.0),
    "boss damage": (0, 417, 417 / BOSS_MAX_HP),
    "player damage": (91, 0, -91 / PLAYER_MAX_HP),
    "both take damage": (91, 417, 417 / BOSS_MAX_HP - 91 / PLAYER_MAX_HP),
    "player dies": (PLAYER_MAX_HP, 0, -1.0 - 0.1),
    "boss dies": (0, BOSS_MAX_HP, 1.0 + 0.1),
    "both die, the boss death counts": (PLAYER_MAX_HP, BOSS_MAX_HP, 1.0 - 1.0 + 0.1),
}


@pytest.mark.parametrize("player_damage, boss_damage, reward", REWARDS.values(), ids=list(REWARDS))
def test_reward_is_the_normalized_damage_plus_the_terminal_bonus(
    player_damage, boss_damage, reward
):
    next_state = _state(PLAYER_MAX_HP - player_damage, BOSS_MAX_HP - boss_damage)
    assert MargitEnv.compute_reward(_state(), next_state) == pytest.approx(reward)


@pytest.mark.parametrize("boss_max_hp", [1000, 2000, 10000])
def test_boss_reward_is_the_fraction_of_the_boss_max_hp(boss_max_hp):
    state = replace(_state(), boss_hp=boss_max_hp, boss_max_hp=boss_max_hp)
    next_state = replace(state, boss_hp=boss_max_hp - boss_max_hp // 10)
    assert MargitEnv.compute_reward(state, next_state) == pytest.approx(0.1)


def test_reward_is_not_shaped_by_poses_animations_or_mana():
    state = _state()
    next_state = replace(
        state,
        player_pose=np.array([100.0, 100.0, 0.0, 0.0]),
        boss_animation="Attack3000",
        player_animation=20100,
        player_mp=0,
    )
    assert MargitEnv.compute_reward(state, next_state) == 0.0


# The steps of an episode: the damage of the player and Margit, the reward and the termination
EPISODES = {
    "the damage accumulates": [(91, 417, 417 / BOSS_MAX_HP - 91 / PLAYER_MAX_HP, False)] * 2,
    "the player HP is depleted": [
        (300, 0, -300 / PLAYER_MAX_HP, False),
        (200, 0, -(PLAYER_MAX_HP - 300) / PLAYER_MAX_HP - 0.1, True),
    ],
    "margit dies": [(0, BOSS_MAX_HP, 1.0 + 0.1, True)],
}


@pytest.mark.parametrize("steps", EPISODES.values(), ids=list(EPISODES))
def test_steps_reward_the_damage_and_end_the_episode_when_an_hp_bar_is_empty(env, steps):
    game = env.game
    _reset(env)
    player_hp, boss_hp = PLAYER_MAX_HP, BOSS_MAX_HP

    for player_damage, boss_damage, reward, terminated in steps:
        game.deal_damage(player=player_damage, boss=boss_damage)
        game.player_mp = 10
        obs, step_reward, step_terminated, truncated, _ = env.step(NO_OP_ACTION)

        player_hp, boss_hp = max(player_hp - player_damage, 0), max(boss_hp - boss_damage, 0)
        assert step_reward == pytest.approx(reward)
        assert (step_terminated, truncated) == (terminated, False)
        assert (obs["player_hp"].tolist(), obs["boss_hp"].tolist()) == ([player_hp], [boss_hp])
        assert env.observation_space.contains(obs)
        # The game is topped up after every step, the environment keeps track of the damage
        assert (game.player_hp, game.margit_hp) == (PLAYER_MAX_HP, BOSS_MAX_HP)
        assert game.player_mp == game.player_max_mp


def test_step_after_the_end_of_the_episode_needs_a_reset(env):
    _reset(env)
    env.game.deal_damage(boss=BOSS_MAX_HP)
    env.step(NO_OP_ACTION)

    with pytest.raises(ResetNeeded):
        env.step(NO_OP_ACTION)
    obs, _ = env.reset()

    assert not env.terminated
    assert obs["player_hp"].tolist() == [PLAYER_MAX_HP]
    assert obs["boss_hp"].tolist() == [BOSS_MAX_HP]
    assert not env.step(NO_OP_ACTION)[2]


def test_a_death_inside_the_game_ends_the_episode_and_the_next_reset_respawns_the_player(env):
    # The debug flags that prevent deaths are not mapped, e.g. a fall from the arena kills the player
    game = env.game
    _reset(env)
    game.deal_damage(player=PLAYER_MAX_HP)

    obs, reward, terminated, truncated, _ = env.step(NO_OP_ACTION)

    assert (terminated, truncated) == (True, False)
    assert obs["player_hp"].tolist() == [0]
    assert reward == pytest.approx(-1.0 - 0.1)
    assert env._reload_required()

    def respawn():  # The reload respawns the player at the grace
        game.calls.append("reload")
        vars(game).update(player_hp=PLAYER_MAX_HP, player_pose=COORDS["bonfire"].copy())

    vars(game)["reload"] = respawn
    obs, _ = _reset(env)

    assert game.calls.count("reload") == 1
    assert not env.terminated
    assert not env._reload_required()
    assert obs["player_hp"].tolist() == [PLAYER_MAX_HP]


def test_close_kills_the_player_without_touching_debug_flags_or_waiting_for_the_reload(env):
    game = env.game
    _reset(env)
    game.writes.clear()
    game.calls.clear()
    env._game_input.calls.clear()
    t_close = game.clock.now

    env.close()

    assert game.writes == ["player_hp"]
    assert game.player_hp == 0
    assert game.calls == ["resume"]
    assert env._game_input.calls == ["reset"]
    assert game.clock.now == t_close


# Animation IDs of Margit that have been inventoried during live phase 1 fights
# fmt: off
PHASE1_ANIMATIONS = {
    2000020: "Idle", 2002000: "Move2000", 2002002: "Move2002", 2002003: "Move2003",
    2003000: "Attack3000", 2003001: "Attack3001", 2003003: "Attack3003", 2003006: "Attack3006",
    2003009: "Attack3009", 2003011: "Attack3011", 2003012: "Attack3012", 2003014: "Attack3014",
    2003015: "Attack3015", 2003017: "Attack3017", 2003021: "Attack3021", 2003024: "Attack3024",
    2003025: "Attack3025", 2003026: "Attack3026", 2003029: "Attack3029", 2005003: "React5003",
    2006002: "Special6002", 2006003: "Special6003", 2008700: "Ranged8700", 2009000: "Special9000",
    2009500: "Special9500", 2011060: "Taunt11060", 41000: "Hit41000", 41002: "Hit41002",
}
# fmt: on
# 0 and -1 are no animations of Margit, 20100 is an animation of the player
UNKNOWN_ANIMATIONS = {i: f"Unknown_{i}" for i in (0, -1, 20100, 2003099, -2099255776)}


@pytest.mark.parametrize(
    "animation_id, name", [*PHASE1_ANIMATIONS.items(), *UNKNOWN_ANIMATIONS.items()]
)
def test_boss_animation_ids_are_translated_to_names_and_observed_as_indices(
    env, animation_id, name
):
    game = env.game
    game.mem.read_record.return_value = animation_id

    env._game_state = env.game_state()
    index = game.data.boss_animations["margit"]["all"].get(name, {"ID": -1})["ID"]

    assert env._game_state.boss_animation == name
    game.mem.read_record.assert_called_once_with(game.data.addresses["MargitAnimation"])
    assert env.obs["boss_animation"] == index
    assert env.observation_space["boss_animation"].contains(index)


def test_step_with_unknown_animations_warns_and_continues(env, caplog):
    _reset(env)
    env.game.mem.read_record.return_value = 2003099  # Not an animation of Margit
    env.game.player_animation = 123456789  # Not an animation of the player

    with caplog.at_level(logging.WARNING, logger="soulsgym.envs.soulsenv"):
        obs, _, terminated, truncated, _ = env.step(NO_OP_ACTION)

    assert "Unknown_2003099" in caplog.text
    assert "123456789" in caplog.text
    assert (obs["boss_animation"], obs["player_animation"]) == (-1, -1)
    assert env.observation_space.contains(obs)
    assert not (terminated or truncated)


def test_margit_animations_are_listed_once_and_translated_by_the_game_interface():
    margit = StaticGameData("EldenRing").boss_animations["margit"]
    sections = [margit[section] for section in ("attacks", "movement", "misc")]
    names = [name for section in sections for name in section]
    ids = [animation_id for section in sections for animation_id in section.values()]

    assert len(names) == len(set(names)) == len(margit["all"])
    assert len(ids) == len(set(ids))
    assert all(isinstance(animation_id, int) for animation_id in ids)
    for animation_id, name in zip(ids, names):
        assert simulated_game(MargitAnimation=animation_id).margit_animation == name

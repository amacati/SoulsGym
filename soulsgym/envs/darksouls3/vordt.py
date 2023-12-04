"""The SoulsGym environment for Vordt of the Boreal Valley.

The player and Vordt always start from their respective start poses at full HP/SP. The player starts
with the stats and weapons as configured in <TODO: add config>. We do not allow shield blocking or
two handing at this point, although this can easily be supported. Parrying is enabled.

Note:
    Phase 2 of the boss fight is available by setting the environment keyword argument ``phase``.
    See :mod:`~.envs` for details.
"""
from __future__ import annotations

import logging

import numpy as np
from gymnasium import spaces

from soulsgym.envs.soulsenv import SoulsEnv
from soulsgym.games import DarkSoulsIII

logger = logging.getLogger(__name__)


class VordtEnv(SoulsEnv):
    """The SoulsGym environment for Vordt of the Boreal Valley."""

    ENV_ID = "vordt"

    def __init__(self, game_speed: float = 1., phase: int = 1):
        """Initialize the observation and action spaces.

        Args:
            game_speed: The speed of the game during :meth:`.SoulsEnv.step`. Defaults to 1.0.
            phase: The phase of the boss fight. Either 1 or 2 for Vordt. Defaults to 1.
        """
        super().__init__(game_speed=game_speed)
        self.game: DarkSoulsIII  # Type hint only
        self.phase = phase
        pose_box_low = np.array(self.env_args.coordinate_box_low, dtype=np.float32)
        pose_box_high = np.array(self.env_args.coordinate_box_high, dtype=np.float32)
        cam_box_low = np.array(self.env_args.coordinate_box_low[:3] + [-1, -1, -1],
                               dtype=np.float32)
        cam_box_high = np.array(self.env_args.coordinate_box_high[:3] + [1, 1, 1], dtype=np.float32)
        player_animations = self.game.data.player_animations
        boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(2, start=1),
            "player_hp": spaces.Box(0, self.env_args.player_max_hp),
            "player_max_hp": spaces.Discrete(1, start=self.env_args.player_max_hp),
            "player_sp": spaces.Box(0, self.env_args.player_max_sp),
            "player_max_sp": spaces.Discrete(1, start=self.env_args.player_max_sp),
            "boss_hp": spaces.Box(0, self.env_args.boss_max_hp),
            "boss_max_hp": spaces.Discrete(1, start=self.env_args.boss_max_hp),
            "player_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
            "boss_pose": spaces.Box(pose_box_low, pose_box_high, dtype=np.float32),
            "camera_pose": spaces.Box(cam_box_low, cam_box_high, dtype=np.float32),
            "player_animation": spaces.Discrete(len(player_animations) + 1, start=-1),
            "player_animation_duration": spaces.Box(0., 10.),
            "boss_animation": spaces.Discrete(len(boss_animations) + 1, start=-1),
            "boss_animation_duration": spaces.Box(0., 10.),
            "lock_on": spaces.Discrete(2)
        })
        self.action_space = spaces.Discrete(len(self.game.data.actions))
        assert phase in (1, 2)
        self.phase = phase
        self._phase_init = False
        # We keep track of the last time the environment has been reset completely. After several
        # hours of gameplay, the game input starts to lag and the agent's actions are not executed
        # properly anymore. We therefore reset the environment every 15 minutes to avoid an
        # unintended performance degradation
        self._last_hard_reset = 0

    @property
    def game_id(self) -> str:
        """Return the ID of the souls game that is required to run for this environment.

        Returns:
            The game ID.
        """
        return "DarkSoulsIII"

    @property
    def obs(self) -> dict:
        """Observation property of the environment.

        Returns:
            The current observation of the environment.
        """
        obs = self._internal_state.as_dict()
        obs["player_hp"] = np.array([obs["player_hp"]], dtype=np.float32)
        obs["player_sp"] = np.array([obs["player_sp"]], dtype=np.float32)
        obs["boss_hp"] = np.array([obs["boss_hp"]], dtype=np.float32)
        # Default animation ID for unknown animations is -1
        _player_animations = self.game.data.player_animations
        obs["player_animation"] = _player_animations.get(obs["player_animation"], {"ID": -1})["ID"]
        _boss_animations = self.game.data.boss_animations[self.ENV_ID]["all"]
        obs["boss_animation"] = _boss_animations.get(obs["boss_animation"], {"ID": -1})["ID"]
        obs["player_animation_duration"] = np.array([obs["player_animation_duration"]],
                                                    dtype=np.float32)
        obs["boss_animation_duration"] = np.array([obs["boss_animation_duration"]],
                                                  dtype=np.float32)
        obs["player_pose"] = obs["player_pose"].astype(np.float32)
        obs["boss_pose"] = obs["boss_pose"].astype(np.float32)
        obs["camera_pose"] = obs["camera_pose"].astype(np.float32)
        return obs

    @property
    def info(self) -> dict:
        """Info property of the environment.

        Returns:
            The current info dict of the environment.
        """
        return {"allowed_actions": self.current_valid_actions()}

from gym.envs.registration import register

register(
    id="envs/game2048-v0",
    entry_point="game2048.envs:game2048Env",
    max_episode_steps=40000,
)
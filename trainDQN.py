import sys

import gym
import game2048
import numpy as np
import os.path
from stable_baselines3 import A2C, PPO, DQN
import matplotlib.pyplot as plt

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import VecFrameStack




if __name__ == "__main__":

    n_evs = int(sys.argv[1])
    TIMESTEPS = int(sys.argv[2])
    print(n_evs)
    print(TIMESTEPS)

    dqn_models_dir = "models/test-2-DQN/" + str(n_evs)
    logdir = "logs"

    if not os.path.exists(dqn_models_dir):
        os.makedirs(dqn_models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    def train_DQN(n_evs, TIMESTEPS):
        env = make_vec_env("envs/game2048-v0", n_envs=n_evs)

        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

        for i in range(1, 1000):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN-test-2-"+ str(n_evs))
            model.save(f"{dqn_models_dir}/{TIMESTEPS*i}")


    train_DQN(n_evs, TIMESTEPS)
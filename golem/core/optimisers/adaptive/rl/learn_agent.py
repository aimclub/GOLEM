import gym
import torch
from sb3_contrib import TRPO, ARS  # Example heavy algorithms
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure


def learn_agent(env):
    device = torch.device('cpu')

    log_dir = "./tensorboard_logs/"
    new_logger = configure(log_dir, ["tensorboard"])

    # Choose the algorithm (TRPO or ARS as examples)
    model = TRPO('MultiInputPolicy', env, verbose=1, device=device)
    # model = ARS('MlpPolicy', env, verbose=1)  # For ARS

    model.set_logger(new_logger)

    # Optionally, Add a checkpoint callback to save the model at certain intervals
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='model_checkpoint')

    # Train the model
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Save the final model
    model.save('final_model')

    # To visualize learning progress, run:
    # tensorboard --logdir ./tensorboard_logs/

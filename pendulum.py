#!/usr/bin/env python3
import os
import gym
import numpy as np
from stable_baselines import bench, logger, A2C, DQN

from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.ppo2 import PPO2
import sys
import gym
import tensorflow as tf
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import multiprocessing


def train():
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    env = DummyVecEnv([lambda: gym.make('Pendulum-v0')])

    env = VecNormalize(env)
    seed = 10
    set_global_seeds(seed)
    policy = MlpPolicy
    model = PPO2(policy=policy, 
                 env=env,
                 n_steps=2048,
                 nminibatches=32,
                 noptepochs=10,
                 #n_steps=32, 
                 #nminibatches=4,
                 lam=0.95, 
                 gamma=0.995, 
                 #noptepochs=4,
                 tensorboard_log="tb_log",
                 #log_interval=1,
                 #tb_log_interval=20,
                 ent_coef=0.0,
                 learning_rate=3e-4,
                 cliprange=0.2)
    for i in range(1):
        mult = int(1e5)
        model.learn(total_timesteps=mult)
        model.save(f"models/PPO2_model_{i*mult:15d}.bin")

    return model, env


def play_random():
    import gym
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    running_reward = 0.0
    alpha = 0.01
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        running_reward = running_reward * (1-alpha) + alpha * reward
        print(obs, reward, done, info, running_reward)
        if done:
            print("Finished after {} timesteps".format(t+1))
            break


def play(model="", train=True):
    import gym
    seed = 10
    set_global_seeds(seed)

    env = gym.make('LunarLander-v2')
    env = DummyVecEnv([lambda: env])

    policy = MlpPolicy
    model = PPO2(policy="MlpPolicy",
                env=env,
                tensorboard_log="tb_log",
                ent_coef=0.0,
                n_steps=500,
                nminibatches=10,
                noptepochs=10,
                learning_rate=0.0003,
                cliprange=0.2,
                gamma=0.998,
                verbose=1)

    def test(model):
        for trial in range(10):
            obs = env.reset()
            running_reward = 0.0
            alpha = 0.01

            for _ in range(500):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                env.render()
                #running_reward = running_reward * (1-alpha) + alpha * reward
                running_reward += reward
                print(obs, reward, done, info, running_reward)
                if done:
                    print("Finished after {} timesteps".format(_+1))
                    break

    if train:
        for i in range(1):
            model.learn(total_timesteps=1000000, log_interval=10)
            model.save(f'models/lunar_{i}.bin')

    model = PPO2.load('models/lunar_0.bin')
    test(model)


def main():
    logger.configure()
    #train()
    play(train=True)

if __name__ == '__main__':
    main()

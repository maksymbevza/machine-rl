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

from environment import MachineProductionEnv


def make_env():
    from lib import read_detail_tree, read_machine_detail
    machine_detail = read_machine_detail("data/machine_detail.txt")
    detail_tree = read_detail_tree("data/detail_tree.txt",
                                   machine_detail.shape[0])
    env = MachineProductionEnv(machine_detail, detail_tree)
    return env


def play(train=True):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    env = DummyVecEnv([make_env])

    env = VecNormalize(env)
    seed = 10
    set_global_seeds(seed)
    model = PPO2(policy="MlpPolicy",
                env=env,
                tensorboard_log="tb_log",
                ent_coef=0.1,
                n_steps=2048,
                nminibatches=32,
                noptepochs=10,
                learning_rate=0.0003,
                cliprange=0.2,
                gamma=0.998,
                verbose=1,
                policy_kwargs={"net_arch": [64, 64, 32, 16, 16, 16]})

    def test(model):
        for trial in range(10):
            obs = env.reset()
            running_reward = 0.0
            alpha = 0.01

            for _ in range(5000):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                env.render()
                #running_reward = running_reward * (1-alpha) + alpha * reward
                running_reward += reward
                #print(obs, reward, done, info, running_reward)
                if done:
                    print("Finished after {} timesteps".format(_+1))
                    break

    if train:
        for i in range(1):
            model.learn(total_timesteps=100_000, log_interval=10)
            model.save(f'models/machine_{i}.bin')

    model = PPO2.load('models/machine_0.bin')
    test(model)


def main():
    logger.configure()
    #train()
    play(train=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import os
import gym
import numpy as np
from stable_baselines import bench, logger, A2C, DQN
from gym.wrappers import TimeLimit

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


GAMMA = 0.97

def make_env():
    from lib import read_detail_tree, read_machine_detail
    machine_detail = read_machine_detail("data/machine_detail.txt")
    detail_tree = read_detail_tree("data/detail_tree.txt",
                                   machine_detail.shape[0])
    env = MachineProductionEnv(machine_detail, detail_tree)
    #env = gym.make('LunarLander-v2')
    env = TimeLimit(env, 1500)
    return env


def play(train=True):
    ncpu = 4
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    n_env = 1
    env = DummyVecEnv([make_env]*n_env)

    #env = VecNormalize(env, gamma=GAMMA)
    seed = 10
    set_global_seeds(seed)
    model = DQN(policy="LnMlpPolicy",
                env=env,
                tensorboard_log="tb_log_new",
                #n_steps=32,
                #nminibatches=4,
                #noptepochs=10,
                learning_rate=0.0003,
                exploration_fraction=0.3,
                #cliprange=0.2,
                #max_grad_norm=0.2,
                gamma=GAMMA,
                verbose=1,
                policy_kwargs={
                    #"net_arch": [128, 64, 32, 32, 32],
                    #"n_lstm": 32
                })

    def test(model):
        env = DummyVecEnv([make_env] * n_env)
        #env = VecNormalize.load("models/machine_snap_env.bin", venv=env)
        #env.training = False
        for trial in range(1):
            obs = env.reset()
            running_reward = 0.0
            alpha = 0.01

            for _ in range(5000):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                reward = reward[0]
                done = done[0]
                info = info[0]
                #running_reward = running_reward * (1-alpha) + alpha * reward
                running_reward += reward
                #print(obs, reward, done, info, running_reward)
                if done:
                    print("Finished after {} timesteps".format(_+1))
                    break
                else:
                    env.envs[0].render()

    def callback(locals_, globals_):
        import ipdb; ipdb.set_trace()
        return True


    if train:
        try:
            model.learn(total_timesteps=3_000_000, log_interval=50)
        except KeyboardInterrupt:
            model.save("models/machine_snap_model.bin")
            env.save("models/machine_snap_env.bin")
            raise
        model.save(f'models/machine_0_model.bin')
        env.save(f'models/machine_0_env.bin')

    model = DQN.load('models/machine_snap_model.bin')
    test(model)


def main():
    logger.configure()
    #train()
    play(train=True)


if __name__ == '__main__':
    main()

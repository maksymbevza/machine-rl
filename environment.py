import gym
import numpy as np
from gym import spaces


DENOMINATOR = 100.0


class MachineProductionEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 machine_detail_time,
                 detail_tree):

        super(MachineProductionEnv, self).__init__()
        self.machine_detail_time = machine_detail_time
        self.detail_tree = detail_tree

        self.d_count = machine_detail_time.shape[0]
        self.m_count = machine_detail_time.shape[1]
        self.action_space = spaces.MultiDiscrete([self.d_count + 1]*self.m_count)

        self.observation_space = spaces.Box(
            low=0,
            high=1,  #TODO: Fix high to dynamic
            shape=(self.d_count*2 + self.m_count,),
            dtype=int)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.produced = np.zeros((self.d_count), dtype=np.int)
        self.needed = np.zeros((self.d_count), dtype=np.int)
        self.scheduled_for = np.zeros((self.m_count), dtype=np.int)
        self.detail_line = []
        self.machine_busy = np.zeros((self.m_count,), dtype=np.int)

        self.needed[0] = 1  # TODO: Read this from file
        self.cumulative_reward = 0

        return self._next_observation()

    def _next_observation(self):
        obs = np.concatenate([self.produced,
                              self.needed,
                              self.machine_busy])

        return obs.astype(np.float32) / DENOMINATOR

    def step(self, action):
        reward = -5  # Penalty for time
        reward += self._take_action(action)
        self._produce()

        self.current_step += 1

        done = (self.needed <= self.produced).all()
        if done:
            reward += 1000
        obs = self._next_observation()
        self.cumulative_reward += reward
        return obs, reward, done, {"cumulative_reward": self.cumulative_reward}

    def _take_action(self, action):
        reward = 0
        for machine_id, detail_id in enumerate(action):
            if detail_id == 0:
                continue
            detail_id -= 1
            if self.machine_busy[machine_id]:
                reward -= 1
                continue
            if (self.detail_tree[detail_id] > self.produced).any():
                reward -= 1
                continue

            self.produced -= self.detail_tree[detail_id]

#            reward += 1
            ttc = self.machine_detail_time[detail_id, machine_id]
            self.machine_busy[machine_id] = 1
            self.scheduled_for[machine_id] = ttc
            self.detail_line.append([ttc,
                                     detail_id,
                                     machine_id])
        return reward

    def _produce(self):
        self.scheduled_for -= self.machine_busy

        new_detail_line = []
        for ttc, detail_id, machine_id in self.detail_line:
            if ttc == 1:
                self.machine_busy[machine_id] = 0
                self.produced[detail_id] += 1
            else:
                new_detail_line.append([ttc - 1, detail_id, machine_id])

        self.detail_line = new_detail_line

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Produced: {self.produced}')
        print(f'Needed: {self.needed}')
        print(f'Scheduled For: {self.scheduled_for}')
        print(f'Machine Busy: {self.machine_busy}')
        print(f'Detail Line: {self.detail_line}')
        print(f'Cumulative Reward: {self.cumulative_reward}')

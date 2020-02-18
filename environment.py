import gym
import numpy as np
from gym import spaces
import copy


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
        print('reset')
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.produced = np.zeros((self.d_count), dtype=np.int)
        self.needed = np.zeros((self.d_count), dtype=np.int)
        self.scheduled_for = np.zeros((self.m_count), dtype=np.int)
        self.detail_line = []
        self.machine_busy = np.zeros((self.m_count,), dtype=np.int)

        self.needed[0] = 1  # TODO: Read this from file
        self.proximity_points = self._calculate_points()
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

        new_points = self._calculate_points()
        if new_points != self.proximity_points:
            delta = new_points - self.proximity_points
            #print('adding: ', self.current_step, delta)
            reward += delta
        self.proximity_points = new_points

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
        print(f'Proximity Points: {self.proximity_points}')

    def _calculate_points(self):
        STARTING_POINTS = 100
        DISCOUNT = 0.3

        needed = copy.deepcopy(self.needed)
        produced = copy.deepcopy(self.produced)
        for i in range(self.d_count):
            needed += self.detail_tree[i] * needed[i]
            produced += self.detail_tree[i] * produced[i]
        overmade = (produced - needed).clip(min=0)

        level = np.ones_like(self.needed)
        level *= -1
        level[0] = 0
        level[1] = 1
        level[2] = 2
        level[3] = 3
        level[4] = 3
        for i in range(1, self.d_count):
            pass

        scores = 0
        made_ok = np.min([needed, produced], axis=0)
        for i in range(self.d_count):
            one_point = int(STARTING_POINTS * DISCOUNT**level[i])
            scores += made_ok[i] * one_point
            scores -= overmade[i] * one_point
            #if i == 0 and produced[i] == 1:
                #import ipdb; ipdb.set_trace()
                #print("")


        return scores

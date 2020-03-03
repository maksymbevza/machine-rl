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
        self.action_space = spaces.Discrete(1 + self.d_count * self.m_count)

        self.observation_space = spaces.Box(
            low=0,
            high=1,  #TODO: Fix high to dynamic
            shape=(self.d_count*2 + self.m_count*self.d_count,),
            dtype=float)

    def _make_estimates(self):
        e_needed = self._extend_details(self.needed)
        e_produced = self._extend_details(self.produced)
        e_diff = (e_needed - e_produced).clip(min=0)

        detail_time_min = self.machine_detail_time.min(1)
        self.estimated_time = (detail_time_min * e_diff).sum()
        # TODO: Add running reward shift

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.produced = np.zeros((self.d_count), dtype=np.int)
        self.needed = np.zeros((self.d_count), dtype=np.int)
        self.scheduled_for = np.zeros((self.m_count), dtype=np.int)
        self.detail_line = []
        self.machine_busy = np.zeros((self.m_count, self.d_count), dtype=np.int)

        self.estimated_time = 0
        for i in range(1):
            self.needed[0] = np.random.randint(1, 4)  # TODO: Read this from file
            self.needed[1] = np.random.randint(1, 4)  # TODO: Read this from file
            self.needed[2] = np.random.randint(1, 4)  # TODO: Read this from file
            for i in range(3):
                self.produced[i] = np.random.randint(self.needed[i])  # TODO: Read this from file
            self._make_estimates()
        print('reset', self.needed, self.produced, self.estimated_time)
        self.proximity_points = self._calculate_points()
        self.cumulative_reward = 0

        return self._next_observation()

    def _next_observation(self):
        obs = np.concatenate([self.produced,
                              self.needed,
                              self.machine_busy.reshape(-1)])

        return obs.astype(np.float32) / DENOMINATOR

    def step(self, action):
        reward = -5 / (self.estimated_time / 100)  # Penalty for time
        reward += self._take_action(action)
        old_produced = list(self.produced)
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
            print('done',
                  self.current_step,
                  self.cumulative_reward + reward,
                  self.estimated_time)
        self.cumulative_reward += reward
        obs = self._next_observation()
        return obs, reward, done, {"cumulative_reward": self.cumulative_reward}

    def _take_action(self, action):
        reward = 0
        if action != 0:
            action -= 1
            machine_id, detail_id = action // self.d_count, action % self.d_count
            if self.machine_busy.sum(1)[machine_id]:
                reward -= 1
            elif (self.detail_tree[detail_id] > self.produced).any():
                reward -= 1
            else:
                self.produced -= self.detail_tree[detail_id]

    #            reward += 1
                ttc = self.machine_detail_time[detail_id, machine_id]
                self.machine_busy[machine_id, detail_id] = 1
                self.scheduled_for[machine_id] = ttc
                self.detail_line.append([ttc,
                                         detail_id,
                                         machine_id])
        return reward

    def _produce(self):
        self.scheduled_for -= self.machine_busy.sum(1)

        new_detail_line = []
        for ttc, detail_id, machine_id in self.detail_line:
            if ttc == 1:
                self.machine_busy[machine_id, :] = 0
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
        level_price = [10, 8, 6, 4, 2]

        needed = copy.deepcopy(self.needed)
        produced = copy.deepcopy(self.produced)
        for _, d, m in self.detail_line:
            produced[d] += 1
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
            one_point = level_price[level[i]]
            scores += made_ok[i] * one_point
            scores -= int((overmade[i])**1.0 * one_point)


        return scores

    def _extend_details(self, details):
        details = copy.deepcopy(details)
        for i in range(self.d_count):
            details += self.detail_tree[i] * details[i]
        return details

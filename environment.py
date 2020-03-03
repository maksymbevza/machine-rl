import gym
import numpy as np
from gym import spaces
import copy


DENOMINATOR = 100.0


class MachineProductionEnv(gym.Env):
    """Machine Production Environment.

    machine_detail_time (np.array): Array of shape (d_count, m_count) where
        `[i,j]` means how long does it take to produce detail `i` on machine
        `j`.
    detail_tree (np.array): Array of shape (d_count, d_count) where `[i,j]` 
        means how many details of type `j` is needed to produce detail `i`.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 machine_detail_time,
                 detail_tree):
        super(MachineProductionEnv, self).__init__()

        # Initialization of the production specifications
        self.machine_detail_time = machine_detail_time
        self.detail_tree = detail_tree

        # d_count - detail types
        self.d_count = machine_detail_time.shape[0]

        # m_count - machine count
        self.m_count = machine_detail_time.shape[1]

        # Actions are the following
        # 0 - NO OPERATION
        # (1 + i*d_count + j) - put detail #i to produce on machine #j.
        #
        # This is not MultiDiscrete because DQN does not work with
        # MultiDiscrete type.
        self.action_space = spaces.Discrete(1 + self.d_count * self.m_count)

        # Observations are described in `_next_observation`
        self.observation_space = spaces.Box(
            low=0,
            high=1,  #TODO: Fix high to dynamic
            shape=(self.d_count*2 + self.m_count*self.d_count,),
            dtype=float)

    def _make_estimates(self):
        """Estimate time needed by a baseline.

        Baseline assigns needed tasks to random machine.
        """
        e_needed = self._extend_details(self.needed)
        e_produced = self._extend_details(self.produced)
        e_diff = (e_needed - e_produced).clip(min=0)

        detail_time_min = self.machine_detail_time.min(1)
        self.estimated_time = (detail_time_min * e_diff).sum()
        # TODO: Add running reward shift

    def reset(self):
        """Reset the state of the environment to an initial state.
        """

        self.current_step = 0

        # Amount of details produced (available for use).
        self.produced = np.zeros((self.d_count), dtype=np.int)

        # Amount of details needed to finish the episode.
        self.needed = np.zeros((self.d_count), dtype=np.int)

        # Time when the machine will be available.
        self.scheduled_for = np.zeros((self.m_count), dtype=np.int)

        # Production line. Contains triples of the form
        # (time-to-complete, detail, machine).
        self.detail_line = []

        # Specifies which machine is busy producing which object.
        self.machine_busy = np.zeros((self.m_count, self.d_count), dtype=np.int)

        # Estimated baseline time.
        self.estimated_time = 0

        # Create needs in details randomly.
        self.needed[0] = np.random.randint(1, 4)  # TODO: Read this from file
        self.needed[1] = np.random.randint(1, 4)  # TODO: Read this from file
        self.needed[2] = np.random.randint(1, 4)  # TODO: Read this from file
        for i in range(3):
            self.produced[i] = np.random.randint(self.needed[i])  # TODO: Read this from file
        self._make_estimates()


        print('reset', self.needed, self.produced, self.estimated_time)

        # Proximity points show how close we are to obtaining solution.
        self.proximity_points = self._calculate_points()

        # Reward of the whole episode.
        self.cumulative_reward = 0

        return self._next_observation()

    def _next_observation(self):
        """Return observation.
        """

        obs = np.concatenate([self.produced,
                              self.needed,
                              self.machine_busy.reshape(-1)])

        # We normalize observations to be closer to (-1,1) space which is good
        # for NN.
        return obs.astype(np.float32) / DENOMINATOR

    def step(self, action):
        """Make a step in environment.
        """

        # Penalize time-wise normalized on the baseline performance.
        reward = -5 / (self.estimated_time / 100)

        # Some actions may lead to penalties.
        reward += self._take_action(action)

        # Produce details and take a step.
        self._produce()

        # Compute new proximity points and issue reward for making more
        # proximity points.
        new_points = self._calculate_points()
        if new_points != self.proximity_points:
            delta = new_points - self.proximity_points
            reward += delta
        self.proximity_points = new_points

        self.current_step += 1

        # If the environment is done issue 1000 points.
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
        """Schedule detail to machine.
        """

        reward = 0
        if action != 0:
            action -= 1
            machine_id, detail_id = action // self.d_count, action % self.d_count

            # Schedulinng machine should not be busy.
            if self.machine_busy.sum(1)[machine_id]:
                reward -= 1
            # There should be enough details to use for production.
            elif (self.detail_tree[detail_id] > self.produced).any():
                reward -= 1
            else:
                # Subtract all needed details.
                self.produced -= self.detail_tree[detail_id]

                # Schedule production.
                ttc = self.machine_detail_time[detail_id, machine_id]
                self.machine_busy[machine_id, detail_id] = 1
                self.scheduled_for[machine_id] = ttc
                self.detail_line.append([ttc,
                                         detail_id,
                                         machine_id])
        return reward

    def _produce(self):
        """Perform production of details.
        """

        # Subtract time of obly busy machines.
        self.scheduled_for -= self.machine_busy.sum(1)

        # Build new production line.
        new_detail_line = []
        for ttc, detail_id, machine_id in self.detail_line:
            if ttc == 1:  # Completed production.
                self.machine_busy[machine_id, :] = 0
                self.produced[detail_id] += 1
            else:  # Just decrease time.
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
        """Calculate proximity points.
        """

        needed = copy.deepcopy(self.needed)
        produced = copy.deepcopy(self.produced)

        # Also include details in production.
        for _, d, m in self.detail_line:
            produced[d] += 1

        # Expand need and produced.
        for i in range(self.d_count):
            needed += self.detail_tree[i] * needed[i]
            produced += self.detail_tree[i] * produced[i]
        overmade = (produced - needed).clip(min=0)

        # Level is 0 for the most complext detail, 1 for the one used to
        # produce level 0, 2 details needed for level 1 and so on.
        level_price = [10, 8, 6, 4, 2]
        level = np.ones_like(self.needed)
        level[0] = 0
        level[1] = 1
        level[2] = 2
        level[3] = 3
        level[4] = 3
        # TODO: Above should be calculated online.

        scores = 0
        made_ok = np.min([needed, produced], axis=0)
        for i in range(self.d_count):
            one_point = level_price[level[i]]
            scores += made_ok[i] * one_point
            scores -= int((overmade[i])**1.0 * one_point)

        return scores

    def _extend_details(self, details):
        """For the list of details return how many of details were produced of
        all levels to get to this state.
        """

        details = copy.deepcopy(details)
        for i in range(self.d_count):
            details += self.detail_tree[i] * details[i]
        return details

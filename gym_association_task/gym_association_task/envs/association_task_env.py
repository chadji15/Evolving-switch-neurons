import random

import gym
import numpy as np
from gym import Space
from gym.utils import seeding


class BinaryList(Space):

    def __init__(self, n, one_hot=False):
        super().__init__()
        self.n = n
        self.actions = BinaryList._gen_bin_space(n, one_hot)
        self.seed()

    def sample(self):
        return self.actions[np.random.choice(range(self.n))]

    def contains(self, x):
        for a in self.actions:
            if tuple(a) == tuple(x):
                return True
        return False

    def _to_list(self):
        return self.actions[:]

    @staticmethod
    def _bit_tuple(x, padding=0):
        return tuple([1 if digit == '1' else 0 for digit in bin(x)[2:].rjust(padding, '0')])

    @staticmethod
    def _gen_bin_space(n, one_hot=False):
        bin_space = []
        if one_hot:
            for i in range(n):
                bin_space.append(BinaryList._bit_tuple(2 ** i, n))
        else:
            for i in range(2 ** n):
                bin_space.append(BinaryList._bit_tuple(i, n))
        return bin_space

#If the agent guessed correctly a reward of 0 is given, else -1

class AssociationTaskEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    step_count = 0

    def __init__(self, input_num, output_num, mode='one-to-one', rand_inter = 0):

        self.n = input_num
        self.m = output_num
        self.reward_range = (-1,0)
        self.rand_inter = rand_inter
        self.obs_gen = self.next_obs()
        if mode == 'one-to-one':
            self.action_space = BinaryList(output_num, one_hot= True)
            self.observation_space = BinaryList(input_num, one_hot= True)
        elif mode == 'one-to-many':
            self.observation_space = BinaryList(input_num,one_hot=True)
            self.action_space = BinaryList(output_num,one_hot=False)
        elif mode == 'many-to-one':
            self.observation_space = BinaryList(input_num,one_hot=False)
            self.action_space = BinaryList(input_num,one_hot=True)
        elif mode == 'many-to-many':
            self.observation_space = BinaryList(input_num, False)
            self.action_space = BinaryList(input_num,False)
        else:
            raise ValueError('Unknown mode used: {}'.format(mode))
        self.associations = {}
        self.reset()

    def step(self, action):

        #print("Observation: {}, Expected: {}, Agent action: {}".format(self.observation, self.associations[self.observation],action))
        action = tuple(action)
        isValid = False
        for output in self.action_space._to_list():
            if output == action:
                isValid = True
        if not isValid:
            raise ValueError('Invalid action taken {}'.format(str(action)))
        self.reward = -1

        if self.associations[self.observation] == action:
            self.reward = 0

        self.step_count += 1
        if self.rand_inter > 0 and self.step_count % self.rand_inter == 0:
            self.randomize_associations()

        #self.observation = list(self.associations.keys())[self.step_count % len(self.associations)]
        self.observation = next(self.obs_gen)
        done = True

        return self.observation, self.reward, done, {}

    def reset(self, rand_iter = -1):
        self.step_count = 0
        self.randomize_associations()
        #self.observation = list(self.associations.keys())[0]
        self.observation = next(self.obs_gen)
        if rand_iter > 0:
            self.rand_inter  = rand_iter
        return self.observation

    def render(self, mode='human'):
        assert mode == 'human', '{} mode not supported'.format(mode)
        if self.step_count > 0:
            print("Reward: {}".format(self.reward))
        print("Observation: {}".format(self.observation))
        print("Actions:")
        print(self.action_space._to_list())

    def randomize_associations(self):
        self.associations = {}
        input = [i for i in self.observation_space._to_list()]
        output = [o for o in self.action_space._to_list()]
        np.random.shuffle(input)
        for pattern in input[:]:
            idx = np.random.choice(range(len(output)))
            self.associations[pattern] = output[idx]
            input.remove(pattern)
            output.pop(idx)
            if len(output) == 0:
                output = [o for o in self.action_space._to_list()]

        return self.associations

    def next_obs(self):
        while(True):
            obs = self.observation_space._to_list()
            random.shuffle(obs)
            while(len(obs) > 0):
                yield obs.pop()

class OneToOne2x2(AssociationTaskEnv):
    def __init__(self):
        super().__init__(2,2)

class OneToOne3x3(AssociationTaskEnv):
    def __init__(self):
        super().__init__(3,3)

class OneToMany3x2(AssociationTaskEnv):
    def __init__(self):
        super().__init__(3,2,'one-to-many')

class ManyToMany2x2Rand(AssociationTaskEnv):
    def __init__(self):
        super().__init__(2,2,'many-to-many',500)

if __name__ == '__main__':
    env = OneToMany3x2()
    while True:
        env.render(mode='human')
        x = input("Select action with binary string e.g. 011 or quit: ")
        if x == 'quit' or x == 'Quit' or x =='q':
            exit(0)
        action = tuple([int(c) for c in x])
        env.step(action)

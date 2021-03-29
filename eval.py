import copy
import math
import random

import gym
import gym_association_task
import t_maze
from functools import partial
###
#All the following evaluation functions take as argument an agent variable. It is assumed that the agent has
#an activate function which takes as input a vector (list) and returns an output which corresponds to the action
#of the agent. The actions are described in each environment
##############

#For a network to be considered to be able to solve the one-to-one 3x3 association task in this case it needs to
#to achieve a score of at least 1976 (2000 - (4*(3*2))  = steps - association_changes*(n*(m-1)).
#Note that scores above this threshold do not mean better performance since the score of 1976 is already considered optimal.
#The network here needs to accept 4 inputs (3 for observation and 1 for reward) and return a vector with 3 binary values.
#For num_episodes = 100 | 1000
#    rand_iter = 25     | 100
#    max fitness = 76   | 940
from utilities import shuffle_lists


def eval_one_to_one_3x3(agent, num_episodes = 2000, rand_iter= 500, descriptor_out=False):
    env = gym.make('OneToOne3x3-v0')
    s = num_episodes
    observation = env.reset(rand_iter=rand_iter)
    input = tuple(list(observation) + [0])
    for i_episode in range(num_episodes):
        action = agent.activate(input)
        observation, reward, done, info = env.step(action)
        input = list(input)
        input[-1] = reward
        agent.activate(input)
        input = tuple(list(observation) + [0])
        s += reward
    env.close()
    return s

#For a network to be considered to be able to solve the one-to-many 3x2 association task in this case it needs to
#to achieve a score of at least 1964 (2000 - 4*(3*(4-1)) = steps - association_changes*(n*(2^m - 1)).
#Note that scores above this threshold do not mean better performance since the score of 1964 is already considered optimal.
#The network accepts 4 inputs (3 for observation and 1 for reward)  and return a vector with two binary values.
def eval_one_to_many_3x2(agent):
    env = gym.make('OneToMany3x2-v0')
    num_episodes = 2000
    sum = num_episodes
    observation = env.reset(rand_iter=500)
    input = list(observation)
    input.append(0)
    for i_episode in range(num_episodes):
        action = agent.activate(input)
        observation, reward, done, info = env.step(action)
        input = list(input)
        input[-1] = reward
        agent.activate(input)
        input = list(observation)
        input.append(0)
        sum += reward
    env.close()
    return sum

def int_to_action(x):
    if x == 0:
        return "Left"
    if x ==1:
        return "Right"
    return "Forward"

#Assuming 8 episodes and one switch, the optimal fitness for an agent would be 6 * 1 + 2 * 0.2 = 6.4
#The one switch will occur at 2 + y
class TmazeEvaluator():

    DOMAIN_CONSTANT = 2
    def __init__(self, num_episodes = 8,samples = 4, debug =  False, descriptor_out = False):
        self.maxparam = num_episodes - 2*self.DOMAIN_CONSTANT
        self.param_list = [i for i in range(0,self.maxparam+1)]
        self.samples = samples
        self.params = random.sample(self.param_list, self.samples)
        self.num_episodes = num_episodes
        self.debug = debug
        self.descriptor_out = descriptor_out
        self.eval_func = self.eval_tmaze

    def eval_tmaze(self, agent):

        env = gym.make('MiniGrid-TMaze-v0')  #init environment
        s = 0       #s = total reward
        pos = 0     #pos = the initial position of the high reward
        bd = []     # behavioural descriptor

        for param in self.params:
            #pos = 0
            for i_episode in range(self.num_episodes):
                reward = 0
                #swap the position of the high reward every s_inter steps
                if i_episode == self.DOMAIN_CONSTANT + param:
                    pos = (pos + 1) % 2
                observation = env.reset(reward_pos= pos)
                #print(f"y = {param}, pos = {pos}")
                #append 0 for the reward
                input = list(observation)
                input.append(0)
                done = False
                #DEBUG INFO
                if self.debug:
                    print("Episode: {}".format(i_episode))
                    print("High pos: {}".format(pos))
                while not done:
                    action = agent.activate(input)
                    observation, reward, done, info = env.step(action)
                    input = list(observation)
                    input.append(reward)
                    #DEBUG INFO
                    if self.debug:
                        print("     {}".format(int_to_action(action)))
                if self.debug:
                    print(input)
                s += reward
                #Add this episode to the behavioural descriptor
                if self.descriptor_out:
                    if math.isclose(reward, t_maze.LOW):
                        des = 'l'
                    elif math.isclose(reward, t_maze.HIGH):
                        des = 'h'
                    else:
                        des = 'n'
                    if math.isclose(reward, t_maze.CRASH_REWARD):
                        des += 'y'
                    else:
                        des += 'n'
                    bd.append(des)
                agent.activate(input)
                #DEBUG INFO
                if self.debug:
                    print("Reward: {}".format(reward))
                    print("--------------")
        env.close()
        if self.debug:
            print(f"Total reward: {s}")
        if self.descriptor_out:
            return s, bd
        return s

#Older version
#For a network to be considered to be able to solve the single t-maze non-homing task in this case it needs to
#to achieve a score of at least 96 (for the default settings). This is because every time we change the place of the high reward, the optimal
#network needs only one step to figure it out and we change the the place of the high reward 5 times through the
#trial (95 * 1 + 5 * 0.2). Performance of 99 or 98 may indicate that network evolved some interesting properties, e.g. maybe it
#figured out when we change the place of the high reward so it doesn't even need that step to learn.
#The network should accept 4 inputs (is agent at home, is agent at turning point, is agent at maze end, reward) and
#return 1 scalar output
# def eval_tmaze(agent, num_episodes=100, s_inter=20, debug=False, descriptor_out=False):
#
#     env = gym.make('MiniGrid-TMaze-v0')  #init environment
#     s = 0       #s = total reward
#     pos = 0     #pos = the initial position of the high reward
#     bd = []     # behavioural descriptor
#
#     for i_episode in range(num_episodes):
#         reward = 0
#         #swap the position of the high reward every s_inter steps
#         if i_episode % s_inter == 0:
#             pos = (pos + 1) % 2
#         observation = env.reset(reward_pos= pos)
#         #append 0 for the reward
#         input = list(observation)
#         input.append(0)
#         done = False
#         #DEBUG INFO
#         if debug:
#             print("Episode: {}".format(i_episode))
#             print("High pos: {}".format(pos))
#         while not done:
#             action = agent.activate(input)
#             observation, reward, done, info = env.step(action)
#             input = list(observation)
#             input.append(reward)
#             #DEBUG INFO
#             if debug:
#                 print("     {}".format(int_to_action(action)))
#         if debug:
#             print(input)
#         s += reward
#         #Add this episode to the behavioural descriptor
#         if descriptor_out:
#             if math.isclose(reward, t_maze.LOW):
#                 des = 'l'
#             elif math.isclose(reward, t_maze.HIGH):
#                 des = 'h'
#             else:
#                 des = 'n'
#             if math.isclose(reward, t_maze.CRASH_REWARD):
#                 des += 'y'
#             else:
#                 des += 'n'
#             bd.append(des)
#         agent.activate(input)
#         #DEBUG INFO
#         if debug:
#             print("Reward: {}".format(reward))
#             print("--------------")
#     env.close()
#     if debug:
#         print(f"Total reward: {s}")
#     if descriptor_out:
#         return s, bd
#     return s

#For a network to be considered to be able to solve the double t-maze non-homing task in this case it needs to
#to achieve a score of at least 88 (for the default settings). This is because every time we change the place of the high reward, the optimal
#network needs only three steps to figure it out and we change the the place of the high reward 5 times through the
#trial (85 * 1 + 3 * 5 * 0.2). Performance of 99 or 98 may indicate that network evolved some interesting properties, e.g. maybe it
#figured out when we change the place of the high reward so it doesn't even need that step to learn.
#The network should accept 4 inputs (is agent at home, is agent at turning point, is agent at maze end, reward) and
#return 1 scalar output
# def eval_double_tmaze(agent, num_episodes=100, s_inter=20, debug=False, descriptor_out=False):
#
#     env = gym.make('MiniGrid-DoubleTMaze-v0')  #init environment
#     env = env.unwrapped
#     s = 0       #s = total reward
#     pos = 0     #pos = the initial position of the high reward
#     bd = []     # behavioural descriptor
#
#     for i_episode in range(num_episodes):
#         reward = 0
#         #swap the position of the high reward every s_inter steps
#         if i_episode % s_inter == 0:
#             #Mod 4 because we have 4 maze ends
#             pos = (pos + 1) % 4
#         observation = env.reset(reward_pos= pos)
#         #append 0 for the reward
#         input = list(observation)
#         input.append(0)
#         done = False
#         #DEBUG INFO
#         if debug:
#             print("Episode: {}".format(i_episode))
#             print("High pos: {}".format(pos))
#         while not done:
#             action = agent.activate(input)
#             observation, reward, done, info = env.step(action)
#             input = list(observation)
#             input.append(reward)
#             #DEBUG INFO
#             if debug:
#                 print("     {}".format(int_to_action(action)))
#         if debug:
#             print(input)
#         s += reward
#         #Add this episode to the behavioural descriptor
#         if descriptor_out:
#             if env.agent_pos == env.END1:
#                 des = 1
#             elif env.agent_pos == env.END2:
#                 des = 2
#             elif env.agent_pos == env.END3:
#                 des = 3
#             elif env.agent_pos == env.END4:
#                 des = 4
#             else:
#                 des = 0
#             bd.append(des)
#         agent.activate(input)
#         #DEBUG INFO
#         if debug:
#             print("Reward: {}".format(reward))
#             print("--------------")
#     env.close()
#     if debug:
#         print(f"Total reward: {s}")
#     if descriptor_out:
#         return s, bd
#     return s

class DoubleTmazeEvaluator():

    DOMAIN_CONSTANT = 4
    def __init__(self, num_episodes=12, samples=4, debug=False, descriptor_out=False):

        self.maxparam = num_episodes - 2 * self.DOMAIN_CONSTANT
        self.param_list = [i for i in range(0,self.maxparam+1)]
        self.samples = samples
        self.params = random.sample(self.param_list, self.samples)
        self.num_episodes = num_episodes
        self.debug = debug
        self.descriptor_out = descriptor_out
        self.eval_func = self.eval_double_tmaze

    def eval_double_tmaze(self, agent):

        env = gym.make('MiniGrid-DoubleTMaze-v0')  #init environment
        env = env.unwrapped
        s = 0       #s = total reward
        pos = 0     #pos = the initial position of the high reward
        bd = []     # behavioural descriptor
        maze_ends = [0,1,2,3]

        for param in self.params:
            for i_episode in range(self.num_episodes):
                reward = 0
                #swap the position of the high reward every s_inter steps
                if i_episode == self.DOMAIN_CONSTANT + param:
                    choices = copy.deepcopy(maze_ends)
                    choices.remove(pos)
                    pos = random.choice(choices)

                observation = env.reset(reward_pos= pos)
                #append 0 for the reward
                input = list(observation)
                input.append(0)
                done = False
                #DEBUG INFO
                if self.debug:
                    print("Episode: {}".format(i_episode))
                    print("High pos: {}".format(pos))
                while not done:
                    action = agent.activate(input)
                    observation, reward, done, info = env.step(action)
                    input = list(observation)
                    input.append(reward)
                    #DEBUG INFO
                    if self.debug:
                        print("     {}".format(int_to_action(action)))
                if self.debug:
                    print(input)
                s += reward
                #Add this episode to the behavioural descriptor
                if self.descriptor_out:
                    if env.agent_pos == env.END1:
                        des = 1
                    elif env.agent_pos == env.END2:
                        des = 2
                    elif env.agent_pos == env.END3:
                        des = 3
                    elif env.agent_pos == env.END4:
                        des = 4
                    else:
                        des = 0
                    bd.append(des)
                agent.activate(input)
                #DEBUG INFO
                if self.debug:
                    print("Reward: {}".format(reward))
                    print("--------------")
        env.close()
        if self.debug:
            print(f"Total reward: {s}")
        if self.descriptor_out:
            return s, bd
        return s


#Old version
#For a network to be considered to be able to solve the single t-maze non-homing task in this case it needs to
#to achieve a score of at least 96 (for the default settings). This is because every time we change the place of the high reward, the optimal
#network needs only one step to figure it out and we change the the place of the high reward 5 times through the
#trial (95 * 1 + 5 * 0.2). Performance of 99 or 98 may indicate that network evolved some interesting properties, e.g. maybe it
#figured out when we change the place of the high reward so it doesn't even need that step to learn.
#The network should accept 4 inputs (is agent at home, is agent at turning point, is agent at maze end, reward) and
#return 1 scalar output
# def eval_tmaze_homing(agent, num_episodes=100, s_inter=20, debug=False, descriptor_out=False):
#
#     env = gym.make('MiniGrid-TMazeHoming-v0')  #init environment
#     s = 0       #s = total reward
#     pos = 0     #pos = the initial position of the high reward
#     bd = []     # behavioural descriptor
#
#     for i_episode in range(num_episodes):
#         reward = 0
#         #swap the position of the high reward every s_inter steps
#         if i_episode % s_inter == 0:
#             pos = (pos + 1) % 2
#         observation = env.reset(reward_pos= pos)
#         #append 0 for the reward
#         input = list(observation)
#         input.append(0)
#         done = False
#         #DEBUG INFO
#         if debug:
#             print("Episode: {}".format(i_episode))
#             print("High pos: {}".format(pos))
#         while not done:
#             action = agent.activate(input)
#             observation, reward, done, info = env.step(action)
#             input = list(observation)
#             input.append(reward)
#             #DEBUG INFO
#             if debug:
#                 print("     {}".format(int_to_action(action)))
#         if debug:
#             print(input)
#         s += reward
#         #Add this episode to the behavioural descriptor
#         if descriptor_out:
#             if math.isclose(reward, t_maze.LOW):
#                 des = 'l'
#             elif math.isclose(reward, t_maze.HIGH):
#                 des = 'h'
#             else:
#                 des = 'n'
#             if math.isclose(reward, t_maze.CRASH_REWARD):
#                 des += 'y'
#             else:
#                 des += 'n'
#             if math.isclose(reward, t_maze.FAIL_HOME):
#                 des += 'n'
#             else:
#                 des += 'y'
#             bd.append(des)
#
#         agent.activate(input)
#         #DEBUG INFO
#         if debug:
#             print("Reward: {}".format(reward))
#             print("--------------")
#     env.close()
#     if debug:
#         print(f"Total reward: {s}")
#     if descriptor_out:
#         return s, bd
#     return s

class HomingTmazeEvaluator():

    DOMAIN_CONSTANT = 2
    def __init__(self, num_episodes=8, samples=4, debug=False, descriptor_out=False):

        self.maxparam = num_episodes - 2 * self.DOMAIN_CONSTANT
        self.param_list = [i for i in range(0,self.maxparam+1)]
        self.samples = samples
        self.params = random.sample(self.param_list, self.samples)
        self.num_episodes = num_episodes
        self.debug = debug
        self.descriptor_out = descriptor_out
        self.eval_func = self.eval_tmaze_homing

    def eval_tmaze_homing(self, agent):

        env = gym.make('MiniGrid-TMazeHoming-v0')  #init environment
        s = 0       #s = total reward
        pos = 0     #pos = the initial position of the high reward
        bd = []     # behavioural descriptor

        for param in self.params:
            for i_episode in range(self.num_episodes):
                reward = 0
                #swap the position of the high reward every s_inter steps
                if i_episode == self.DOMAIN_CONSTANT + param:
                    pos = (pos + 1) % 2
                observation = env.reset(reward_pos= pos)
                #append 0 for the reward
                input = list(observation)
                input.append(0)
                done = False
                #DEBUG INFO
                if self.debug:
                    print("Episode: {}".format(i_episode))
                    print("High pos: {}".format(pos))
                while not done:
                    action = agent.activate(input)
                    observation, reward, done, info = env.step(action)
                    input = list(observation)
                    input.append(reward)
                    #DEBUG INFO
                    if self.debug:
                        print("     {}".format(int_to_action(action)))
                if self.debug:
                    print(input)
                s += reward
                #Add this episode to the behavioural descriptor
                if self.descriptor_out:
                    if math.isclose(reward, t_maze.LOW):
                        des = 'l'
                    elif math.isclose(reward, t_maze.HIGH):
                        des = 'h'
                    else:
                        des = 'n'
                    if math.isclose(reward, t_maze.CRASH_REWARD):
                        des += 'y'
                    else:
                        des += 'n'
                    if math.isclose(reward, t_maze.FAIL_HOME):
                        des += 'n'
                    else:
                        des += 'y'
                    bd.append(des)

                agent.activate(input)
                #DEBUG INFO
                if self.debug:
                    print("Reward: {}".format(reward))
                    print("--------------")
            env.close()
            if self.debug:
                print(f"Total reward: {s}")
            if self.descriptor_out:
                return s, bd
            return s

#The simplest case for test the network's implementation. Use when in doubt about the rest.
#Optimal network should have a score of 4 or very close to 4.
#The network takes 2 input and returns one output

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

TRIALS = 10

def eval_net_xor(net):
    sum = 0
    for i in range(TRIALS):
        fitness = 4
        trial_in, trial_out = shuffle_lists(xor_inputs, xor_outputs)
        for xi, xo in zip(trial_in, trial_out):
            output = net.activate(xi)
            fitness -= abs(output[0] - xo[0])
        sum += fitness
    return sum/TRIALS

class NoveltyEvaluator():

    #distance_func: a function that computes the distance of two behavioural descriptors
    #k: the number used for the nearest neighbour calculation
    #threshold: the sparseness threshold for a gene to enter the archive
    #The archive has the key of the genome as the key and another dictionary as values with bd, novelty and fitness
    def __init__(self, eval_func, threshold = 1, k = 15):
        self.eval_func = eval_func
        self.threshold = threshold
        self.k = k
        self.archive = {}
        self.visited_novelty = {}

    def distance_func(self, bd1, bd2):
        assert False, "NoveltyEvaluator.distance_func needs to be overloaded!"

    def get_best_id(self):

        maxid = list(self.archive.keys())[0]
        maxfitness = self.archive[maxid]['fitness']
        for key in self.archive:
            if self.archive[key]['fitness'] > maxfitness:
                maxfitness = self.archive[key]['fitness']
                maxid = key

        return maxid

    #Find the novelty score for an agent.
    def eval(self,key, agent):
        if key in self.visited_novelty:
            return self.visited_novelty[key]
        fitness, bd = self.eval_func(agent)
        cache = []
        if not self.archive:
            self.archive[key] = {'bd': bd, 'novelty': 0, 'fitness':fitness, 'agent' : agent}
            return len(bd)
        for k in self.archive:
            dist = self.distance_func(bd, self.archive[k]['bd'])
            if len(cache) > self.k and dist < min(cache):
                cache.remove(min(cache))
            if len(cache) < self.k:
                cache.append(dist)

        novelty = sum(cache) / len(cache)
        self.visited_novelty[key] = novelty
        if novelty > self.threshold:
            self.archive[key] = {'bd': bd, 'novelty': novelty, 'fitness':fitness, 'agent': agent}
        return novelty



class TmazeNovelty(NoveltyEvaluator):

    def __init__(self, n_episodes, samples, threshold = 50):
        self.evaluator = TmazeEvaluator(num_episodes=n_episodes,samples=samples, debug =  False, descriptor_out = True)
        eval_func = self.evaluator.eval_tmaze
        super().__init__(eval_func, threshold=threshold)

    def distance_func(self, bd1, bd2):
        total = 0
        for t1, t2 in zip(bd1, bd2):
            for i in range(len(t1)):
                if t1[i] != t2[i]:
                    total += 1

        return total

class DoubleTmazeNovelty(NoveltyEvaluator):

    def __init__(self, n_episodes, samples, threshold=50):
        self.evaluator = DoubleTmazeEvaluator(num_episodes=n_episodes,samples=samples,debug=False,descriptor_out=True)
        eval_func = self.evaluator.eval_double_tmaze
        super().__init__(eval_func, threshold=threshold)

    def distance_func(self, bd1, bd2):
        score = 0
        for i, j in zip(bd1, bd2):
            if i != j:
                score += 1
        return score

class HomingTmazeNovelty(NoveltyEvaluator):

    def __init__(self, n_episodes, samples, threshold=150):
        self.evaluator = HomingTmazeEvaluator(num_episodes=n_episodes, samples=samples,debug=False, descriptor_out=True)
        eval_func = self.evaluator.eval_tmaze_homing
        super().__init__(eval_func, threshold=threshold)

    def distance_func(self, bd1, bd2):
        total = 0
        for t1, t2 in zip(bd1, bd2):
            for i in range(len(t1)):
                if t1[i] != t2[i]:
                    total += 1

        return total
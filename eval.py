###
# This file contains evaluation functions for all the problems used in the experiments.
#All the following evaluation functions take as argument an agent variable. It is assumed that the agent has
#an activate function which takes as input a vector (list) and returns an output which corresponds to the action
#of the agent. The actions are described in each environment
##############
import copy
import math
import random
import gym
import gym_association_task
import t_maze
from functools import partial
#import logging
#logging.basicConfig(filename="skinner.log", level=logging.DEBUG, format="%(message)s")
import numpy as np
from utilities import shuffle_lists


def eq_tuples(tup1, tup2):
    for x,y in zip(tup1, tup2):
        if x != y:
            return False
    return True

def eq_snapshots(s1,s2):
    for key in s1:
        if not eq_tuples(s1[key], s2[key]):
            return False
    return True

#For a network to be considered to be able to solve the one-to-one 3x3 association task in this case it needs to
#to achieve a score of at least 1976 (2000 - (5*(3*2))  = steps - (association_changes+1)*(n*(m-1)).
#Note that scores above this threshold do not mean better performance since the score of 1976 is already considered optimal.
#The network here needs to accept 4 inputs (3 for observation and 1 for reward) and return a vector with 3 binary values.
#For num_episodes = 100 | 1000  |   200
#    rand_iter = 25     | 100   |   20
#    max fitness = 70   | 940   |   140
# def eval_one_to_one(env_name, agent, num_episodes, rand_iter,snapshot_inter, descriptor_out=False, debug=False):
#     env = gym.make(env_name)
#     s = num_episodes
#     observation = env.reset(rand_iter=rand_iter)
#     input = tuple(list(observation) + [0])
#     responses = {}
#     prevsnapshot = {}
#     bd = []
#     for i_episode in range(num_episodes):
#         action = agent.activate(input)
#         if descriptor_out:
#             t_in = input[:-1]
#             if t_in not in prevsnapshot:
#                 prevsnapshot[t_in] = action
#             responses[t_in] = action
#             #if i_episode != snapshot_inter and i_episode%snapshot_inter == 0 and i_episode>0:
#             if i_episode%snapshot_inter == 0 and i_episode > 0:
#                 if eq_snapshots(responses, prevsnapshot):
#                     bd.append(0)
#                 else:
#                     bd.append(1)
#                 prevsnapshot = copy.deepcopy(responses)
#         observation, reward, done, info = env.step(action)
#         # if debug:
#         #     logging.debug(f"Episode{i_episode}:\tInput: {input}\t Action:{action} Reward:{reward}")#debug
#         input = list(input)
#         input[-1] = reward
#         agent.activate(input)
#         input = tuple(list(observation) + [0])
#         s += reward
#     env.close()
#     if descriptor_out:
#         return s, bd
#         #print(bd)
#     else:
#         return s

# #Version 2, separate training and test associations and change the descriptor
# def eval_one_to_one(env_name, agent, num_episodes=72, rand_iter=12,snapshot_inter=3, descriptor_out=False,
#                     mode = None,debug=False):
#     env = gym.make(env_name)
#     s = num_episodes
#     observation = env.reset(rand_iter=rand_iter, mode = mode)
#     input = tuple(list(observation) + [0])
#     responses = {}
#     prevsnapshot = {}
#     bd = []
#     for i_episode in range(1,num_episodes+1):
#         action = agent.activate(input)
#         if descriptor_out:
#             t_in = input[:-1]
#             if t_in not in prevsnapshot:
#                 prevsnapshot[t_in] = action
#             responses[t_in] = action
#             #if i_episode != snapshot_inter and i_episode%snapshot_inter == 0 and i_episode>0:
#             if i_episode%snapshot_inter == 0 and i_episode > 0:
#                 #if i_episode != snapshot_inter:
#                 if eq_snapshots(responses, prevsnapshot):
#                     bd.append(0)
#                 else:
#                     bd.append(1)
#                 prevsnapshot = copy.deepcopy(responses)
#         observation, reward, done, info = env.step(action)
#         # if debug:
#         #     logging.debug(f"Episode{i_episode}:\tInput: {input}\t Action:{action} Reward:{reward}")#debug
#         input = list(input)
#         input[-1] = reward
#         agent.activate(input)
#         input = tuple(list(observation) + [0])
#         s += reward
#     env.close()
#
#     #prepare bd##########
#     new_bd = copy.deepcopy(bd)
#     for i in range(len(bd)):
#         if i % 4 == 1 or i % 4 == 2:
#             new_bd[i] = None
#         elif i % 4 == 3:
#             new_bd[i] = int(any((new_bd[i-1], new_bd[i])))
#     new_bd = [x for x in new_bd if x is not None]
#     #####################
#     if descriptor_out:
#         return s, new_bd
#         #print(bd)
#     else:
#         return s

# Generic evaluation method for all the one-to-one association problems
#Version 3, separate training and test associations and change the descriptor, 27 associations, float descriptor
def eval_one_to_one(env_name, agent, num_episodes=72, rand_iter=12,snapshot_inter=3, descriptor_out=False,
                    mode = None, trials = 20, debug=False):
    env = gym.make(env_name)
    bds = []
    scores = []
    for trial in range(trials):
        s = num_episodes
        observation = env.reset(rand_iter=rand_iter, mode = mode)
        input = tuple(list(observation) + [0])
        responses = {}
        prevsnapshot = {}
        bd = []
        for i_episode in range(1,num_episodes+1):
            action = agent.activate(input)
            if descriptor_out:
                t_in = input[:-1]
                if t_in not in prevsnapshot:
                    prevsnapshot[t_in] = action
                else:
                    prevsnapshot[t_in] = responses[t_in]
                responses[t_in] = action
                if eq_tuples(prevsnapshot[t_in], responses[t_in]):
                    bd.append(0)
                else:
                    bd.append(1)
            observation, reward, done, info = env.step(action)
            # if debug:
            #     logging.debug(f"Episode{i_episode}:\tInput: {input}\t Action:{action} Reward:{reward}")#debug
            input = list(input)
            input[-1] = reward
            agent.activate(input)
            input = tuple(list(observation) + [0])
            s += reward

        #prepare bd
        if descriptor_out:
            new_bd = []
            while bd:
                for i in range(snapshot_inter):
                    bd.pop(0)
                c1 = 0
                for i in range(8*snapshot_inter):
                    c1 += bd.pop(0)
                new_bd.append(c1/(8*snapshot_inter))
                c2 = 0
                for i in range(snapshot_inter):
                    c2 += bd.pop(0)
                new_bd.append(c2/snapshot_inter)
            bds.append(new_bd)
        scores.append(s)
    env.close()
    fitness = float(np.average(scores))
    if descriptor_out:
        bdsarr = np.array(bds, dtype=float)
        avg = bdsarr.mean(axis=0)
        bd = avg.tolist()
        # print("bds:", bds)
        # print("bd:", bd)

        return fitness, bd
        #print(bd)
    else:
        return fitness

def eval_one_to_one_3x3(agent, num_episodes = 200, rand_iter= 40,snapshot_inter=20, descriptor_out=False,
                        mode='training', trials=30, debug=False):
    return eval_one_to_one('OneToOne3x3-v0', agent, num_episodes, rand_iter, snapshot_inter, descriptor_out,mode,trials, debug)

def eval_one_to_one_2x2(agent, num_episodes = 50, rand_iter= 10,snapshot_inter=5, descriptor_out=False,mode='training',
                        trials=10,debug=False):
    return eval_one_to_one('OneToOne2x2-v0', agent, num_episodes, rand_iter, snapshot_inter, descriptor_out,mode,trials, debug)

def eval_one_to_one_4x4(agent, num_episodes = 200, rand_iter= 40,snapshot_inter=20, descriptor_out=False,mode='trainig',
                        trials=10, debug=False):
    return eval_one_to_one('OneToOne4x4-v0', agent, num_episodes, rand_iter, snapshot_inter, descriptor_out,mode, trials,debug)

#n=300, r=100, threshold = 219
def eval_one_to_one_3x10(agent, num_episodes=300, rand_iter=100, snapshot_inter=10, descriptor_out=False,
                        mode='training', trials=10, debug=False):
    return eval_one_to_one('OneToOne3x10-v0', agent, num_episodes, rand_iter, snapshot_inter,descriptor_out, mode, trials,debug)

#For a network to be considered to be able to solve the one-to-many 3x2 association task in this case it needs to
#to achieve a score of at least 1964 (2000 - 4*(3*(4-1)) = steps - association_changes*(n*(2^m - 1)).
#Note that scores above this threshold do not mean better performance since the score of 1964 is already considered optimal.
#The network accepts 4 inputs (3 for observation and 1 for reward)  and return a vector with two binary values.
#evals  | rand  | fitthresh
#1000   | 200   | 955
def eval_one_to_many_3x2(agent):
    env = gym.make('OneToMany3x2-v0')
    num_episodes = 1000
    sum = num_episodes
    observation = env.reset(rand_iter=200)
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

#The following evaluator classes were used with my implementation of the novelty search. I deemed it necessary
# to create these because it was needed to associate each evaluation function with a distance function as well as
# reevaluate the archive at every generation.
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

    def eval_float_bd(self, agent):

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
                    bd.append(reward)
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

#In this second version we follow the implementation described in the novelty search paper
#Max fitness: 100 total trials, an optimal agent is allowed to make mistakes:
#for the 1-scenario: 2 -> max fitness = 100 - 2*0.8 = 98.4
#for the 5-scenario: 10 -> max fitness = 100 - 10*0.8 = 92
#for the 10-scenario: 20 -> max fitness = 100 - 20*0.8 = 84
def eval_tmaze_v2(agent, scenario=5):

    env = gym.make('MiniGrid-TMaze-v0')  #init environment
    s = 0.0       #s = total reward
    bd = []     # behavioural descriptor
    deployments = 10
    trials = 10
    switch_points = None
    if scenario == 1:
        switch_points = [5]
    elif scenario == 5:
        switch_points = [1,3,5,7,9]
    elif scenario == 10:
        switch_points = [0,1,2,3,4,5,6,7,8,9]

    for deployment in range(deployments):
        pos = 0 #pos = the initial position of the high reward
        for trial in range(trials):
            reward = 0
            #swap the position of the high reward
            if trial == trials//2 and deployment in switch_points:
                pos = 1
            observation = env.reset(reward_pos= pos)
            #append 0 for the reward
            input = list(observation)
            input.append(0)
            done = False
            #DEBUG INFO
            # if debug:
            #     print("Episode: {}".format(i_episode))
            #     print("High pos: {}".format(pos))
            while not done:
                action = agent.activate(input)
                observation, reward, done, info = env.step(action)
                input = list(observation)
                input.append(reward)
                #DEBUG INFO
                # if debug:
                #     print("     {}".format(int_to_action(action)))
            # if debug:
            #     print(input)
            s += reward
            #Add this episode to the behavioural descriptor
            bd.append(reward)
            agent.activate(input) #We activate the network a second time to allow it to calibrate itself if it needs to
            #DEBUG INFO
            # if debug:
            #     print("Reward: {}".format(reward))
            #     print("--------------")
    env.close()
    # if debug:
    #     print(f"Total reward: {s}")
    # if descriptor_out:
    #     return s, bd
    return s, bd

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

    #num episodes = 12
    #samples 4
    #max fitness = 36
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
                    if eq_tuples(env.agent_pos, env.END1):
                        des = 1
                    elif eq_tuples(env.agent_pos, env.END2):
                        des = 2
                    elif eq_tuples(env.agent_pos, env.END3):
                        des = 3
                    elif eq_tuples(env.agent_pos, env.END4):
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
    #The archive has the key of the genome as the key and another dictionary as values with bd, novelty and fitness, genome
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
    def eval(self,key,genome ,agent):
        #if key in self.visited_novelty:
        #    return self.visited_novelty[key]
        fitness, bd = self.eval_func(agent)

        cache = []
        if not self.archive:
            self.archive[key] = {'bd': bd, 'novelty': len(bd), 'fitness':fitness, 'agent' : agent, 'genome': genome}
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
            self.archive[key] = {'bd': bd, 'novelty': novelty, 'fitness':fitness, 'agent': agent, 'genome': genome}
        return novelty

    def reevaluate_archive(self):
        if not self.archive:
            return
        for key in self.archive:
            agent = self.archive[key]['agent']
            fitness, bd = self.eval_func(agent)
            self.archive[key]['bd'] = bd
            self.archive[key]['fitness'] = fitness
            #print(bd)

class TmazeNovelty(NoveltyEvaluator):

    def __init__(self, n_episodes, samples, threshold = 8):
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

    def __init__(self, n_episodes, samples, threshold=10):
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

    def __init__(self, n_episodes, samples, threshold=1):
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

class AssociationNovelty(NoveltyEvaluator):

    def __init__(self, num_episodes, rand_iter,snapshot_inter,threshold=2):
        eval_f = partial(eval_one_to_one_3x3,num_episodes = num_episodes, rand_iter= rand_iter,
                                     snapshot_inter=snapshot_inter, descriptor_out=True)
        super().__init__(eval_f, threshold=threshold)

    def distance_func(self, bd1, bd2):
        sum = 0
        for x,y in zip(bd1,bd2):
            if x!=y:
                sum += 1
        return sum

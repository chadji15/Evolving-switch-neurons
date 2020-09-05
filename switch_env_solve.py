from math import tanh

from switch_neuron import Neuron, SwitchNeuron, SwitchNeuronNetwork
import gym
import copy
import gym_association_task
from t_maze.envs.t_maze import TMazeEnv
import t_maze

#In this script I try to recreate a network designed by hand which solves 3x3 one-to-one association tasks using
#switch neuron.

def heaviside(x):
    if x < 0:
        return 0
    return 1

def clamp(x,low,high):
    if x < low:
        return low
    if x > high:
        return high
    return x

def mult(w_inputs):
    product = 1
    for w_i in w_inputs:
        product *= w_i
    return product

def convert_to_action(scalar):
    if scalar > 3.3:
        return (1,0,0)
    if scalar < -3.3:
        return (0,0,1)
    return (0,1,0)

def eval_one_to_one(network):
    env = gym.make('OneToOne3x3-v0')
    num_episodes = 2000
    s = num_episodes
    observation = env.reset(rand_iter=500)
    input = tuple(list(observation) + [0])
    for i_episode in range(num_episodes):
        output = network.activate(input)[0]
        action = convert_to_action(output)
        observation, reward, done, info = env.step(action)
        input = list(input)
        input[-1] = reward
        network.activate(input)
        input = tuple(list(observation) + [0])
        s += reward
    env.close()
    return s

def eval_one_to_many(network):
    env = gym.make('OneToMany3x2-v0')
    num_episodes = 2000
    sum = num_episodes
    observation = env.reset(rand_iter=500)
    input = tuple(list(observation) + [0])
    for i_episode in range(num_episodes):
        output = network.activate(input)
        action = tuple(output)
        observation, reward, done, info = env.step(action)
        input = list(input)
        input[-1] = reward
        network.activate(input)
        input = tuple(list(observation) + [0])
        sum += reward
    env.close()
    return sum

def solve_one_to_one_3x3():
    input_keys = [-1, -2, -3, -4]
    output_keys = [0]
    switch_keys = [1, 2, 3]
    node_keys = [4, 5, 6]

    nodes = []

    modulating_nodes_dict = {
        'activation_function': lambda x: clamp(x,-10,10),
        'integration_function': mult,
        'activity': 0,
        'output': 0,
    }

    node_weights = {4: [(-1, 1), (-4, 1)], 5: [(-2, 1), (-4, 1)], 6: [(-3, 1), (-4, 1)]}
    for i in node_keys:
        node_dict = copy.deepcopy(modulating_nodes_dict)
        node_dict['weights'] = node_weights[i]
        nodes.append(Neuron(i, node_dict))

    switch_std_weights = {
        1: [(-1, 10), (-1, 0), (-1, -10)],
        2: [(-2, 10), (-2, 0), (-2, -10)],
        3: [(-3, 10), (-3, 0), (-3, -10)]
    }
    switch_mod_weights = {
        1: [(4, 1 / 3)],
        2: [(5, 1 / 3)],
        3: [(6, 1 / 3)]
    }
    for key in switch_keys:
        nodes.append(SwitchNeuron(key, switch_std_weights[key], switch_mod_weights[key]))

    node_0_std = {
        'activation_function': lambda x: clamp(x,-10,10),
        'integration_function': sum,
        'activity': 0,
        'output': 0,
        'weights': [(1, 1), (2, 1), (3, 1)]
    }
    nodes.append(Neuron(0, node_0_std))

    net = SwitchNeuronNetwork(input_keys, output_keys, nodes)
    score = eval_one_to_one(net)
    print("Score: {}".format(score))

def solve_one_to_many():

    input_keys = [-1, -2, -3, -4]
    output_keys = [0,1]
    node_keys = [3,4,5]
    switch_keys = [7,8,9,10,11,12]

    nodes = []

    node_weights = {3: [(-1, 1), (-4, 1)], 4: [(-2, 1), (-4, 1)], 5: [(-3, 1), (-4, 1)]}
    modulating_nodes_dict = {
        'activation_function': lambda x: clamp(x,-1,1),
        'integration_function': mult,
        'activity': 0,
        'output': 0,
    }
    for i in node_keys:
        node_dict = copy.deepcopy(modulating_nodes_dict)
        node_dict['weights'] = node_weights[i]
        nodes.append(Neuron(i, node_dict))

    slow, fast = 0,0
    switch_std_w = {}
    while fast < len(switch_keys):
        switch_std_w[switch_keys[fast]] = [(input_keys[slow], 1), (input_keys[slow], -1)]
        fast += 1
        switch_std_w[switch_keys[fast]] = [(input_keys[slow], 1), (input_keys[slow], -1)]
        fast+=1
        slow+=1

    w = 0.5
    switch_mod_w = {7: [(3,w)], 8: [(7,w)], 9: [(4,w)], 10:[(9,w)], 11: [(5,w)], 12: [(11,w)]}

    for key in switch_keys:
        nodes.append(SwitchNeuron(key,switch_std_w[key],switch_mod_w[key]))

    out_w = {0 : [(8,1), (10,1), (12,1)], 1: [(7,1), (9,1), (11,1)]}
    out_dict = {
        'activation_function': heaviside,
        'integration_function': sum,
        'activity': 0,
        'output': 0,
    }
    for key in output_keys:
        params = copy.deepcopy(out_dict)
        params['weights'] = out_w[key]
        nodes.append(Neuron(key,params))

    net = SwitchNeuronNetwork(input_keys,output_keys,nodes)
    score = eval_one_to_many(net)
    print("Score: {}".format(score))


def eval_tmaze(network):
    env = gym.make('MiniGrid-TMaze-v0')
    num_episodes = 100
    s = 0
    pos = 0
    for i_episode in range(num_episodes):
        reward = 0
        if i_episode % 20 == 0:
            pos = (pos + 1) % 2
        observation = env.reset(reward_pos= pos)
        # 1 for the bias term and 0 for the reward
        input = list(observation)
        input.insert(0, 1)
        input.append(0)
        done = False
        #DEBUG INFO
        #print("Episode: {}".format(i_episode))
        #print("High pos: {}".format(pos))
        while not done:
            output = network.activate(input)[0]
            if output < -0.33:
                action = TMazeEnv.Actions.left
            elif output > 0.33:
                action = TMazeEnv.Actions.right
            else:
                action = TMazeEnv.Actions.forward
            observation, reward, done, info = env.step(action)
            input = list(observation)
            input.insert(0,1)
            input.append(reward)
            #DEBUG INFO
            #print("     {}".format(int_to_action(action)))
        #print(input)
        s += reward
        network.activate(input)
        #DEBUG INFO
        #print("Reward: {}".format(reward))
        #print("--------------")
    env.close()
    return s

def int_to_action(x):
    if x == 0:
        return "Left"
    if x ==1:
        return "Right"
    return "Forward"

def solve_tmaze():

    input_keys = [-1,-2,-3,-4,-5]
    output_keys = [0]
    node_keys = [1,2,3]

    nodes = []

    #Aggregating neuron
    params = {
        'activation_function' : lambda x : x,
        'integration_function' : sum,
        'activity': 0,
        'output' : 0,
        'weights' : [(-1,-1), (-5,1)]
    }
    nodes.append(Neuron(1,params))

    m_params = {
        'activation_function': lambda x: clamp(mult(x), -0.8,0),
        'integration_function': lambda x: x,
        'activity': 0,
        'output': 0,
        'weights': [(1, 1), (-4, 1)]
    }
    nodes.append(Neuron(2,m_params))

    std_weights = [(-3,5), (-3,-5)]
    mod_weights = [(2,-1.25*0.5)]
    nodes.append(SwitchNeuron(3,std_weights,mod_weights))

    o_params = {
        'activation_function': tanh,
        'integration_function': sum,
        'activity': 0,
        'output': 0,
        'weights': [(3,1)]
    }
    nodes.append(Neuron(0,o_params))

    net = SwitchNeuronNetwork(input_keys,output_keys,nodes)
    score = eval_tmaze(net)
    print("Score: {:.3f}".format(score))

if __name__ == '__main__':

    solve_tmaze()

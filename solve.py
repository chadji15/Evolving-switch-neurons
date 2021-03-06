import copy
from collections import namedtuple

from switch_neuron import Neuron, SwitchNeuron, SwitchNeuronNetwork, Agent
from math import tanh
from t_maze.envs import TMazeEnv
from utilities import mult, clamp, heaviside


def convert_to_action(scalar):
    if scalar[0] > 3.3:
        return (1,0,0)
    if scalar[0] < -3.3:
        return (0,0,1)
    return (0,1,0)
#Returns an agent which solves the 3x3 one-to-one association task
#We say that a network solves this problem when it manages to learn a new association within n*(m-1) steps,
#in this case 6 steps.

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
        'bias':1
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
        'weights': [(1, 1), (2, 1), (3, 1)],
        'bias' : 0
    }
    nodes.append(Neuron(0, node_0_std))

    net = SwitchNeuronNetwork(input_keys, output_keys, nodes)
    agent = Agent(net,lambda x: x,lambda x: convert_to_action(x))
    return agent

#Returns an agent which solves the 3x3 one-to-one association task
#We say that a network solves this problem when it manages to learn a new association within n*(2^m - 1) steps,
#in this case 9 steps.
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
        'bias' : 1
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

    w1, w2 = 0.5, 1
    switch_mod_w = {7: [(3,w2)], 8: [(7,w1)], 9: [(4,w2)], 10:[(9,w1)], 11: [(5,w2)], 12: [(11,w1)]}

    for key in switch_keys:
        nodes.append(SwitchNeuron(key,switch_std_w[key],switch_mod_w[key]))

    out_w = {0 : [(8,1), (10,1), (12,1)], 1: [(7,1), (9,1), (11,1)]}
    out_dict = {
        'activation_function': heaviside,
        'integration_function': sum,
        'activity': 0,
        'output': 0,
        'bias' : 0
    }
    for key in output_keys:
        params = copy.deepcopy(out_dict)
        params['weights'] = out_w[key]
        nodes.append(Neuron(key,params))

    net = SwitchNeuronNetwork(input_keys,output_keys,nodes)
    return net

def convert_to_direction(x):
    if x[0] < -0.33:
        return  TMazeEnv.Actions.left
    if x[0] > 0.33:
        return TMazeEnv.Actions.right
    return TMazeEnv.Actions.forward

#Returns an agent which solves the single t-maze non-homing task.
#We say a network solves the problem when it it needs at most one step figure out that the high reward has switched
#positions.
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
        'weights' : [(-1,-1), (-5,1)],
        'bias':0
    }
    nodes.append(Neuron(1,params))

    m_params = {
        'activation_function': lambda x: clamp(x, -0.8,0),
        'integration_function': mult,
        'activity': 0,
        'output': 0,
        'weights': [(1, 1), (-4, 1)],
        'bias' : 1
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
        'weights': [(3,1)],
        'bias' : 0
    }
    nodes.append(Neuron(0,o_params))

    net = SwitchNeuronNetwork(input_keys,output_keys,nodes)
    #For input, append the bias to -1 input
    agent = Agent(net, append_bias, convert_to_direction)
    return agent

def append_bias(x):
    x.insert(0,1)
    return x

def solve_xor_rec():

    in_keys = [-1,-2]
    out_keys = [0]
    hidden_keys = [1]

    nodes = []
    nodes.append(Neuron(1, {
        'activation_function': heaviside,
        'integration_function': sum,
        'bias' : -1.5,
        'activity': 0,
        'output': 0,
        'weights': [(-1, 1),(-2,1)]
    }))

    nodes.append(Neuron(0, {
        'activation_function': heaviside,
        'integration_function': sum,
        'bias': -0.5,
        'activity': 0,
        'output': 0,
        'weights': [(-1,1),(-2,1),(1,-10)]
    }))

    net = SwitchNeuronNetwork(in_keys,out_keys,nodes)
    return net

def binary_3x3_optimal_genome():
    Config = namedtuple("Config",["genome_config"])
    Genome_config = namedtuple("Genome_config", ["input_keys", "output_keys", "activation_defs", "aggregation_function_defs"])
    from utilities import identity
    from neat.activations import sigmoid_activation
    from neat.aggregations import sum_aggregation, product_aggregation
    activations = {
        "sigmoid": sigmoid_activation,
        "identity": identity
    }
    aggregations = {
        "sum": sum_aggregation,
        "product": product_aggregation
    }
    genome_config = Genome_config(input_keys=[-1,-2], output_keys= [0], activation_defs=activations, aggregation_function_defs=aggregations)
    config = Config(genome_config=genome_config)
    Genome = namedtuple("Genome", ["nodes", "connections"])
    Node = namedtuple("Node", ['bias', 'activation', 'aggregation', 'is_isolated', 'is_switch'])
    Connection = namedtuple("Connection", ["key", "one2one", "extended", "uniform", "weight", "enabled", "is_mod"])

    gatinglayer = Node(bias=1, activation='identity', aggregation='product', is_isolated=False, is_switch=False)
    switchlayer = Node(bias=0, activation='identity', aggregation='sum', is_isolated=False, is_switch=True)
    outputlayer = Node(bias=0, activation='identity', aggregation="sum", is_isolated=True, is_switch=False)

    nodes = {0: outputlayer,
             1: gatinglayer,
             2: switchlayer}

    inpgatconn = Connection(key = (-1,1), one2one=True, extended=False, uniform=True, weight=1, enabled=True, is_mod=False)
    inpswiconn = Connection(key = (-1,2),one2one=True, extended=True, uniform=False, weight=10, enabled=True, is_mod=False)
    gatswiconn = Connection(key = (1,2),one2one=True, extended=False, uniform=True, weight=0.33, enabled=True, is_mod=True)
    rewgatconn = Connection(key = (-2,1),one2one=True, extended=False, uniform=True, weight=1, enabled=True, is_mod=False)
    swioutconn = Connection(key = (2,0),one2one=True, extended=False, uniform=True, weight=1, enabled=True, is_mod=False)

    connections = {
        (-1,1) : inpgatconn,
        (-1,2) : inpswiconn,
        (-2,1) : rewgatconn,
        (1,2) : gatswiconn,
        (2,0) : swioutconn
    }

    genome = Genome(nodes, connections)
    import extended_maps
    network = extended_maps.create(genome, config, 3)
    import render_network
    render_network.draw_net(network,False, "optimal3x3")
    from eval import eval_one_to_one_3x3
    from switch_neuron import Agent
    agent  = Agent(network,reorder_inputs, convert_to_action)
    score = eval_one_to_one_3x3(agent, 1000, 100)
    print(score)

    # agent = solve_one_to_one_3x3()
    # score = eval_one_to_one_3x3(agent, 1000, 100)
    # print(score)

#For the guided maps encoding
#input order is: input1, reward, input2, input3
def reorder_inputs(l):
    new_l = [l[0], l[3],l[1],l[2]]
    return new_l

if __name__ == "__main__":
    binary_3x3_optimal_genome()
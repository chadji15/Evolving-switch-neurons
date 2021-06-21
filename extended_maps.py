#####################################
#This is an early implementation of a preliminary experiment with very specific guidance provided to NEAT
#in order to see if it can help the process. If so, then we have a good case for trying with a more
#generalized toolkit.
#######################################
import copy
import os
import pickle
import sys
import numpy as np
import Reporters
import neat
import render_network
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from switch_neuron import Neuron, SwitchNeuronNetwork, SwitchNeuron, Agent
from neat.six_util import itervalues
import math
from utilities import order_of_activation, identity, clamp
from functools import partial

class ExtendedMapConnectionGene(BaseGene):

    #Various parameters for defining a connection.
    _gene_attributes = [BoolAttribute('one2one'), #if true then the connection scheme is one to one, else one to all
                        BoolAttribute('extended'),
                        BoolAttribute('uniform'),  #step or uniform
                        FloatAttribute('weight'),#Weigth is used as the mean of the normal distribution for 1-to-all
                        BoolAttribute('enabled'),
                        BoolAttribute('is_mod')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    #Define the distance between two genes
    def distance(self, other, config):
        d = abs(self.weight - other.weight) + int(self.one2one == other.one2one) \
            + int(self.uniform == other.uniform) + int(self.enabled == other.enabled) + int(self.is_mod == other.is_mod) \
            + int(self.extended == other.extended)
        return d * config.compatibility_weight_coefficient

class ExtendedMapNodeGene(DefaultNodeGene):

    _gene_attributes = [FloatAttribute('bias'), #The bias of the neuron
                        StringAttribute('activation', options='sigmoid'), # The activation function, tunable from the config
                        StringAttribute('aggregation', options='sum'), #The aggregation function
                        BoolAttribute('is_isolated'), #Map vs isolated neuron
                        BoolAttribute('is_switch')]

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + int(self.activation == other.activation) + int(self.aggregation == other.aggregation) \
            + int(self.is_isolated == other.is_isolated) + int(self.is_switch - other.is_switch)
        return d * config.compatibility_weight_coefficient


class GuidedMapGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = ExtendedMapNodeGene
        param_dict['connection_gene_type'] =ExtendedMapConnectionGene
        return DefaultGenomeConfig(param_dict)

def calculate_weights(is_uniform, weight, map_size):
    if is_uniform:
        return [weight for _ in range(map_size)]
    start = -weight
    end = weight
    weights = list(np.linspace(start, end, map_size, endpoint=True))
    return weights

def create(genome, config, map_size):
    """ Receives a genome and returns its phenotype (a SwitchNeuronNetwork). """
    genome_config = config.genome_config
    #required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

    input_keys = copy.deepcopy(genome_config.input_keys)
    output_keys = copy.deepcopy(genome_config.output_keys)
    # Gather inputs and expressed connections.
    std_inputs = {}
    mod_inputs = {}
    children = {}
    node_keys = set(genome.nodes.keys())  # + list(genome_config.input_keys[:])
    aux_keys = set()


    # Here we populate the children dictionary for each unique not isolated node.
    for n in genome.nodes.keys():

        children[n] = []
        if n in output_keys:
            continue

        if not genome.nodes[n].is_isolated:
            for _ in range(1, map_size):
                new_idx = max(node_keys) + 1
                children[n].append(new_idx)
                node_keys.add(new_idx)

    #assume 2 input nodes: the first one will be scaled to a map and the second one will represent the reward
    n = input_keys[0]
    children[n] = []
    for _ in range(1, map_size):
        new_idx = min(input_keys) - 1
        children[n].append(new_idx)
        input_keys.append(new_idx)
    n = input_keys[1]
    children[n] = []

    # We don't scale the output with the map size to keep passing the parameters of the network easy.
    # This part can be revised in the future
    for n in output_keys:
        children[n] = []
    #Iterate over every connection
    for cg in itervalues(genome.connections):
        #If it's not enabled don't include it
        # if not cg.enabled:
        #     continue

        i, o = cg.key
        #If neither node is required for output then skip the connection
        # if o not in required and i not in required:
        #     continue

        #Find the map corresponding to each node of the connection
        in_map = [i] + children[i]
        out_map = [o] + children[o]
        #If the connection is modulatory and the output neuron a switch neuron then the new weights are stored
        #in the mod dictionary. We assume that only switch neurons have a modulatory part.
        if cg.is_mod and genome.nodes[o].is_switch:
            node_inputs = mod_inputs
        else:
            node_inputs = std_inputs
        for n in out_map:
            if n not in node_inputs.keys():
                node_inputs[n] = []

        if len(in_map) == map_size and len(out_map) == map_size:
            # Map to map connectivity
            if cg.one2one:
                if cg.extended:
                    #extended one-to-
                    #create a new intermediatery map
                    for j in range(0, map_size):
                        idx = max(node_keys.union(aux_keys)) + 1
                        children[idx] = []
                        aux_keys.add(idx)
                        for _ in range(1, map_size):
                            new_idx = max(node_keys.union(aux_keys)) + 1
                            children[idx].append(new_idx)
                            aux_keys.add(new_idx)
                        aux_map = [idx] + children[idx]
                        for node in aux_map:
                            node_inputs[node] = []
                        #add one to one connections between in_map and aux map with weight 1
                        for i in range(map_size):
                            node_inputs[aux_map[i]].append((in_map[j], 1))

                        #add one to one connections between aux map and out map with stepped weights
                        weights = calculate_weights(False,cg.weight,map_size)
                        for i in range(map_size):
                            node_inputs[out_map[j]].append((aux_map[i], weights[i]))
                else:
                    weight = cg.weight
                    for i in range(map_size):
                        node_inputs[out_map[i]].append((in_map[i], weight))

            else:
                # 1-to-all
                if not cg.uniform:
                    # Step
                    start = -cg.weight
                    end = cg.weight
                    step = (end - start) / (map_size - 1)
                    for o_n in out_map:
                        s = start
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, s))
                            s += step
                else:
                    # Uniform
                    for o_n in out_map:
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, cg.weight))

        else:
            # Map-to-isolated or isolated-to-isolated
            if not cg.uniform:
                # Step
                start = -cg.weight
                end = cg.weight
                step = (end - start) / (map_size - 1)
                for o_n in out_map:
                    s = start
                    for i_n in in_map:
                        node_inputs[o_n].append((i_n, s))
                        s += step
            else:
                # Uniform
                for o_n in out_map:
                    for i_n in in_map:
                        node_inputs[o_n].append((i_n, cg.weight))

    nodes = []

    #Sometimes the output neurons end up not having any connections during the evolutionary process. While we do not
    #desire such networks, we should still allow them to make predictions to avoid fatal errors.
    for okey in output_keys:
        if okey not in node_keys:
            node_keys.add(okey)
            std_inputs[okey] = []

    # While we cannot deduce the order of activations of the neurons due to the fact that we allow for arbitrary connection
    # schemes, we certainly want the output neurons to activate last.
    conns = {}
    for k in node_keys.union(aux_keys):
        conns[k] = []
    parents = children.keys()
    for k in conns.keys():
        if k in input_keys:
            continue
        if k not in conns.keys():
            conns[k] = []
        if k in std_inputs.keys():
            conns[k].extend([i for i, _ in std_inputs[k]])
        if k in mod_inputs.keys():
            conns[k].extend([i for i, _ in mod_inputs[k]])
    sorted_keys = order_of_activation(conns, input_keys, output_keys)

    #Edge case: when a genome has no connections, sorted keys ends up empty and crashes the program
    #If this happens, just activate the output nodes with the default activation: 0
    if not sorted_keys:
        sorted_keys = output_keys

    for node_key in sorted_keys:
        #all the children are handled with the parent
        if node_key not in parents:
            continue
        #if the node we are examining is not in our keys set then skip it. It means that it is not required for output.
        # if node_key not in node_keys:
        #     continue

        #if the node one of the originals present in the genotype, i.e. it's not one of the nodes we added for the
        #extended one to one scheme
        if node_key in genome.nodes:
            node = genome.nodes[node_key]
            node_map = [node_key] + children[node_key]
            if node.is_switch:
                # If the switch neuron does not have any incoming cnnections
                if node_key not in std_inputs.keys() and node_key not in mod_inputs.keys():
                    for n in node_map:
                        std_inputs[n] = []
                        mod_inputs[n] = []
                # if the switch neuron only has modulatory weights then we copy those weights for the standard part as well.
                # this is not the desired behaviour but it is done to avoid errors during forward pass.
                if node_key not in std_inputs.keys() and node_key in mod_inputs.keys():
                    for n in node_map:
                        std_inputs[n] = mod_inputs[n]
                if node_key not in mod_inputs.keys():
                    for n in node_map:
                        mod_inputs[n] = []
                #For the guided maps, all modulatory weights to switch neurons now weight 1/m
                if mod_inputs[node_key]:
                    for n in node_map:
                        new_w = 1 / len(std_inputs[n])
                        new_mod_w = [(inp, new_w) for inp, w in mod_inputs[n]]
                        mod_inputs[n] = new_mod_w
                for n in node_map:
                    nodes.append(SwitchNeuron(n,std_inputs[n],mod_inputs[n]))
                continue
            ###################### Switch neuron part ends here
            for n in node_map:
                if n not in std_inputs:
                    std_inputs[n] = []
                #For these guided maps, every hidden neuron that is not a switch neuron is a gating neuron

                params = {
                    'activation_function': genome_config.activation_defs.get(node.activation),
                    'integration_function': genome_config.aggregation_function_defs.get(node.aggregation),
                    'bias': node.bias,
                    'activity': 0,
                    'output': 0,
                    'weights': std_inputs[n]
                }
                nodes.append(Neuron(n, params))

        #if the node is one of those we added with the extended one to one scheme
        else:
            node_map = [node_key] + children[node_key]
            for n in node_map:
                if n not in std_inputs:
                    std_inputs[n] = []

            # Create the standard part dictionary for the neuron
            for n in node_map:
                params = {
                    'activation_function': identity,
                    'integration_function': sum,
                    'bias': 0,
                    'activity': 0,
                    'output': 0,
                    'weights': std_inputs[n]
                }
                nodes.append(Neuron(n, params))

    return SwitchNeuronNetwork(input_keys, output_keys, nodes)

def prod(l):
    i = 1
    for item in l:
        i *= item
    return i

MAP_SIZE = 3
def make_eval_fun(evaluation_func, in_proc, out_proc):

    def eval_genomes (genomes, config):
        for genome_id, genome in genomes:
            net = create(genome,config,MAP_SIZE)
            #Wrap the network around an agent
            agent = Agent(net, in_proc, out_proc)
            #Evaluate its fitness based on the function given above.
            genome.fitness = evaluation_func(agent)

    return eval_genomes

#For the guided maps encoding
#input order is: input1, reward, input2, input3
def reorder_inputs(l):
    new_l = [l[0], l[3],l[1],l[2]]
    return new_l

#A dry test run for the binary association problem to test if the above implementation works
def run(config_file, generations, binary_file, drawfile, progressfile, statsfile):

    #Configuring the agent and the evaluation function
    from eval import eval_one_to_one_3x3
    eval_func = eval_one_to_one_3x3
    #Preprocessing for inputs: none
    in_func = reorder_inputs
    from solve import convert_to_action
    out_func = convert_to_action
    #Preprocessing for output - convert float to boolean

    # Load configuration.
    config = neat.Config(GuidedMapGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = Reporters.StatReporterv2()
    p.add_reporter(stats)
    p.add_reporter(Reporters.ProgressTracker(progressfile))
    #p.add_reporter(Reporters.NetRetriever())
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(make_eval_fun(eval_func, in_func, out_func), generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create(winner, config, MAP_SIZE)
    winner_agent = Agent(winner_net,in_func, out_func)
    print("Score in task: {}".format(eval_func(winner_agent)))
    print("Input function: Reorder_inputs")
    print("Output function: convert_to_action")
    render_network.draw_net(winner_net, filename=drawfile)

    #Log the maximum fitness over generations
    from visualize import plot_stats
    plot_stats(stats,False,view=False,filename=statsfile)

    #Uncomment the following if you want to save the network in a binary file
    fp = open(binary_file,'wb')
    pickle.dump(winner_net,fp)
    fp.close()


def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config = sys.argv[1]
    generations = int(sys.argv[2])
    binary_file = sys.argv[3]
    drawfile = sys.argv[4]
    progressfile = sys.argv[5]
    statsfile = sys.argv[6]
    run(config, generations, binary_file, drawfile, progressfile, statsfile)

if __name__ == '__main__':
    main()

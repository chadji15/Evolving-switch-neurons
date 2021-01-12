import os
import pickle

import Reporters
import neat
import numpy as np
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from switch_neuron import Neuron, SwitchNeuronNetwork, SwitchNeuron, Agent
from neat.six_util import itervalues
from itertools import chain

from utilities import order_of_activation


class SwitchMapConnectionGene(BaseGene):

    #Various parameters for defining a connection.
    _gene_attributes = [BoolAttribute('one2one'), #if true then the connection scheme is one to one, else one to all
                        BoolAttribute('gaussian'),  #Gaussian or uniform distribution
                        FloatAttribute('weight'),#Weigth is used as the mean of the normal distribution for 1-to-all
                        FloatAttribute('sigma'), #The standard deviation for the gaussian
                        BoolAttribute('enabled'),
                        BoolAttribute('is_mod')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    #Define the distance between two genes
    def distance(self, other, config):
        d = abs(self.sigma - other.sigma) + abs(self.weight - other.weight) + int(self.one2one == other.one2one) \
            + int(self.gaussian == other.gaussian) + int(self.enabled == other.enabled) + int(self.is_mod == other.is_mod)
        return d * config.compatibility_weight_coefficient

class SwitchMapNodeGene(DefaultNodeGene):

    _gene_attributes = [FloatAttribute('bias'), #The bias of the neuron
                        StringAttribute('activation', options='sigmoid'), # The activation function, tunable from the config
                        StringAttribute('aggregation', options='sum'), #The aggregation function
                        BoolAttribute('is_isolated'),
                        BoolAttribute('is_switch')] #Map vs isolated neuron

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + int(self.activation == other.activation) + int(self.aggregation == other.aggregation)\
            + int(self.is_isolated == other.is_isolated) + int(self.is_switch - other.is_switch)
        return d * config.compatibility_weight_coefficient


class SwitchMapGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = SwitchMapNodeGene
        param_dict['connection_gene_type'] = SwitchMapConnectionGene
        return DefaultGenomeConfig(param_dict)


def create(genome, config, map_size):
    """ Receives a genome and returns its phenotype (a SwitchNeuronNetwork). """
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

    input_keys = genome_config.input_keys
    output_keys = genome_config.output_keys
    # Gather inputs and expressed connections.
    std_inputs = {}
    mod_inputs = {}
    children = {}
    node_keys = set(genome.nodes.keys())  # + list(genome_config.input_keys[:])

    # Here we populate the children dictionay for each unique not isolated node.
    for n in genome.nodes.keys():
        children[n] = []
        if not genome.nodes[n].is_isolated:
            for _ in range(1, map_size):
                new_idx = max(node_keys) + 1
                children[n].append(new_idx)
                node_keys.add(new_idx)
    # We don't scale the output with the map size to keep passing the parameters of the network easy.
    # This part can be revised in the future
    for n in chain(input_keys, output_keys):
        children[n] = []
    #Iterate over every connection
    for cg in itervalues(genome.connections):
        #If it's not enabled don't include it
        if not cg.enabled:
            continue

        i, o = cg.key
        #If neither node is required for output then skip the connection
        if o not in required and i not in required:
            continue

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
                # 1-to-1 mapping
                weight = 5 * cg.weight
                for i in range(map_size):
                    node_inputs[out_map[i]].append((in_map[i], weight))

            else:
                # 1-to-all
                if cg.gaussian:
                    # Gaussian
                    for o_n in out_map:
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, np.random.normal(cg.weight, cg.sigma)))
                else:
                    # Uniform
                    for o_n in out_map:
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, 5 * cg.weight))

        else:
            # Map-to-isolated or isolated-to-isolated
            if cg.gaussian:
                # Gaussian
                for o_n in out_map:
                    for i_n in in_map:
                        node_inputs[o_n].append((i_n, np.random.normal(cg.weight, cg.sigma)))
            else:
                # Uniform
                for o_n in out_map:
                    for i_n in in_map:
                        node_inputs[o_n].append((i_n, 5 * cg.weight))

    nodes = []

    #Sometimes the output neurons end up not having any connections during the evolutionary process. While we do not
    #desire such networks, we should still allow them to make predictions to avoid fatal errors.
    for okey in output_keys:
        if okey not in node_keys:
            node_keys.add(okey)
            std_inputs[okey] = []

    # While we cannot deduce the order of activations of the neurons due to the fact that we allow for arbitrary connection
    # schemes, we certainly want the output neurons to activate last.
    input_keys = genome_config.input_keys
    output_keys = genome_config.output_keys
    conns = {}
    for k in genome.nodes.keys():
        if k not in std_inputs:
            std_inputs[k] = []
            if k in children:
                for c in children[k]:
                    std_inputs[c] = []
        conns[k] = [i for i, _ in std_inputs[k]]
    sorted_keys = order_of_activation(conns, input_keys, output_keys)

    for node_key in sorted_keys:
        #if the node we are examining is not in our keys set then skip it. It means that it is not required for output.
        if node_key not in node_keys:
            continue

        node = genome.nodes[node_key]
        node_map = [node_key] + children[node_key]
        if node.is_switch:
            # If the switch neuron has both modulatory and standard weights then we can add it normally to the nodes
            # of the network.
            if node_key in std_inputs.keys() and node_key in mod_inputs.keys():
                for n in node_map:
                    nodes.append(SwitchNeuron(n,std_inputs[n],mod_inputs[n]))
                continue
            # if the switch neuron only has modulatory weights then we copy those weights for the standard part as well.
            # this is not the desired behaviour but it is done to avoid errors during forward pass.
            elif node_key not in std_inputs.keys() and node_key in mod_inputs.keys():
                for n in node_map:
                    std_inputs[n] = mod_inputs[n]
            else:
                for n in node_map:
                    std_inputs[n] = []
        else:
            for n in node_map:
                if n not in std_inputs:
                    std_inputs[n] = []

        # Create the standard part dictionary for the neuron
        params = {
            'activation_function': genome_config.activation_defs.get(node.activation),
            'integration_function': genome_config.aggregation_function_defs.get(node.aggregation),
            'bias': node.bias,
            'activity': 0,
            'output': 0,
            'weights': std_inputs[node_key]
        }
        for n in node_map:
            nodes.append(Neuron(n, params))

    return SwitchNeuronNetwork(input_keys, output_keys, nodes)

MAP_SIZE = 1
def make_eval_fun(evaluation_func, in_proc, out_proc):

    def eval_genomes (genomes, config):
        for genome_id, genome in genomes:
            net = create(genome,config,MAP_SIZE)
            #Wrap the network around an agent
            agent = Agent(net, in_proc, out_proc)
            #Evaluate its fitness based on the function given above.
            genome.fitness = evaluation_func(agent)

    return eval_genomes
#A dry test run for the xor problem to test if the above implementation works
def run(config_file):

    #Configuring the agent and the evaluation function
    from eval import eval_net_xor
    eval_func = eval_net_xor
    #Preprocessing for inputs: none
    in_func = out_func = lambda x: x
    #Preprocessing for output - convert float to boolean

    # Load configuration.
    config = neat.Config(SwitchMapGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = Reporters.StatReporterv2()
    p.add_reporter(stats)
    #p.add_reporter(Reporters.NetRetriever())
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(make_eval_fun(eval_func, in_func, out_func), 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create(winner, config, MAP_SIZE)
    winner_agent = Agent(winner_net,in_func, out_func)
    print("Score in task: {}".format(eval_func(winner_agent)))

    #Uncomment the following if you want to save the network in a binary file
    fp = open('winner_net.bin','wb')
    pickle.dump(winner_net,fp)
    fp.close()
    #visualize.draw_net(config, winner, True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-switch_maps')
    run(config_path)

if __name__ == '__main__':
    main()

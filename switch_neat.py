from __future__ import print_function

import pickle

import visualize
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from neat.six_util import itervalues
from switch_neuron import SwitchNeuron, SwitchNeuronNetwork, Neuron, Agent
import os
import neat
import Reporters
from utilities import order_of_activation

#The class for the gene representing the neurons in our network.
class SwitchNodeGene(DefaultNodeGene):

    _gene_attributes = [FloatAttribute('bias'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum'),
                        BoolAttribute('is_switch')]

    def distance(self, other, config):
        d = abs(self.bias + other.bias)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        if self.is_switch != other.is_switch:
            d =3
        return d * config.compatibility_weight_coefficient

#The gene for the connections in our network.
class SwitchConnectionGene(DefaultConnectionGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('is_mod'),
                        BoolAttribute('enabled')] #The neat package complains if this attribute is not present.

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        if self.is_mod != other.is_mod:
            d += 1
        return d * config.compatibility_weight_coefficient

#Create a switch genome class to replace the default genome class in our experiments.
class SwitchGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = SwitchNodeGene
        param_dict['connection_gene_type'] = SwitchConnectionGene
        return DefaultGenomeConfig(param_dict)

#Takes a genome and the configuration object and returns the network encoded in the genome.
def create(genome, config):
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
    input_keys = genome_config.input_keys
    output_keys = genome_config.output_keys

    #A dictionary where we keep the modulatory weights for every node
    mod_weights = {}
    #A dictionary where we keep the standard weights for every node
    std_weights = {}
    #Create a set with the keys of the nodes in the network
    keys = set()
    #Iterate over the connections
    for cg in itervalues(genome.connections):
        #if not cg.enabled:
        #    continue

        i, o = cg.key
        #If neither of the nodes in the connection are required for output then skip this connection
        if o not in required and i not in required:
            continue

        if i not in input_keys:
            keys.add(i)
        keys.add(o)
        #In this implementation, only switch neurons have a modulatory part
        if genome.nodes[o].is_switch:
            #Add the weight to the modulatory part of the o node and continue with the next connection
            if cg.is_mod:
                if o not in mod_weights.keys():
                    mod_weights[o] = [(i,cg.weight)]
                else:
                    mod_weights[o].append((i,cg.weight))
                continue
        #If the connection is not modulatory
        #Add the weight to the standard weight of the o node.
        if o not in std_weights.keys():
            std_weights[o] = [(i,cg.weight)]
        else:
            std_weights[o].append((i,cg.weight))
    #Create the array with the network's nodes
    nodes = []

    #Sometimes the output neurons end up not having any connections during the evolutionary process. While we do not
    #desire such networks, we should still allow them to make predictions to avoid fatal errors.
    for okey in output_keys:
        keys.add(okey)

    for k in keys:
        if k not in std_weights:
            std_weights[k] = []

    #While we cannot deduce the order of activations of the neurons due to the fact that we allow for arbitrary connection
    #schemes, we certainly want the output neurons to activate last.
    conns = {}
    for k in keys:
        conns[k] = [i for i, w in std_weights[k]]
    sorted_keys = order_of_activation(conns, input_keys, output_keys)

    #Create the nodes of the network based on the weights dictionaries created above and the genome.
    for node_key in sorted_keys:
        if node_key not in keys:
            continue
        node = genome.nodes[node_key]
        if node.is_switch:
            #If the switch neuron has both modulatory and standard weights then we can add it normally to the nodes
            #of the network.
            if node_key in std_weights.keys() and node_key in mod_weights.keys():
                nodes.append(SwitchNeuron(node_key, std_weights[node_key], mod_weights[node_key]))
                continue
            #if the switch neuron only has modulatory weights then we copy those weights for the standard part as well.
            #this is not the desired behaviour but it is done to avoid errors during forward pass.
            elif node_key not in std_weights.keys() and node_key in mod_weights:
                std_weights[node_key] = mod_weights[node_key]
            else:
                std_weights[node_key] = []
        else:
            if node_key not in std_weights:
                std_weights[node_key] = []

        #Create the standard part dictionary for the neuron
        params = {
            'activation_function' : genome_config.activation_defs.get(node.activation),
            'integration_function' : genome_config.aggregation_function_defs.get(node.aggregation),
            'bias' : node.bias,
            'activity' : 0,
            'output' : 0,
            'weights' : std_weights[node_key]
        }
        nodes.append(Neuron(node_key,params))

    return SwitchNeuronNetwork(input_keys,output_keys,nodes)

#This function wraps a given evaluation function to make it suitable for evaluating an array of genomes
#produced by neat and returns it.
def make_eval_fun(evaluation_func, in_proc, out_proc):

    def eval_genomes (genomes, config):
        for genome_id, genome in genomes:
            net = create(genome,config)
            #Wrap the network around an agent
            agent = Agent(net, in_proc, out_proc)
            #Evaluate its fitness based on the function given above.
            genome.fitness = evaluation_func(agent)

    return eval_genomes

#A dry test run for the one to one association task. I haven't ran it to end yet because it needs too much time as of now and at some point
#it even reaches a plateau.
def run(config_file):

    #Configuring the agent and the evaluation function
    from eval import eval_net_xor
    eval_func = eval_net_xor
    #Preprocessing for inputs: none
    in_func = out_func = lambda x: x


    #Preprocessing for outputs: one-hot max encoding.


    # Load configuration.
    config = neat.Config(SwitchGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    from utilities import heaviside
    config.genome_config.add_activation('heaviside', heaviside)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = Reporters.StatReporterv2()
    p.add_reporter(stats)
    #p.add_reporter(Reporters.NetRetriever())
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(make_eval_fun(eval_func, in_func, out_func), 2000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create(winner, config)
    winner_agent = Agent(winner_net,in_func, out_func)
    print("Score in task: {}".format(eval_func(winner_agent)))

    #for i, o in (((0,0),0), ((0,1),1), ((1,0),1), ((1,1),0)):
    #    print(f"Input: {i}, Expected: {o}, got {winner_agent.activate(i)}")
    #Uncomment the following if you want to save the network in a binary file
    fp = open('winner_net.bin','wb')
    pickle.dump(winner_net,fp)
    fp.close()
    #visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config/config-switch')
    run(config_path)

if __name__ == '__main__':
    main()

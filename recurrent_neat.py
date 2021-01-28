##################################
#Not working properly ?
#
####################################


import os
import pickle

from neat.graphs import required_for_output
from neat.six_util import itervalues
from switch_neuron import Neuron
import neat
from eval import eval_net_xor
import visualize
from utilities import order_of_activation

class RecurrentNetwork(object):
    def __init__(self, inputs, outputs, nodes):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.nodes = nodes
        self.nodes_dict = {}
        for node in nodes:
            self.nodes_dict[node.key] = node


    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = {}

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v


        for node in self.nodes:
            standard_inputs = []
            for key, weight in node.standard['weights']:
                if key in ivalues.keys():
                    val = ivalues[key]
                else:
                    val = self.nodes_dict[key].standard['output']
                standard_inputs.append(val * weight)
            # add the bias
            standard_inputs.append(node.standard['bias'])
            # Calculate the neuron's activity and output based on it's standard functions.
            node.standard['activity'] = node.standard['integration_function'](standard_inputs)
            node.standard['output'] = node.standard['activation_function'](node.standard['activity'])

        output = [self.nodes_dict[key].standard['output'] for key in self.output_nodes]
        return output

    #Takes a genome and the configuration object and returns the network encoded in the genome.
    @staticmethod
    def create(genome, config):
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
        input_keys = genome_config.input_keys
        output_keys = genome_config.output_keys


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
            if okey not in keys:
                keys.add(okey)
                std_weights[okey] = []

        #Sometimes a neuron emerges which is only connected to the output neurons with an outgoing connection
        for k in keys:
            if k not in std_weights.keys():
                std_weights[k] = []
        #Deduce the order of activation of the neurons
        conns = {}
        for node in std_weights.keys():
            conns[node] = [inp for inp, weight in std_weights[node]]
        sorted_keys = order_of_activation(conns, input_keys, output_keys)

        #Create the nodes of the network based on the weights dictionaries created above and the genome.
        for node_key in sorted_keys:
            if node_key not in keys:
                continue
            node = genome.nodes[node_key]

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

        return RecurrentNetwork(input_keys,output_keys,nodes)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = RecurrentNetwork.create(genome, config)
        genome.fitness = eval_net_xor(net)

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 2000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = RecurrentNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    pickle.dump(winner_net, open('xor_winner.bin', 'wb'))
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config/config-recurrent')
    run(config_path)
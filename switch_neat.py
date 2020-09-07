from __future__ import print_function

from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from neat.six_util import itervalues
from switch_neuron import SwitchNeuron, SwitchNeuronNetwork, Neuron
import os
import neat
import visualize

class SwitchNodeGene(DefaultNodeGene):

    _gene_attributes = [FloatAttribute('bias'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum'),
                        BoolAttribute('is_switch')]

    def distance(self, other, config):
        d = 0
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        if self.is_switch != other.is_switch:
            d =3
        return d * config.compatibility_weight_coefficient

class SwitchConnectionGene(DefaultConnectionGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('is_mod'),
                        BoolAttribute('enabled')]

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        if self.is_mod != other.is_mod:
            d += 1
        return d * config.compatibility_weight_coefficient

class SwitchGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = SwitchNodeGene
        param_dict['connection_gene_type'] = SwitchConnectionGene
        return DefaultGenomeConfig(param_dict)

#Return SwitchNeuronNetwork
def create(genome, config):
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
    input_keys = genome_config.input_keys
    output_keys = genome_config.output_keys

    mod_weights = {}
    std_weights = {}
    for cg in itervalues(genome.connections):
        if not cg.enabled:
            continue

        i, o = cg.key
        if o not in required and i not in required:
            continue

        if genome.nodes[o].is_switch:
            if cg.is_mod:
                if o not in mod_weights.keys():
                    mod_weights[o] = [(o,cg.weight)]
                else:
                    mod_weights[o].append((o,cg.weight))
                continue

        if o not in std_weights.keys():
            std_weights[o] = [(o,cg.weight)]
        else:
            std_weights[o].append((o,cg.weight))

    nodes = []
    keys = list(mod_weights.keys()) + list(std_weights.keys())
    for okey in output_keys:
        if okey not in keys:
            keys.append(okey)
            std_weights[okey] = []

    for node_key in keys:
        node = genome.nodes[node_key]
        if node.is_switch:
            if node_key in std_weights.keys() and node_key in mod_weights.keys():
                nodes.append(SwitchNeuron(node_key, std_weights[node_key], mod_weights[node_key]))
                continue
            elif node_key not in std_weights.keys():
                std_weights[node_key] = mod_weights[node_key]

        params = {
            'activation_function' : genome_config.activation_defs.get(node.activation),
            'integration_function' : genome_config.aggregation_function_defs.get(node.aggregation),
            'activity' : 0,
            'output' : 0,
            'weights' : std_weights[node_key]
        }
        nodes.append(Neuron(node_key,params))


    return SwitchNeuronNetwork(input_keys,output_keys,nodes)

from switch_env_solve import eval_one_to_one

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = create(genome,config)
        genome.fitness = eval_one_to_one(net)

def run(config_file):
    # Load configuration.
    config = neat.Config(SwitchGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create(winner, config)
    print("Score in task: {}".format(eval_one_to_one(winner_net)))

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    #
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)

def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-switch')
    run(config_path)

if __name__ == '__main__':
    main()

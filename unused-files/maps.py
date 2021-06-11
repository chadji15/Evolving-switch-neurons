
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from neat.six_util import itervalues
import numpy as np
from switch_neuron import Neuron
from utilities import order_of_activation

class MapNetwork:

    #A lot of this code is taken directly from neat.nn.RecurrentNetwork and modified because it is very close to
    #the desired outcome
    def __init__(self,inputs, outputs, nodes):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.nodes = nodes
        self.nodes_dict = {}
        for node in nodes:
            self.nodes_dict[node.key] = node


    @staticmethod
    def create(genome, config, map_size):
        """ Receives a genome and returns its phenotype (a MapNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        children = {}
        node_keys = list(genome.nodes.keys())[:] #+ list(genome_config.input_keys[:])
        # for key in genome_config.input_keys + genome_config.output_keys:
        #     children[key] = []
        #     for i in range(1,map_size):
        #         if key < 0:
        #             new_idx = min(node_keys) - 1
        #         else:
        #             new_idx = max(node_keys) + 1
        #         children[key].append(new_idx)
        #         node_keys.append(new_idx)

        for cg in itervalues(genome.connections):
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            for n in [i,o]:
                if n in children.keys():
                    continue
                children[n] = []
                if n in genome_config.input_keys or n in genome_config.output_keys:
                    continue
                if not genome.nodes[n].is_isolated:
                    for _ in range(1,map_size):
                        new_idx = max(node_keys) + 1
                        children[n].append(new_idx)
                        node_keys.append(new_idx)

            in_map = [i] + children[i]
            out_map = [o] + children[o]
            for n in out_map:
                if n not in node_inputs.keys():
                    node_inputs[n] = []

            if len(in_map) == map_size and len(out_map) == map_size:
                #Map to map connectivity
                if cg.one_to_one:
                    #1-to-1 mapping
                    weight = 5*cg.weight
                    for i_n in range(map_size):
                        node_inputs[out_map[i_n]].append((in_map[i_n], weight))

                else:
                    #1-to-all
                    if cg.is_gaussian:
                        #Gaussian
                        for o_n in out_map:
                            for i_n in in_map:
                                node_inputs[o_n].append((i_n, np.random.normal(cg.weight,cg.sigma)))
                    else:
                        #Uniform
                        for o_n in out_map:
                            for i_n in in_map:
                                node_inputs[o_n].append((i_n, 5*cg.weight))

            else:
                #Map-to-isolated or isolated-to-isolated
                if cg.is_gaussian:
                    # Gaussian
                    for o_n in out_map:
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, np.random.normal(cg.weight, cg.sigma)))
                else:
                    # Uniform
                    for o_n in out_map:
                        for i_n in in_map:\
                            node_inputs[o_n].append((i_n, 5 * cg.weight))

        input_keys = genome_config.input_keys
        output_keys = genome_config.output_keys
        conns = {}
        for k in genome.nodes.keys():
            if k not in node_inputs:
                node_inputs[k] = []
                if k in children:
                    for c in children[k]:
                        node_inputs[c] = []
            conns[k] = [i for i, _ in node_inputs[k]]
        sorted_keys = order_of_activation(conns, input_keys, output_keys)
        nodes = []
        for node_key in sorted_keys:
            if node_key not in genome.nodes.keys():
                continue
            node = genome.nodes[node_key]

            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            nodes.append(Neuron(node_key, {
                'activation_function': activation_function,
                'integration_function': aggregation_function,
                'bias': node.bias,
                'activity': 0,
                'output': 0,
                'weights': node_inputs[node_key]
            }))

            if node_key not in children:
                continue

            for n in children[node_key]:
                nodes.append(Neuron(n, {
                    'activation_function': activation_function,
                    'integration_function': aggregation_function,
                    'bias': node.bias,
                    'activity': 0,
                    'output': 0,
                    'weights': node_inputs[n]
                }))



        # for key in genome_config.input_keys:
        #     input_keys.append(key)
        #     if key in children.keys():
        #         for child in children[key]:
        #             input_keys.append(child)
        #
        # for key in genome_config.output_keys:
        #     output_keys.append(key)
        #     if key in children.keys():
        #         for child in children[key]:
        #             output_keys.append(child)

        return MapNetwork(input_keys, output_keys, nodes)

    #Perform a forward pass in the network with the given inputs. Since we are working with recurrent networks
    #and arbitrary connections, no separation of the neurons is performed between the neurons and the activation
    #sequence is determined from their order in the node_evals array.
    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = {}
        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v

        for node in self.nodes:
            standard_inputs = []
            # Collect the weighted inputs in an array
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

class MapConnectionGene(BaseGene):

    #Various parameters for defining a connection.
    _gene_attributes = [BoolAttribute('one_to_one'), #1-to-1 or 1-to-all scheme
                        BoolAttribute('is_gaussian'),  #Gaussian or uniform distribution
                        FloatAttribute('weight'),#Weigth is used as the mean of the normal distribution for 1-to-all
                        FloatAttribute('sigma'), #The standard deviation for the gaussian
                        BoolAttribute('enabled')] #<- maybe remove this trait

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    #Define the distance between two genes
    def distance(self, other, config):
        d = abs(self.sigma - other.sigma) + abs(self.weight - other.weight) + int(self.one_to_one != other.one_to_one)
        + int(self.is_gaussian != other.is_gaussian)
        return d * config.compatibility_weight_coefficient

class MapNodeGene(DefaultNodeGene):

    _gene_attributes = [FloatAttribute('bias'), #The bias of the neuron
                        StringAttribute('activation', options='sigmoid'), # The activation function, tunable from the config
                        StringAttribute('aggregation', options='sum'), #The aggregation function
                        BoolAttribute('is_isolated')] #Map vs isolated neuron

    def distance(self, other, config):
        d = 0
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        if self.is_isolated != other.is_isolated:
            d += 1
        return d * config.compatibility_weight_coefficient

class MapNode:

    def __init__(self,key, activation_function, aggregation_function, bias,is_isolated, links):
        self.key = key
        self.activation_function = activation_function
        self.aggregation_function = aggregation_function
        self.bias = bias
        self.is_isolated = is_isolated
        self.links = links
        self.activity = 0

class MapGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = MapNodeGene
        param_dict['connection_gene_type'] = MapConnectionGene
        return DefaultGenomeConfig(param_dict)


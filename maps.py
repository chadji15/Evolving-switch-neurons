
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
import numpy as np

#Somehow a genotype can map to different phenotypes?

class MapNetwork():

    #A lot of this code is taken directly from neat.nn.RecurrentNetwork and modified because it is very close to
    #the desired outcome
    def __init__(self,inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_is_isolated, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

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
                node_inputs[n] = []

            if len(in_map) == map_size and len(out_map) == map_size:
                #Map to map connectivity
                if cg.c < 0.5:
                    #1-to-1 mapping
                    weight = 5*cg.weight
                    for i in range(map_size):
                        node_inputs[out_map[i]].append((in_map[i], weight))

                else:
                    #1-to-all
                    if cg.k < 0.5:
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
                if cg.k < 0.5:
                    # Gaussian
                    for o_n in out_map:
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, np.random.normal(cg.weight, cg.sigma)))
                else:
                    # Uniform
                    for o_n in out_map:
                        for i_n in in_map:
                            node_inputs[o_n].append((i_n, 5 * cg.weight))

        node_evals = []
        for node_key, inputs in iteritems(node_inputs):
            if node_key not in genome.nodes.keys():
                continue
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.is_isolated, inputs))
            for n in children[node_key]:
                node_evals.append(
                    (n, activation_function, aggregation_function, node.bias, node.is_isolated, node_inputs[n]))

        input_keys = genome_config.input_keys
        output_keys = genome_config.output_keys

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

        return MapNetwork(input_keys, output_keys, node_evals)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, is_isolated, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias +  s)

        return [ovalues[i] for i in self.output_nodes]

class MapConnectionGene(BaseGene):

    _gene_attributes = [FloatAttribute('c'),
                        FloatAttribute('k'),
                        FloatAttribute('weight'),#Weigth is used as the mean of the normal distribution for 1-to-all
                        FloatAttribute('sigma'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.c - other.c) + abs(self.k - other.k) + abs(self.sigma - other.sigma) + abs(self.weight - other.weight)
        return d * config.compatibility_weight_coefficient

class MapNodeGene(DefaultNodeGene):
    _gene_attributes = [FloatAttribute('bias'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum'),
                        BoolAttribute('is_isolated')]

    def distance(self, other, config):
        d = 0
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        if self.is_isolated != other.is_isolated:
            d += 1
        return d * config.compatibility_weight_coefficient

class MapNode():

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


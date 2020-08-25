from collections import OrderedDict
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.genes import BaseGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output, feed_forward_layers
from neat.six_util import itervalues, iterkeys
import numpy as np

class MapNetwork():

    def __init__(self, nodes, inputs, outputs):
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def create(genome, config, map_size):
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
        nodes = OrderedDict()
        children = {}
        for n in genome.nodes.keys():
            ng = genome.nodes[n]
            children[n] = []
            activation_function = genome_config.activation_defs.get(ng.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(ng.aggregation)
            if not ng.is_isolated:
                for i in range(1,map_size):
                    new_idx = max(list(required)) + 1
                    children[n].append(new_idx)
                    required = required.union(set(new_idx))
            for idx in children[n] + [n]:
                nodes[idx] = MapNode(idx, activation_function, aggregation_function, ng.bias,ng.is_isolated, {})
        for cg in itervalues(genome.connections):
            i, o = cg.key
            if o not in required and i not in required:
                continue

            if i < 0:
                in_map = [i] + [i for _ in range(1,map_size)]
            else:
                in_map = [i] + children[i]
            out_map = [o] + children[o]
            if cg.c < 0.5:
                #1 to 1
                weight = 5 * cg.gamma
                for x in range(map_size):
                    nodes[out_map[x]].links[in_map[x]] = weight
            else:
                #Have to revisit this, what about the Gaussian
                #1 to all
                if cg.k <= 0.5:
                    weights = []
                    for _ in range(map_size**2):
                        weights.append(5.0 * cg.gamma)
                else:
                    weights = np.random.normal(cg.gamma,abs(cg.sigma),(1,map_size**2))
                for x in range(map_size):
                    for y in range(map_size):
                        nodes[out_map[x]].links[in_map[y]] = weights[x*map_size + y]

        genome_config = config.genome_config
        input_keys, output_keys = genome_config.input_keys, genome_config.output_keys
        for s in [input_keys, output_keys]:
            for key in s[:]:
                if key < 0:
                    continue
                for child in children[key]:
                    s.append(child)

        MapNetwork.organiseNodes(input_keys, nodes)
        return MapNetwork(nodes,input_keys, output_keys)

    def activate(self, inputs):
        if len(self.inputs) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.inputs), len(inputs)))

        for i, v in zip(self.inputs, inputs):
            self.nodes[i].activity = v

        for node in self.nodes:
            node.activate(self.nodes)

        return [self.nodes[idx].activity for idx in self.outputs]

    @staticmethod
    def organiseNodes(inputs,nodes):
        new_nodes = OrderedDict()
        visited = set(inputs)
        for i in inputs:
            if i < 0:
                continue
            new_nodes[i] = nodes[i]
            nodes.pop(i)
        while len(nodes) > 0:
            added = set()
            for node in itervalues(nodes):
                if all([key in new_nodes or key == node.key or key < 0 for key in node.links.keys()]):
                    new_nodes[node.key] = node
                    added.add(node.key)
            for n in added:
                nodes.pop(n)
        nodes = new_nodes
        return nodes

class MapConnectionGene(BaseGene):

    _gene_attributes = [FloatAttribute('c'),
                        FloatAttribute('k'),
                        FloatAttribute('gamma'),
                        FloatAttribute('sigma'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.c - other.c) + abs(self.k - other.k) + abs(self.gamma - other.gamma) + abs(self.sigma - other.sigma)
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

    def activate(self,nodes):
        agg = self.aggregation_function([nodes[i].activity * self.links[i] for i in self.links.keys()])
        self.activity = self.activation_function(agg)
        return self.activity

class MapGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = MapNodeGene
        param_dict['connection_gene_type'] = MapConnectionGene
        return DefaultGenomeConfig(param_dict)


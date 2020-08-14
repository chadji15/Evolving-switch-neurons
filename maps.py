from neat.attributes import FloatAttribute, BoolAttribute
from neat.genes import BaseGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from neat.six_util import itervalues, iterkeys
import numpy as np

class lPDSNeuron():

    def __init__(self,is_excitatory, time_constant, threshold, is_isolated, links):
        self.is_excitatory = is_excitatory
        self.time_constant = time_constant
        self.threshold = threshold
        self.is_isolated = is_isolated
        self.links = links
        self.activation = 0

    def aggregate(self, inputs):
        assert len(inputs) == len(self.links), "Unexpected number of inputs"
        return (inputs * np.ndarray([w for i, w in self.links])).sum()

    def activate(self, inputs):
        agg = self.aggregate(inputs)
        val = self.activation + 1/self.time_constant * (agg - self.activation + self.threshold)
        val = min (1, val)
        self.activation = max(0, val)
        return self.activation * (1 if self.is_excitatory else -1)

    # How is the distance defined?
    def distance(self, other):
        d = (self.time_constant - other.time_constant)**2 + (self.threshold - other.threshold)**2 + \
            (1 if self.is_isolated != other.is_isolated else 0) + (1 if self.is_excitatory != other.is_excitatory else 0)
        return d

class MapNetwork():

    def __init__(self, neurons, inputs, outputs):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def create(genome, config, map_size):
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
        nodes = {}
        children = {}
        for n in iterkeys(genome.nodes):
            ng = genome.nodes[n]
            children[n] = []
            if not ng.is_isolated:
                for i in range(1,map_size):
                    new_idx = max(list(required)) + 1
                    children[n].append(new_idx)
                    required = required.union(set(new_idx))
                for idx in children[n] + [n]:
                    nodes[idx] = lPDSNeuron(ng.is_excitatory,ng.time_constant,ng.threshold,ng.is_isolated, [])
        for cg in itervalues(genome.connections):
            i, o = cg.key
            if o not in required and i not in required:
                continue
            in_map = children[i] + [i]
            out_map = children[o] + [o]
            if cg.c < 0.5:
                #1 to 1
                weight = 5 * ng.Gamma
                for x in range(map_size):
                    nodes[out_map[x]].links.append((in_map[x], weight))
            else:
                #Have to revisit this, what about the Gaussian
                #1 to all
                if cg.k <= 0.5:
                    weights = np.ndarray([5 * ng.Gamma for _ in range(map_size**2)])
                else:
                    weights = np.random.normal(ng.Gamma,abs(ng.sigma),(1,map_size**2))
                for x in range(map_size):
                    for y in range(map_size):
                        nodes[out_map[x]].links.append((in_map[y], weights[x*map_size + y]))

        genome_config = config.genome_config
        input_keys, output_keys = genome_config.input_keys, genome_config.output_keys
        for s in [input_keys, output_keys]:
            for key in s[:]:
                for child in children[key]:
                    s.append(child)

        return MapNetwork(nodes,input_keys, output_keys)

    def forward(self, inputs):
        if len(self.inputs) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.inputs), len(inputs)))

        for i, v in zip(self.inputs, inputs):
            self.neurons[i].activation = v

        for node in self.neurons:
            node_inputs = [self.neurons[i].activation for i, w in node.links]
            node.activate(node_inputs)

        return [self.neurons[idx].activation for idx in self.outputs]

class lPDSNodeGene(BaseGene):
    isIsolated = BoolAttribute('is_isolated')
    isIsolated._config_items["rate_to_true_add"] = [float, 0.25]
    _gene_attributes = [BoolAttribute('is_excitatory'),
                        FloatAttribute('time_constant'),
                        FloatAttribute('threshold'),
                        isIsolated]

    def distance(self, other, config):
        s = abs(self.time_constant - other.time_constant) + abs(self.threshold - other.threshold)
        if self.is_excitatory != other.is_excitatory:
            s += 1
        if self.is_isolated != other.is_isolated:
            s += 1
        return s * config.compatibility_weight_coefficient

class lPDSConnectionGene(BaseGene):

    _gene_attributes = [FloatAttribute('c'),
                        FloatAttribute('k'),
                        FloatAttribute('Gamma'),
                        FloatAttribute('sigma')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.c - other.c) + abs(self.k - other.k) + abs(self.Gamma - other.Gamma) + abs(self.sigma - other.sigma)
        return d * config.compatibility_weight_coefficient

class lPDSGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = lPDSNodeGene
        param_dict['connection_gene_type'] = lPDSConnectionGene
        return DefaultGenomeConfig(param_dict)


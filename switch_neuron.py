from math import floor

from neat.graphs import required_for_output


def identity(activity):
    return activity

class Neuron():

    def __init__(self,key, standard_dict, modulatory_dict = None):
        #Each dictionary contains the following: activation_function, integration_function, activity, output, weights
        self.key = key
        self.standard= standard_dict
        self.modulatory = modulatory_dict
        self.has_modulatory = True
        if modulatory_dict is None:
            self.has_modulatory = False

class SwitchNeuron(Neuron):

    def __init__(self,key, std_weights, mod_weights):

        std_dict = {'activation_function': identity,
                    'integration_function': self.std_integration_function,
                    'activity': 0,
                    'output': 0,
                    'weights': std_weights}

        if not std_weights:
            mod_activity = 0
        else:
            mod_activity = 1 / (2 * len(std_weights))
        mod_dict = {
            'activation_function': identity,
            'integration_function': self.mod_integration_function,
            'activity': mod_activity,
            'output': mod_activity,
            'weights': mod_weights
        }

        super().__init__(key,std_dict, mod_dict)


    def std_integration_function(self, w_inputs):
        idx = floor(len(self.standard['weights']) * self.modulatory['output'])
        return w_inputs[idx]


    def mod_integration_function(self, w_inputs):
        self.modulatory['activity'] += sum(w_inputs)
        self.modulatory['activity'] -= floor(self.modulatory['activity'])
        return self.modulatory['activity']

class IntegratingNeuron(Neuron):

    THETA = 1
    BASELINE = 0

    def __init__(self, key, weights):
        params = {
            'activation_function' : self.tri_step,
            'integration_function' : self.perfect_integration,
            'activity' : IntegratingNeuron.BASELINE,
            'output' : 0,
            'weights': weights
        }

        super().__init__(key,params)

    def perfect_integration(self, w_inputs):
        self.standard['activity'] += sum(w_inputs)
        return self.standard['activity']

    def tri_step(self, activity):
        if activity >= IntegratingNeuron.THETA:
            self.standard['activity'] = 0
            return 1
        elif activity < -IntegratingNeuron.THETA:
            self.standard['activity'] = 0
            return -1
        return 0

class SwitchNeuronNetwork():

    def __init__(self,inputs, outputs, nodes):
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.nodes_dict = {}
        for node in nodes:
            self.nodes_dict[node.key] = node

        temp_nodes = nodes[:]
        for node in temp_nodes:
            if isinstance(node, SwitchNeuron):
                self.make_switch_module(node.key)

    def activate(self, inputs):
        assert len(self.inputs) == len(inputs), "Expected {:d} inputs, got {:d}".format(len(self.inputs), len(inputs))

        ivalues = {}
        for i, v in zip(self.inputs, inputs):
            ivalues[i] = v

        for node in self.nodes:
            if node.has_modulatory:
                mod_inputs = []
                for key, weight in node.modulatory['weights']:
                    if key in ivalues.keys():
                        val = ivalues[key]
                    else:
                        val = self.nodes_dict[key].standard['output']
                    mod_inputs.append(val*weight)
                node.modulatory['activity'] = node.modulatory['integration_function'](mod_inputs)
                node.modulatory['output'] = node.modulatory['activation_function'](node.modulatory['activity'])

            standard_inputs = []
            for key, weight in node.standard['weights']:
                if key in ivalues.keys():
                    val = ivalues[key]
                else:
                    val = self.nodes_dict[key].standard['output']
                standard_inputs.append(val * weight)
            node.standard['activity'] = node.standard['integration_function'](standard_inputs)
            node.standard['output'] = node.standard['activation_function'](node.standard['activity'])

        output =  [self.nodes_dict[key].standard['output'] for key in self.outputs]
        return output

    #Check if a switch neuron needs to be converted to a module AND converts it if so
    def make_switch_module(self, key):
        s_node = self.nodes_dict[key]
        assert isinstance(s_node, SwitchNeuron), \
            "Argument passed in is {} instead of SwitchNeuron type".format(s_node.__class__.__name__)
        out_mod = []
        for node in self.nodes:
            if node.has_modulatory:
                found = False
                for i, w in node.modulatory['weights']:
                    if i == key:
                        out_mod.append(node.key)
                        found = True
                        break
                if found:
                    break

        if not out_mod:
            return

        modulating_key = max(list(self.nodes_dict.keys())) + 1
        modulating_weights = s_node.modulatory['weights']
        modulating_dict = {
            'activation_function': identity,
            'integration_function': sum,
            'activity' : 0,
            'output' : 0,
            'weights': modulating_weights
        }
        modulating_neuron = Neuron(modulating_key,modulating_dict)
        self.nodes_dict[modulating_key] = modulating_neuron

        s_node.modulatory['weights'] = [(modulating_key, 1/len(s_node.standard['weights']))]

        integrating_key = max(list(self.nodes_dict.keys())) + 1
        integrating_weights = [(modulating_key, 1/len(s_node.standard['weights']))]
        integrating_neuron = IntegratingNeuron(integrating_key,integrating_weights)
        self.nodes_dict[integrating_key] = integrating_neuron

        idx = None
        for i, node in enumerate(self.nodes):
            if node.key == key:
                idx = i
                break

        assert idx != None, "Switch Neuron passed not found in the network nodes"

        self.nodes.insert(idx,integrating_neuron)
        self.nodes.insert(idx,modulating_neuron)

        for n in out_mod:
            node = self.nodes_dict[n]
            for ind, (i, w) in enumerate(node.modulatory['weights']):
                if i == key:
                    node.modulatory['weights'][ind] = (integrating_key, w)



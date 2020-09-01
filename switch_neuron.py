from math import floor


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

        std_dict = {'activation_function': self.identity,
                    'integration_function': self.std_integration_function,
                    'activity': 0,
                    'output': 0,
                    'weights': std_weights}

        mod_activity = 1 / (2 * len(std_weights))
        mod_dict = {
            'activation_function': self.identity,
            'integration_function': self.mod_integration_function,
            'activity': mod_activity,
            'output': mod_activity,
            'weights': mod_weights
        }

        super().__init__(key,std_dict, mod_dict)


    def std_integration_function(self, w_inputs):
        idx = floor(len(self.standard['weights']) * self.modulatory['output'])
        return w_inputs[idx]

    def identity(self, activity):
        return activity

    def mod_integration_function(self, w_inputs):
        self.modulatory['activity'] += sum(w_inputs)
        self.modulatory['activity'] -= floor(self.modulatory['activity'])
        return self.modulatory['activity']

class SwitchNeuronNetwork():

    def __init__(self,inputs, outputs, nodes):
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.nodes_dict = {}
        for node in nodes:
            self.nodes_dict[node.key] = node

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

        return [self.nodes_dict[key].standard['output'] for key in self.outputs]
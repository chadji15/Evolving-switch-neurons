class Neuron():

    def __init__(self,key, standard_dict, modulatory_dict = None):
        #Each dictionary contains the following: activation_function, integration_function, activity, output, weights
        self.key = key
        self.standard= standard_dict
        self.modulatory = modulatory_dict
        self.has_modulary = False
        if modulatory_dict is True:
            self.has_modulary = False

class SwitchNeuronNetwork():

    def __init__(self,inputs, outputs, nodes):
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.nodes_dict = {}
        for node in nodes:
            self.nodes[node.key] = node

    def activate(self, inputs):
        assert len(self.inputs) == len(inputs), "Expected {:d} inputs, got {:d}".format(len(self.inputs), inputs)

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
                node.modulatory['activity'] = node.modulatory.integration_fuction(mod_inputs)
                node.modulatory['output'] = node.modulatory.activation_function(node.modulatory['activity'])

            standard_inputs = []
            for key, weight in node.standard['weights']:
                if key in ivalues.keys():
                    val = ivalues[key]
                else:
                    val = self.nodes_dict[key].standard['output']
                standard_inputs.append(val * weight)
            node.modulatory['activity'] = node.standard.integration_fuction(standard_inputs)
            node.modulatory['output'] = node.standard.activation_function(node.modulatory['activity'])

        return [self.nodes_dict[key].standard['output'] for key in self.outputs]


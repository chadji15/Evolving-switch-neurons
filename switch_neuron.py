import math
import sys
from math import floor
from utilities import identity


#This is a wrapper class for the neural networks to allow pre-processing of the input and the output which may help the
#network
class Agent():

    def __init__(self, network, setup_inputs, prepare_outputs):
        self.network = network
        self.setup_inputs = setup_inputs
        self.prepare_outputs = prepare_outputs

    def activate(self,inputs):

        proc_inputs = self.setup_inputs(inputs)
        output = self.network.activate(proc_inputs)
        return self.prepare_outputs(output)

#This is the basis class for the neurons that we will be using for our experiments.
class Neuron():

    #Consider abolishing the dictionaries and go for a more object-oriented approach
    def __init__(self,key, standard_dict, modulatory_dict = None):
        #Each dictionary contains the following: activation_function, integration_function, activity, output, weights
        #Standard dictionary also contains bias
        self.key = key
        self.standard= standard_dict
        self.modulatory = modulatory_dict
        self.has_modulatory = (self.modulatory is not None)

class SwitchNeuron(Neuron):

    #Each of std_weights and mod_weights is an array of tuples of the form (<node>,<weight>)
    def __init__(self,key, std_weights, mod_weights):

        #The standard part of the neuron is initialized according to the definition of the switch neuron in Christodoulou's
        #and Vassiliades paper
        std_dict = {'activation_function': identity,
                    'integration_function': self.std_integration_function,
                    'bias' : 0,
                    'activity': 0,
                    'output': 0,
                    'weights': std_weights}

        #Initialize the standard activity of the neuron
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

    #The standard integration function is basically a gate that propagates one signal from its inputs based on
    #the neurons modulatory activity.
    def std_integration_function(self, w_inputs):
        idx = floor(len(self.standard['weights']) * self.modulatory['output'])
        return w_inputs[idx]

    #The modulatory function acts as perfect integrator which we make behave in a cyclical fashion in order to keep
    #its value in bounds. This cyclical behaviour is key to the switch neuron's role as it enables the  alternating (when
    #needed) propagation of the signals from the input.
    def mod_integration_function(self, w_inputs):
        self.modulatory['activity'] += sum(w_inputs)
        ###################
        #The following bounding of the activity very rarely happens and is there to combat overflows.
        #Although, these overflows shouldn't even be happening in the first place.
        if math.isnan(self.modulatory['activity']):
            self.modulatory['activity'] = 0
        self.modulatory['activity'] = max(min(self.modulatory['activity'], sys.maxsize), -sys.maxsize)
        ###################
        self.modulatory['activity'] -= floor(self.modulatory['activity'])
        return self.modulatory['activity']

#This class models the integrating neuron used in the switch module. It continually integrates the signals from
#its inputs and when the sum is over a certain threshold it fires a signal itself. It is comparable to the
#integrate-and-fire neurons .
class IntegratingNeuron(Neuron):

    THETA = 1
    BASELINE = 0

    def __init__(self, key, weights):
        params = {
            'activation_function' : self.tri_step,
            'integration_function' : self.perfect_integration,
            'bias' : 0,
            'activity' : IntegratingNeuron.BASELINE,
            'output' : 0,
            'weights': weights
        }

        super().__init__(key,params)

    def perfect_integration(self, w_inputs):
        self.standard['activity'] += sum(w_inputs)
        return self.standard['activity']

    #When the neuron's activity goes over the threshold we fire a 1, and when the neuron's activity goes below the
    # -threshold we fire a -1. The we reset the activity to 0.
    def tri_step(self, activity):
        if activity >= IntegratingNeuron.THETA:
            self.standard['activity'] = 0
            return 1
        elif activity < -IntegratingNeuron.THETA:
            self.standard['activity'] = 0
            return -1
        return 0

class SwitchNeuronNetwork():

    #inputs: an array containing the keys of the input pins
    #outputs: an array containing the keys of the output neurons
    #nodes: an array containing the neurons of the network, in the order they are supposed to fire.
    def __init__(self,inputs, outputs, nodes):
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        #We create in memory a dictionary of the nodes for faster on-demand indexing
        self.nodes_dict = {}
        for node in nodes:
            self.nodes_dict[node.key] = node

        #We convert any switch neurons that need to be converted to switch modules, i.e. the switch neurons that modulate
        #other neurons.
        temp_nodes = nodes[:]
        for node in temp_nodes:
            if isinstance(node, SwitchNeuron) and len(node.standard["weights"]) > 0:
                self.make_switch_module(node.key)

    #Perform a forward pass through the network. Since arbitrary connection schemes are allowed, the neurons are not
    #divided into layers and its order of activation is assumed based on their order of the nodes array.
    def activate(self, inputs):
        assert len(self.inputs) == len(inputs), "Expected {:d} inputs, got {:d}".format(len(self.inputs), len(inputs))

        #Store the values from the input pins in a dictionary
        ivalues = {}
        for i, v in zip(self.inputs, inputs):
            ivalues[i] = v

        for node in self.nodes:
            #First calculate the modulatory output of the neuron
            if node.has_modulatory:
                mod_inputs = []
                #Collect all the weighted inputs in array
                for key, weight in node.modulatory['weights']:
                    if key in ivalues.keys():
                        val = ivalues[key]
                    else:
                        val = self.nodes_dict[key].standard['output']
                    mod_inputs.append(val*weight)
                #Calculate the neurons modulatory activity and output based on it's functions.
                node.modulatory['activity'] = node.modulatory['integration_function'](mod_inputs)
                node.modulatory['output'] = node.modulatory['activation_function'](node.modulatory['activity'])

            standard_inputs = []
            #Collect the weighted inputs in an array
            for key, weight in node.standard['weights']:
                if key in ivalues.keys():
                    val = ivalues[key]
                else:
                    val = self.nodes_dict[key].standard['output']
                standard_inputs.append(val * weight)
            #add the bias
            standard_inputs.append(node.standard['bias'])
            #Calculate the neuron's activity and output based on it's standard functions.
            node.standard['activity'] = node.standard['integration_function'](standard_inputs)
            node.standard['output'] = node.standard['activation_function'](node.standard['activity'])

        output =  [self.nodes_dict[key].standard['output'] for key in self.outputs]
        return output

    #Check if a switch neuron needs to be converted to a module AND converts it if so. A neuron needs to be
    #converted if it modulates other neurons.
    def make_switch_module(self, key):
        #Check if the key of the switch neuron is valid.
        idx = None
        for i, node in enumerate(self.nodes):
            if node.key == key:
                idx = i
                break

        assert idx != None, "Switch Neuron passed not found in the network nodes"
        s_node = self.nodes_dict[key]
        assert isinstance(s_node, SwitchNeuron), \
            "Argument passed in is {} instead of SwitchNeuron type".format(s_node.__class__.__name__)
        out_mod = []
        #Find if this switch neurons modulates other neurons
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
        #If not, then stop the process
        if not out_mod:
            return

        #Create a unique key for the modulating neurons
        modulating_key = max(list(self.nodes_dict.keys())) + 1
        #The modulating neuron inherits the initial switch neuron's modulatory weights as its standard weights.
        modulating_weights = s_node.modulatory['weights']
        modulating_dict = {
            'activation_function': identity,
            'integration_function': sum,
            'bias' : 0,
            'activity' : 0,
            'output' : 0,
            'weights': modulating_weights
        }
        #Create the modulating neuron and add it to the network's neurons
        modulating_neuron = Neuron(modulating_key,modulating_dict)
        self.nodes_dict[modulating_key] = modulating_neuron
        #The switch neuron's is only modulated by the modulating neuron with a weight inverse to it's initial
        #input neurons.
        s_node.modulatory['weights'] = [(modulating_key, 1)]#1/len(s_node.standard['weights']))]

        #Create a unique key for the integrating neuron
        integrating_key = max(list(self.nodes_dict.keys())) + 1
        #The integrating neuron shares the connection from the modulating neuron with the switch neuron.
        integrating_weights = [(modulating_key, 1)]#1/len(s_node.standard['weights']))]
        #Create the neuron and add it to the network's neurons.
        integrating_neuron = IntegratingNeuron(integrating_key,integrating_weights)
        self.nodes_dict[integrating_key] = integrating_neuron

        self.nodes.insert(idx,integrating_neuron)
        self.nodes.insert(idx,modulating_neuron)

        #Modify the input weights of each neuron that was modulated by the switch neuron to be instead modulated
        #by the integrating neuron.
        for n in out_mod:
            node = self.nodes_dict[n]
            for ind, (i, w) in enumerate(node.modulatory['weights']):
                if i == key:
                    node.modulatory['weights'][ind] = (integrating_key, w)



import random

def identity(activity):
    return activity

def clamp(x,low,high):
    if x < low:
        return low
    if x > high:
        return high
    return x

def heaviside(x):
    if x < 0:
        return 0
    return 1

def mult(w_inputs):
    product = 1
    for w_i in w_inputs:
        product *= w_i
    return product

def shuffle_lists(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    list1, list2 = [], []
    for a, b in temp:
        list1.append(a)
        list2.append(b)
    return list1, list2

#Return the order of neuron activation in a cyclical directed graph.
#conns is a dictionary with:
#   key: the name of the node
#   value: the nodes inputs in this form: [node1, node2, node3]
def order_of_activation(conns, inputs, outputs):
    ordered_list = []
    visited = set()
    #The number of incoming connections
    #If a neuron has a recurrent connection to itself we don't count it
    magnitude = {k: len([i for i in conns[k] if i != k]) for k in conns.keys()}
    #Inputs always activate first
    for i in inputs:
        magnitude[i] = 0

    frontier = [i for i in magnitude.keys() if magnitude[i] == min(magnitude.values())]
    def activate_n(node):
        new_nodes = []
        for k in magnitude.keys():
            if k in inputs:
                continue
            if node in conns[k]:
                magnitude[k] -= 1
                new_nodes.append(k)
        return new_nodes

    while len(visited) <  len(conns.keys()) + len(inputs):
        if frontier == []:
            minmagn = min([magnitude[v] for v in magnitude.keys() if v not in visited])
            frontier = [i for i in magnitude.keys() if magnitude[i] == minmagn]
        minn = frontier[0]
        for node in frontier:
            if magnitude[node] < magnitude[minn]:
                minn = node

        if minn not in inputs:
            ordered_list.append(minn)
        frontier.remove(minn)
        visited.add(minn)
        new_nodes = activate_n(minn)
        frontier.extend([n for n in new_nodes if n not in visited and n not in frontier])

    return ordered_list
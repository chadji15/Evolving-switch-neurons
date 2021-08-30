
import copy
import warnings
from collections import namedtuple

import graphviz


def draw_map_genotype(config, genome, filename=None, fmt='svg'):
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return
    node_names = {}
    node_colors = {}
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.1',
        'width': '0.1'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, engine='neato')
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')
        node_attrs['shape'] = 'doublecircle' if genome.nodes[k].is_switch else 'circle'

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        attrs['shape'] = 'doublecircle' if genome.nodes[n].is_switch else 'circle'
        dot.node(str(n), label=str(genome.nodes[n].activation),_attributes=attrs)

    for cg in genome.connections.values():
        #if cg.input not in used_nodes or cg.output not in used_nodes:
        #    continue
        input, output = cg.key
        a = node_names.get(input, str(input))
        b = node_names.get(output, str(output))
        style = 'solid' if not cg.is_mod else 'dotted'
        #color = 'green' if cg.weight > 0 else 'red'
        color = 'black'
        if cg.one2one and cg.extended:
            color = 'blue'
        elif cg.uniform:
            color = 'yellow'
        else:
            color = 'pink'

        width = str(0.1 + abs(cg.weight / 5.0))
        fontsize = '5'
        eattrs = {'style': style, 'color': color, 'penwidth': width, 'fontsize': fontsize}
        addlab = 'one2one' if cg.one2one else 'one2all'
        label = f"{cg.weight:.3f}, {addlab}"
        dot.edge(a, b, label= label, _attributes=eattrs)
    dot.render(filename, view=False)
    return dot


def draw_genotype(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg', map_size = -1):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.1',
        'width': '0.1'}


    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, engine='neato')
    maps = {}
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)
        maps[str(k)] = [str(k)]

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')
        node_attrs['shape'] = 'doublecircle' if genome.nodes[k].is_switch else 'circle'

        dot.node(name, _attributes=node_attrs)
        maps[str(k)] = [str(k)]

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())




    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        attrs['shape'] = 'doublecircle' if genome.nodes[n].is_switch else 'circle'
        dot.node(str(n), label=str(genome.nodes[n].activation),_attributes=attrs)
        if map_size > 0:
            if not genome.nodes[n].is_isolated:
                maps[str(n)] = [str(n) + ' ' + str(i) for i in range(map_size)]
                maps[str(n)].append(str(n))
            else:
                maps[str(n)] = [str(n)]

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if not cg.is_mod else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            fontsize = '5'
            eattrs = {'style': style, 'color': color, 'penwidth': width, 'fontsize': fontsize}

            if map_size > 0:
                in_map = maps[a]
                out_map = maps[b]
                if len(in_map) == map_size and len(out_map) == map_size:
                    # Map to map connectivity
                    if cg.one2one:
                        # 1-to-1 mapping
                        for i in range(map_size):
                            dot.edge(in_map[i], out_map[i], label= f"{cg.weight:.3f}",
                                     _attributes=eattrs)

                    else:
                        # 1-to-all
                        if not cg.uniform:
                            # Step
                            start = -cg.weight
                            end = cg.weight
                            step = (end - start) / map_size
                            for o_n in out_map:
                                s = start
                                for i_n in in_map:
                                    dot.edge(i_n, o_n, label= f"{s:.3f}",
                                             _attributes=eattrs)
                                    s += step
                        else:
                            # Uniform
                            for o_n in out_map:
                                for i_n in in_map:
                                    dot.edge(i_n, o_n, label= f"{cg.weight:.3f}",
                                             _attributes=eattrs)

                else:
                    # Map-to-isolated or isolated-to-isolated
                    if not cg.uniform:
                        # Step
                        start = -cg.weight
                        end = cg.weight
                        step = (end - start) / map_size
                        for o_n in out_map:
                            s = start
                            for i_n in in_map:
                                dot.edge(i_n, o_n, label= f"{s:.3f}",
                                         _attributes=eattrs)
                                s += step
                    else:
                        # Uniform
                        for o_n in out_map:
                            for i_n in in_map:
                                dot.edge(i_n, o_n, label= f"{cg.weight:.3f}",
                                         _attributes=eattrs)

            else:
                dot.edge(a, b, label= f"{cg.weight:.3f}", _attributes=eattrs)

    dot.render(filename, view=view)

    return dot
from switch_neuron import SwitchNeuron, SwitchNeuronNetwork, Neuron


def draw_net(network, view=False, filename=None, node_names=None, node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.1',
        'width': '0.1'}


    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, engine='neato')
    maps = {}
    inputs = set()
    for k in network.inputs:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)
        maps[str(k)] = [str(k)]

    outputs = set()
    for k in network.outputs:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue'),
                      'shape': 'doublecircle' if type(network.nodes_dict[k]) is SwitchNeuron else 'circle'}

        dot.node(name, _attributes=node_attrs)
        maps[str(k)] = [str(k)]

    used_nodes = [n.key for n in network.nodes]




    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white'),
                 'shape': 'doublecircle' if type(network.nodes_dict[n]) is SwitchNeuron else 'circle'}
        #dot.node(str(n), label=str(network.nodes_dict[n].standard['activation_function'].__name__),_attributes=attrs)
        dot.node(str(n), label=str(n),_attributes=attrs)

    Conn = namedtuple('Conn',['key', 'weight','is_mod'])

    connections = []
    for neuron in network.nodes:
        for i, w in neuron.standard['weights']:
            connections.append(Conn(key = (i, neuron.key), weight = w, is_mod = False))
        if not neuron.has_modulatory:
            continue
        for i, w in neuron.modulatory['weights']:
            connections.append(Conn(key = (i, neuron.key), weight = w, is_mod = True))

    for cg in connections:
        #if cg.input not in used_nodes or cg.output not in used_nodes:
        #    continue
        input, output = cg.key
        a = node_names.get(input, str(input))
        b = node_names.get(output, str(output))
        style = 'solid' if not cg.is_mod else 'dotted'
        color = 'green' if cg.weight > 0 else 'red'
        width = str(0.1 + abs(cg.weight / 5.0))
        fontsize = '5'
        eattrs = {'style': style, 'color': color, 'penwidth': width, 'fontsize': fontsize}

        dot.edge(a, b, label= f"{cg.weight:.3f}", _attributes=eattrs)

    dot.render(filename, view=view)

    return dot
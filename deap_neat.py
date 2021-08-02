import argparse
import copy
import os
import pickle
import random
from math import sqrt

import yaml
from deap import base, creator, tools
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.base import ParallelismManager
from qdpy.containers import Grid, NoveltyArchive, OrderedSet, CVTGrid

from qdpy.plots import plotGridSubplots
from switch_maps import SwitchMapGenome
from switch_neat import SwitchNodeGene, SwitchConnectionGene, SwitchGenome, Agent
from neat import DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, Config
from solve import convert_to_action3, convert_to_action2, convert_to_action4, convert_to_direction
from eval import eval_one_to_one_3x3, eval_one_to_one_2x2, eval_one_to_one_4x4, eval_tmaze_v2
from functools import partial
from itertools import count
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import switch_neat
import switch_maps

genome_indexer = count(1)

class DeapSwitchGenome(SwitchGenome):

    def __init__(self,key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

class DeapSwitchMapGenome(SwitchMapGenome):

    def __init__(self,key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

def expr(config):
    ind = creator.Individual(next(genome_indexer))
    ind.configure_new(config)
    return ind

#evalc = count(0)
def evaluate_skinner(ind, config, eval, sat_fit, outf, createf):
    #200 episodes, interval = 40 => max fitness = 170

    in_proc = lambda x: x
    out_proc = outf
    net = createf(ind,config)
    agent = Agent(net, in_proc, out_proc)
    fitness, bd = eval(agent)
    #If the agent seems satisfactory, test it a few more times to make sure it is
    #By evaluating it a few more times and taking the minimum fitness we try to punish luck
    if fitness >= sat_fit:
        for i in range(9):
            f2, bd2 = eval(agent)

            if f2 < fitness:
                fitness = f2
                bd = copy.deepcopy(bd2)
                #If the fitness is lower than 170 then the network is not optimal and we don't care
                if f2 < sat_fit:
                    break
    # if fitness > 169:
    #     logging.debug(f"Evaluation {next(evalc)}") #debug
    #     logging.debug("=================") #debug
    #     logging.debug(f"Fitness: {fitness}\t BD: {bd}")
    #     fitness2, bd2 = eval_one_to_one_3x3(agent, 200,40, 20, True, True)
    #     logging.debug(f"Fitness2: {fitness2}\t BD: {bd2}\n")
    return [fitness,], bd

def eval_tmaze(ind, config, scenario, createf):
    #initializing the evaluator inside the function means different switch intervals for each agent
    #if the opposite is desired then the evaluator needs to be initialized outside

    in_proc = lambda x: x
    out_proc = convert_to_direction
    net = createf(ind, config)
    agent = Agent(net, in_proc, out_proc)
    fitness, bd = eval_tmaze_v2(agent, scenario)
    return [fitness,], bd

def mutate_genome(ind, config):
    ind.mutate(config)
    return ind,

def mate_genome(ind1, ind2, genfunc, config):
    child1 = genfunc(next(genome_indexer))
    child1.configure_crossover(ind1, ind2, config)
    child2 = genfunc(next(genome_indexer))
    child2.configure_crossover(ind1, ind2, config)
    return child1, child2

def neat_toolbox(conf):
    genome_conf = conf.genome_config
    toolbox = base.Toolbox()
    toolbox.register("expr", expr, config=genome_conf)
    toolbox.register("individual", toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom) #Map-Elites = random selection on a grid container
    toolbox.register("mutate", mutate_genome, config=genome_conf)
    toolbox.register("mate", mate_genome, genfunc=toolbox.individual, config=genome_conf)
    return toolbox

#Calculate the distance between two feature vectors
#We don't use the built-in eucleidian because we need to normalize the values in the 0 to 1 range
#due to the default novelty being 0.1 for some reason
#Considering a 100 length feature vector and a range of [-0.4, 1] for each one
#The final distance metric using the eucleidian distance would be in the [0,14] range
def tmaze_distance(vector1, vector2):
    dmin = 0
    dmax = 14
    d = 0
    #eucleidian
    for feat1, feat2 in zip(vector1, vector2):
        d += (feat1-feat2) ** 2
    d = sqrt(d)
    #normalize
    return (d-dmin) / (dmax-dmin)

distance_functions = {
    'tmaze' : tmaze_distance
}

problems = ['skinner2', 'skinner3', 'skinner4', 'tmaze']
def main():

    parser = argparse.ArgumentParser(description="Evolve neural networks with neat")
    parser.add_argument('-p', '--problem', help=f"Available problems: {','.join(problems)}", required=True, type=str,
                        choices=problems)
    parser.add_argument('-c', '--config', help="The NEAT configuration file", required=True, type=str)
    parser.add_argument('-hp', '--hyperparams', help="The yaml hyperparameter file", required=True, type=str)
    args=parser.parse_args()

    seed = np.random.randint(100000000)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    #Load the qd hyperparameter configuration file
    params = yaml.safe_load(open(args.hyperparams, 'r'))
    genome_type = None
    createf = None
    if params['encoding'] == 'direct':
        genome_type = DeapSwitchGenome
        createf = switch_neat.create
    elif params['encoding'] == 'map-based':
        genome_type = DeapSwitchMapGenome
        map_size = params['map_size']
        createf = partial(switch_maps.create, map_size=map_size)

    evaluate_skinner3 = partial(evaluate_skinner,
                                eval =partial(eval_one_to_one_3x3, num_episodes=params['num_episodes'],
                                              rand_iter=params['rand_iter'], snapshot_inter=params['snap_iter'], descriptor_out=True,
                                              mode='training', trials=10),
                                sat_fit = params['sat_fit'],outf = convert_to_action3)
    evaluate_skinner2 = partial(evaluate_skinner,
                                eval =partial(eval_one_to_one_2x2,  num_episodes=params['num_episodes'],
                                              rand_iter=params['rand_iter'], snapshot_inter=params['snap_iter'], descriptor_out=True,
                                              mode='training', trials=10),
                                sat_fit = params['sat_fit'],
                                outf = convert_to_action2)
    evaluate_skinner4 = partial(evaluate_skinner,
                                eval =partial(eval_one_to_one_4x4,  num_episodes=params['num_episodes'],
                                              rand_iter=params['rand_iter'], snapshot_inter=params['snap_iter'], descriptor_out=True,
                                              mode='training', trials=10),
                                sat_fit = params['sat_fit'],
                                outf = convert_to_action4)

    evalfs = {
        "tmaze" : partial(eval_tmaze,scenario=5),
        "skinner2" : evaluate_skinner2,
        "skinner3" : evaluate_skinner3,
        "skinner4" : evaluate_skinner4
    }

    evalf = partial(evalfs[args.problem], createf=createf)
    #Load the NEAT configuration file
    config_file = args.config
    conf = Config(genome_type, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)

    #Create the class for the fitness
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", DeapSwitchGenome, fitness=creator.FitnessMax, features = list)

    #Extract the variables
    nb_features = params['nb_features'] #Length of the descriptor
    bins_per_dim = params['bins_per_dim']  #Bins per dimension of the descriptor
    nb_bins = (bins_per_dim,) * nb_features #Number of bins
    fitness_domain = params['fitness_domain'] #Range of fitness
    init_batch_size = params['init_batch_size']
    batch_size = params['batch_size']
    nb_iterations = params['nb_iterations'] #Generations
    mutation_pb = params['mutation_pb'] #1 because the actual mutation probabilities are controlled through the config
    max_items_per_bin = params['max_items_per_bin'] #How many solutions in each bin
    final_filename = params['final_filename']
    features_domain =  params['features_domain'] #The range of the feature for each dimension

    verbose = True
    show_warnings = True
    log_base_path = "."


    toolbox = neat_toolbox(conf)
    toolbox.register("evaluate", evalf, config = conf)
    # Create a dict storing all relevant infos
    results_infos = {'features_domain': features_domain, 'fitness_domain': fitness_domain, 'nb_bins': nb_bins,
                     'init_batch_size': init_batch_size, 'nb_iterations': nb_iterations, 'batch_size': batch_size,
                     'mutation_pb': mutation_pb}

    fitness_weight = 1.
    if params['algorithm'] == 'NoveltySearch':
        df = distance_functions[args.problem]
        k = params['k']
        threshold_novelty = params['threshold_novelty']
        container = NoveltyArchive(k=k, threshold_novelty=threshold_novelty, fitness_domain=fitness_domain, features_domain=features_domain,
                                   storage_type=list, depot_type=OrderedSet, novelty_distance=df)
    elif params['algorithm'] == 'CVTMapElites':
        shape = params['shape']
        container = CVTGrid(shape=shape, max_items_per_bin=max_items_per_bin, grid_shape=nb_bins, nb_sampled_points=10000,
                            fitness_domain=fitness_domain, features_domain=features_domain, storage_type=OrderedSet,
                            depot_type=OrderedSet)
    elif params['algorithm'] == 'MapElites':
        container = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)
    else:
        print('Invalid algorithm')
        exit()

    with ParallelismManager("multithreading", toolbox=toolbox) as pMgr:
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, container, init_batch_size = init_batch_size,
                               batch_size = batch_size, niter = nb_iterations,
                               verbose = verbose, show_warnings = show_warnings,
                               results_infos = results_infos, log_base_path = log_base_path, final_filename = final_filename)
        # Run the illumination process !
        algo.run()

    # Print results info
    print(f"Total elapsed: {algo.total_elapsed}\n")
    print(container.summary())
    print("Best ever fitness: ", container.best_fitness)
    print("Best ever ind: ", container.best)

    #plot_path =  os.path.join(log_base_path, "performancesGrid.pdf")
    #ValueError: plotGridSubplots only supports up to 4 dimensions.
    # plotGridSubplots(grid.quality_array[..., 0], plot_path, plt.get_cmap("nipy_spectral_r"), features_domain, fitness_domain[0], nbTicks=None)
    # print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
    # print("All results are available in the '%s' pickle file." % algo.final_filename)

if __name__ == "__main__":
    main()
import argparse
import copy
import os
import pickle
import random
from math import sqrt

from deap import base, creator, tools
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.base import ParallelismManager
from qdpy.containers import Grid, NoveltyArchive, OrderedSet

from qdpy.plots import plotGridSubplots
from switch_neat import SwitchNodeGene, SwitchConnectionGene, SwitchGenome, create, Agent
from neat import DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, Config
from solve import convert_to_action3, convert_to_action2, convert_to_action4, convert_to_direction
from eval import eval_one_to_one_3x3, eval_one_to_one_2x2, eval_one_to_one_4x4, eval_tmaze_v2
from functools import partial
from itertools import count
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
#import logging
#logging.basicConfig(filename="skinner.log", level=logging.DEBUG, format="%(message)s")


genome_indexer = count(1)

class DeapSwitchGenome(SwitchGenome):

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

evalc = count(0)
def evaluate_skinner(ind, config, eval, sat_fit, outf):
    #200 episodes, interval = 40 => max fitness = 170

    in_proc = lambda x: x
    out_proc = outf
    net = create(ind,config)
    agent = Agent(net, in_proc, out_proc)
    fitness, bd = eval(agent)
    #If the agent seems satisfactory, test it a few more times to make sure it is
    #By evaluating it a few more times and taking the minimum fitness we try to punish luck
    if fitness > sat_fit:
        for i in range(99):
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

evaluate_skinner3 = partial(evaluate_skinner,
                            eval =partial(eval_one_to_one_3x3, num_episodes=350, rand_iter=50, snapshot_inter=25, descriptor_out=True),
                            sat_fit = 307,
                            outf = convert_to_action3)
evaluate_skinner2 = partial(evaluate_skinner,
                            eval =partial(eval_one_to_one_2x2, num_episodes=50, rand_iter=10, snapshot_inter=5, descriptor_out=True),
                            sat_fit = 39,
                            outf = convert_to_action2)
evaluate_skinner4 = partial(evaluate_skinner,
                            eval =partial(eval_one_to_one_4x4, num_episodes=200, rand_iter=40, snapshot_inter=20, descriptor_out=True),
                            sat_fit = 139,
                            outf = convert_to_action4)

def eval_tmaze(ind, config, scenario):
    #initializing the evaluator inside the function means different switch intervals for each agent
    #if the opposite is desired then the evaluator needs to be initialized outside

    in_proc = lambda x: x
    out_proc = convert_to_direction
    net = create(ind, config)
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

skinner2_params = {
    'nb_features' : 9, #Length of the descriptor
    'bins_per_dim' : 2,  #Bins per dimension of the descriptor
    'fitness_domain' : [(0., 50.)], #Range of fitness
    'init_batch_size' : 10000,
    'batch_size' : 2000,
    'nb_iterations' : 50 ,#Generations
    'mutation_pb' : 1., #1 because the actual mutation probabilities are controlled through the config
    'max_items_per_bin' : 1, #How many solutions in each bin
}
skinner2_params['features_domain'] = [(0.,1.)] * skinner2_params['nb_features']

skinner3_params = {
    'nb_features' : 13, #Length of the descriptor
    'bins_per_dim' : 2,  #Bins per dimension of the descriptor
    'fitness_domain' : [(0., 350.)], #Range of fitness
    'init_batch_size' : 10000,
    'batch_size' : 128,
    'nb_iterations' : 4000 ,#Generations
    'mutation_pb' : 1., #1 because the actual mutation probabilities are controlled through the config
    'max_items_per_bin' : 1, #How many solutions in each bin
}
skinner3_params['features_domain'] = [(0.,1.)] * skinner3_params['nb_features']

skinner4_params = {
    'nb_features' : 9, #Length of the descriptor
    'bins_per_dim' : 2,  #Bins per dimension of the descriptor
    'fitness_domain' : [(0., 200.)], #Range of fitness
    'init_batch_size' : 10000,
    'batch_size' : 10000,
    'nb_iterations' : 100 ,#Generations
    'mutation_pb' : 1., #1 because the actual mutation probabilities are controlled through the config
    'max_items_per_bin' : 1, #How many solutions in each bin
}
skinner4_params['features_domain'] = [(0.,1.)] * skinner4_params['nb_features']

tmaze_parameters = {
    'k' : 15,
    'threshold_novelty' : 0.05,
    'fitness_domain' : [(-41., 101.)], #Range of fitness
    'nb_features' : 100,
    'bins_per_dim': 1,
    'init_batch_size' : 10000,
    'batch_size' : 2000,
    'nb_iterations' : 50,
    'mutation_pb' : 1,
    'max_items_per_bin' : 1,
}
tmaze_parameters['features_domain'] = [(-0.4, 1.)] * tmaze_parameters['nb_features']

parameters ={
    "skinner2" : skinner2_params,
    'skinner3' : skinner3_params,
    "skinner4" : skinner4_params,
    "tmaze" : tmaze_parameters
}

configs = {
    "skinner2": "config/deap-skinner2",
    "skinner3": "config/deap-skinner3",
    "skinner4": "config/deap-skinner4",
    "tmaze" : "config/deap-tmaze"
}

problems = {
    "tmaze" : partial(eval_tmaze,scenario=5),
    "skinner2" : evaluate_skinner2,
    "skinner3" : evaluate_skinner3,
    "skinner4" : evaluate_skinner4
}

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


def main():

    parser = argparse.ArgumentParser(description="Evolve neural networks with neat")
    parser.add_argument('-p', '--problem', help=f"Available problems: {','.join(problems.keys())}", required=True, type=str,
                        choices=problems.keys())
    args=parser.parse_args()

    seed = np.random.randint(100000000)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    config_file = configs[args.problem]
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", DeapSwitchGenome, fitness=creator.FitnessMax, features = list)

    params = parameters[args.problem]

    nb_features = params['nb_features'] #Length of the descriptor
    bins_per_dim = params['bins_per_dim']  #Bins per dimension of the descriptor
    nb_bins = (bins_per_dim,) * nb_features #Number of bins

    features_domain =  params['features_domain']#The range of the feature for each dimension
    fitness_domain = params['fitness_domain'] #Range of fitness
    init_batch_size = params['init_batch_size']
    batch_size = params['batch_size']
    nb_iterations = params['nb_iterations'] #Generations
    mutation_pb = params['mutation_pb'] #1 because the actual mutation probabilities are controlled through the config
    max_items_per_bin = params['max_items_per_bin'] #How many solutions in each bin

    verbose = True
    show_warnings = True
    log_base_path = "."


    toolbox = neat_toolbox(conf)
    toolbox.register("evaluate", problems[args.problem], config = conf)
    # Create a dict storing all relevant infos
    results_infos = {'features_domain': features_domain, 'fitness_domain': fitness_domain, 'nb_bins': nb_bins,
                     'init_batch_size': init_batch_size, 'nb_iterations': nb_iterations, 'batch_size': batch_size,
                     'mutation_pb': mutation_pb}

    fitness_weight = 1.
    if args.problem == 'tmaze':
        k = params['k']
        threshold_novelty = params['threshold_novelty']
        container = NoveltyArchive(k=k, threshold_novelty=threshold_novelty, fitness_domain=fitness_domain, features_domain=features_domain,
                                   storage_type=list, depot_type=OrderedSet, novelty_distance=tmaze_distance)
    else:
        container = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)

    with ParallelismManager("multithreading", toolbox=toolbox) as pMgr:
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, container, init_batch_size = init_batch_size,
                               batch_size = batch_size, niter = nb_iterations,
                               verbose = verbose, show_warnings = show_warnings,
                               results_infos = results_infos, log_base_path = log_base_path)
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
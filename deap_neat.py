import os
import random
from deap import base, creator, tools
from qdpy.algorithms.deap import DEAPQDAlgorithm
from qdpy.base import ParallelismManager
from qdpy.containers import Grid
from qdpy.plots import plotGridSubplots
from switch_neat import SwitchNodeGene, SwitchConnectionGene, SwitchGenome, create, Agent
from neat import DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, Config
from solve import convert_to_action
from eval import eval_one_to_one_3x3
from functools import partial
from itertools import count
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm

#200 episodes, interval = 40 => max fitness = 170
eval_f = partial(eval_one_to_one_3x3, num_episodes=200, rand_iter=40, snapshot_inter=20, descriptor_out=True)
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

def evaluate_skinner(ind, config):
    in_proc = lambda x: x
    out_proc = convert_to_action
    net = create(ind,config)
    agent = Agent(net, in_proc, out_proc)
    fitness, bd = eval_f(agent)
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


def main():

    seed = np.random.randint(100000000)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    config_file = "config/binary-deap"
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)
    genome_conf = conf.genome_config

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", DeapSwitchGenome, fitness=creator.FitnessMax, features = list)

    nb_features = 9 #Length of the descriptor
    bins_per_dim = 2  #Bins per dimension of the descriptor
    nb_bins = (bins_per_dim,) * nb_features #Number of bins
    features_domain = [(0., 1.)] * nb_features #The range of the feature for each dimension
    fitness_domain = [(0., 200.)] #Range of fitness
    init_batch_size = 5000
    batch_size = 1000
    nb_iterations = 20 #Generations
    mutation_pb = 1. #1 because the actual mutation probabilities are controlled through the config
    max_items_per_bin = 2 #How many solutions in each bin
    verbose = True
    show_warnings = True
    log_base_path = "."


    toolbox = base.Toolbox()
    toolbox.register("expr", expr, config=genome_conf)
    toolbox.register("individual", toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_skinner, config = conf)
    toolbox.register("select", tools.selRandom) #Map-Elites = random selection on a grid container
    toolbox.register("mutate", mutate_genome, config=genome_conf)
    toolbox.register("mate", mate_genome, genfunc=toolbox.individual, config=genome_conf)

    # Create a dict storing all relevant infos
    results_infos = {'features_domain': features_domain, 'fitness_domain': fitness_domain, 'nb_bins': nb_bins,
                     'init_batch_size': init_batch_size, 'nb_iterations': nb_iterations, 'batch_size': batch_size,
                     'mutation_pb': mutation_pb}

    fitness_weight = 1.
    grid = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)

    with ParallelismManager("multithreading", toolbox=toolbox) as pMgr:
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, grid, init_batch_size = init_batch_size,
                               batch_size = batch_size, niter = nb_iterations,
                               verbose = verbose, show_warnings = show_warnings,
                               results_infos = results_infos, log_base_path = log_base_path)
        # Run the illumination process !
        algo.run()

    # Print results info
    print(f"Total elapsed: {algo.total_elapsed}\n")
    print(grid.summary())
    print("Best ever fitness: ", grid.best_fitness)
    print("Best ever ind: ", grid.best)

    #plot_path =  os.path.join(log_base_path, "performancesGrid.pdf")
    #ValueError: plotGridSubplots only supports up to 4 dimensions.
    # plotGridSubplots(grid.quality_array[..., 0], plot_path, plt.get_cmap("nipy_spectral_r"), features_domain, fitness_domain[0], nbTicks=None)
    # print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
    # print("All results are available in the '%s' pickle file." % algo.final_filename)

if __name__ == "__main__":
    main()
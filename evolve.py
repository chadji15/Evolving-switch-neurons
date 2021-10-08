###############3
# This is the main file used for running experiments with the NEAT algorithm and switch neurons.
# It uses the neat-python package and tries to squeeze the utility of all the relevant files into one.
# It takes from the command line the parameters of the experiment and prints on the standard output the progress of
# the experiment. It also produces a binary file with the winning genome, a file with the maximum fitness
# progression and a visualization of the network if told to.

import pickle
import argparse
from functools import partial
from eval import eval_one_to_one_3x3, eval_net_xor, TmazeNovelty, \
    DoubleTmazeNovelty, HomingTmazeNovelty, TmazeEvaluator, DoubleTmazeEvaluator, HomingTmazeEvaluator, \
    AssociationNovelty
import switch_neat
from maps import MapNetwork, MapGenome
import switch_maps
from recurrent_neat import RecurrentNetwork
from solve import convert_to_action, convert_to_direction
import neat
import Reporters
from utilities import heaviside
from switch_neuron import Agent
import render_network
from neat.statistics import StatisticsReporter

def identity(x):
    return x

def main():
    schemes = {'switch':switch_neat.create , 'recurrent': RecurrentNetwork.create,
               'switch_maps' : switch_maps.create}
    problems = {'xor' : eval_net_xor, 'binary_association':eval_one_to_one_3x3, 'tmaze': TmazeEvaluator().eval_tmaze,
                'double_tmaze':
                DoubleTmazeEvaluator.eval_double_tmaze, 'homing_tmaze': HomingTmazeEvaluator().eval_tmaze_homing}

    domain_constant = {'tmaze': 2, 'double_tmaze': 4, 'homing_tmaze':2}

    parser = argparse.ArgumentParser(description="Evolve neural networks with neat")
    parser.add_argument('-s', '--scheme', help=f"Choose between the available schemes: {','.join(schemes.keys())}",
                         type=str, required=True, choices=schemes.keys())
    parser.add_argument('-c', '--config', help="The config file", required=True, type=str)
    parser.add_argument('-g', '--generations', help="The number of generations", type=int)
    parser.add_argument('-p', '--problem', help=f"Available problems: {','.join(problems.keys())}", required=True, type=str,
                        choices=problems.keys())
    parser.add_argument('--map_size', help="Set the map size for the relevant schemes", type=int)
    parser.add_argument('--dump', help="Dump the network in a binary file", type=str)
    parser.add_argument('--num_episodes', help="Number of episodes for tmaze/binary_association", type=int)
    parser.add_argument('--switch_interval', help="Interval of episodes for "
                                                  "shuffling the associations", type=int )
    parser.add_argument('--novelty', help='Use the novelty metric instead of the fitness function', action="store_true")
    parser.add_argument('--threshold', help='Threshold for a new genome to enter the archive', type=float, default=1)
    parser.add_argument("--snap_inter", help="Snapshot interval for association problem novelty search", type = int)
    parser.add_argument("--draw", help='Render the network to a picture. Provide the name of the picture.', type=str, default=None)
    parser.add_argument("--log", help="Log the max fitness per generation to text file. (Append)", type=str, default=None)
    args=parser.parse_args()

    eval_f = problems[args.problem]
    in_f = identity
    out_f = identity
    genome = neat.DefaultGenome
    evaluator = None
    #Configure genome based on the encoding scheme and neurons used
    if args.scheme == 'switch':
        genome = switch_neat.SwitchGenome
    elif args.scheme == 'switch_maps':
        genome = switch_maps.SwitchMapGenome

    #Configure the pre-processing and post-processing functions based on
    #the environment
    if args.problem == 'binary_association':
        out_f = convert_to_action
    elif args.problem in ['tmaze', 'double_tmaze', 'homing_tmaze'] :
        out_f = convert_to_direction

    #If we use the map-based encoding scheme add the map size parameter to the function
    #responsible for creating the network from the genotype.
    create_f = None
    if args.map_size is not None and (args.scheme in ['maps', 'switch_maps']):
        create_f = partial(schemes[args.scheme], map_size=args.map_size)
    else:
        create_f = schemes[args.scheme]

    num_episodes = 100
    s_inter = 20
    if args.num_episodes is not None:
        num_episodes = args.num_episodes
    if args.switch_interval is not None:
        s_inter = args.switch_interval
    #If the problem is the t-maze task, use the extra parameters episodes and switch interval
    if args.problem == 'tmaze':
        if args.novelty:
            evaluator = TmazeNovelty(num_episodes,samples=4, threshold=args.threshold)
            eval_f = evaluator.eval
        else:
            evaluator = TmazeEvaluator(num_episodes, samples=4)
            eval_f = evaluator.eval_tmaze
    elif args.problem == 'double_tmaze':
        if args.novelty:
            evaluator = DoubleTmazeNovelty(num_episodes,samples=4, threshold=args.threshold)
            eval_f = evaluator.eval
        else:
            evaluator = DoubleTmazeEvaluator(num_episodes, samples=4)
            eval_f = evaluator.eval_double_tmaze
    elif args.problem == 'homing_tmaze':
        if args.novelty:
            evaluator = HomingTmazeNovelty(num_episodes,samples=4, threshold=args.threshold)
            eval_f = evaluator.eval
        else:
            evaluator = HomingTmazeEvaluator(num_episodes, samples=4)
            eval_f = evaluator.eval_tmaze_homing
    elif args.problem == 'binary_association':
        if args.novelty:
            evaluator = AssociationNovelty(num_episodes,rand_iter=args.switch_interval,snapshot_inter=args.snap_inter,
                                           threshold=args.threshold)
            eval_f = evaluator.eval
        else:
            eval_f = partial (eval_one_to_one_3x3,num_episodes=num_episodes, rand_iter=s_inter)

    def make_eval_fun(evaluation_func, in_proc, out_proc, evaluator=None):

        def eval_genomes (genomes, config):
            for genome_id, genome in genomes:
                net = create_f(genome,config)
                #Wrap the network around an agent
                agent = Agent(net, in_proc, out_proc)
                #Evaluate its fitness based on the function given above.
                genome.fitness = evaluation_func(agent)

        def eval_genomes_novelty(genomes, config):
            #Re-evaluate the archive
            evaluator.reevaluate_archive()
            for genome_id, genome in genomes:
                net = create_f(genome,config)
                #Wrap the network around an agent
                agent = Agent(net, in_proc, out_proc)
                #Evaluate its fitness based on the function given above.
                genome.fitness = evaluation_func(genome_id, genome, agent)

        if args.novelty:
            return eval_genomes_novelty
        else:
            return eval_genomes

    config = neat.Config(genome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.config)
    config.genome_config.add_activation('heaviside', heaviside)


    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    #Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = StatisticsReporter()
    p.add_reporter(stats)
    if args.problem in  ['double_tmaze', 'tmaze', 'homing_tmaze']:
        if args.novelty:
            mutator = Reporters.EvaluatorMutator(evaluator.evaluator)
        else:
            mutator = Reporters.EvaluatorMutator(evaluator)
        p.add_reporter(mutator)

    # Run for up to ... generations.
    if args.novelty:
        f = make_eval_fun(eval_f, in_f, out_f, evaluator)
    else:
        f = make_eval_fun(eval_f, in_f, out_f)
    winner = p.run(f, args.generations)

    #If we are using the novelty metric get the winner from the archive
    if args.novelty:
        winnerid = evaluator.get_best_id()
        winner = evaluator.archive[winnerid]['genome']
        winner_agent = evaluator.archive[winnerid]['agent']
    else:
        print('\nBest genome:\n{!s}'.format(winner))
        winner_net = create_f(winner, config)
        winner_agent = Agent(winner_net,in_f, out_f)


    if args.novelty:
        if args.problem == 'binary_association':
            score = evaluator.eval_func(winner_agent)[0]
        else:
            score = evaluator.evaluator.eval_func(winner_agent)[0]
    else:
        score = eval_f(winner_agent)
    print("Score in task: {}".format(score))
    if args.draw is not None:
        if args.scheme in ['maps', 'switch_maps']:
            map_size = args.map_size
        else:
            map_size = -1
        render_network.draw_genotype(config, winner,filename = args.draw, map_size=map_size)

    if args.log is not None:
        fp = open(args.log, 'a')
        best_fitness = [str(c.fitness) for c in stats.most_fit_genomes]
        mfs = ' '.join(best_fitness)
        fp.write(mfs)
        fp.write("\n")
        fp.close()

    if args.dump is not None:
        fp = open(args.dump,'wb')
        pickle.dump(winner,fp)
        fp.close()
        print(f'Agent pre-processing function: {in_f.__name__}')
        print(f'Agent post-processing function: {out_f.__name__}')

if __name__ == "__main__":
    main()

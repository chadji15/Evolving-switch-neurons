import pickle
import argparse
from eval import eval_one_to_one_3x3, eval_tmaze, eval_net_xor
import switch_neat
from maps import MapNetwork, MapGenome
import switch_maps
from recurrent_neat import RecurrentNetwork
from solve import convert_to_action, convert_to_direction
import neat
import Reporters
from utilities import heaviside
from switch_neuron import Agent

def identity(x):
    return x

def main():
    schemes = {'switch':switch_neat.create , 'maps' : MapNetwork.create, 'recurrent': RecurrentNetwork.create,
               'switch_maps' : switch_maps.create}
    problems = {'xor' : eval_net_xor, 'binary_association':eval_one_to_one_3x3, 't-maze':eval_tmaze}

    parser = argparse.ArgumentParser(description="Evolve neural networks with neat")
    parser.add_argument('-s', '--scheme', help=f"Choose between the available schemes: {','.join(schemes.keys())}",
                         type=str, required=True, choices=schemes.keys())
    parser.add_argument('-c', '--config', help="The config file", required=True, type=str)
    parser.add_argument('-g', '--generations', help="The number of generations", type=int)
    parser.add_argument('-p', '--problem', help=f"Available problems: {','.join(problems.keys())}", required=True, type=str,
                        choices=problems.keys())
    parser.add_argument('--map_size', help="Set the map size for the relevant schemes", type=int)
    parser.add_argument('--dump', help="Dump the network in a binary file", type=str)
    parser.add_argument('--num_episodes', help="Number of episodes for tmaze", type=int)
    parser.add_argument('--switch_interval', help="Interval of episodes for switching the position of the high reward"
                                                  "for the t-maze task", type=int )
    args=parser.parse_args()

    eval_f = problems[args.problem]
    in_f = identity
    out_f = identity
    genome = neat.DefaultGenome

    #Configure genome based on the encoding scheme and neurons used
    if args.scheme == 'switch':
        genome = switch_neat.SwitchGenome
    elif args.scheme == 'maps':
        genome = MapGenome
    elif args.scheme == 'switch_maps':
        genome = switch_maps.SwitchMapGenome

    #Configure the pre-processing and post-processing functions based on
    #the environment
    if args.problem == 'binary_association':
        out_f = convert_to_action
    elif args.problem == 't-maze':
        out_f = convert_to_direction

    #If we use the map-based encoding scheme add the map size parameter to the function
    #responsible for creating the network from the genotype.
    create_f = None
    if args.map_size is not None and (args.scheme in ['maps', 'switch_maps']):
        def create_func(genome, config):
            return schemes[args.scheme](genome,config, args.map_size)

        create_f = create_func
    else:
        create_f = schemes[args.scheme]

    num_episodes = 100
    s_inter = 20
    #If the problem is the t-maze task, use the extra parameters episodes and switch interval
    if args.problem == 't-maze':
        if args.num_episodes is not None:
            num_episodes = args.num_episodes
        if args.switch_interval is not None:
            s_inter = args.switch_interval
        eval_f = lambda agent: eval_tmaze(agent, num_episodes, s_inter)

    def make_eval_fun(evaluation_func, in_proc, out_proc):

        def eval_genomes (genomes, config):
            for genome_id, genome in genomes:
                net = create_f(genome,config)
                #Wrap the network around an agent
                agent = Agent(net, in_proc, out_proc)
                #Evaluate its fitness based on the function given above.
                genome.fitness = evaluation_func(agent)

        return eval_genomes

    config = neat.Config(genome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.config)
    config.genome_config.add_activation('heaviside', heaviside)


    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    #Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = Reporters.StatReporterv2()
    p.add_reporter(stats)

    # Run for up to ... generations.
    winner = p.run(make_eval_fun(eval_f, in_f, out_f), args.generations)
    print('\nBest genome:\n{!s}'.format(winner))
    winner_net = create_f(winner, config)
    winner_agent = Agent(winner_net,in_f, out_f)
    print("Score in task: {}".format(eval_f(winner_agent)))

    if args.dump is not None:
        fp = open(args.dump,'wb')
        pickle.dump(winner_net,fp)
        fp.close()
        print(f'Agent pre-processing function: {in_f.__name__}')
        print(f'Agent post-processing function: {out_f.__name__}')

if __name__ == "__main__":
    main()

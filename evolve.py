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
    parser.add_argument('--map_size', help="Set the map size for the relevant schemes", type=int, default=1)
    parser.add_argument('--dump', help="Dump the network in a binary file", type=str)
    args=parser.parse_args()
    print(args)
    eval_f = problems[args.problem]
    in_f = identity
    out_f = identity
    genome = neat.DefaultGenome

    if args.scheme == 'switch':
        genome = switch_neat.SwitchGenome
    elif args.scheme == 'maps':
        genome = MapGenome
    elif args.scheme == 'switch_maps':
        genome = switch_maps.SwitchMapGenome

    if args.problem == 'binary_association':
        out_f = convert_to_action
    elif args.problem == 't-maze':
        out_f = convert_to_direction

    create_f = None
    if args.map_size is not None:
        def create_func(genome, config):
            schemes[args.scheme](genome,config, args.map_size)

        create_f = create_func
    else:
        create_f = schemes[args.scheme]

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

"""
2-input XOR example -- this is most likely the simplest possible example.
Mainly for testing purposes.
"""

from __future__ import print_function
import os
import neat
import visualize
import maps
import pickle
from utilities import shuffle_lists
# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

TRIALS = 10
MAP_SIZE = 1
########################################
#Randomize inputs and outputs
#It appears that winning neural networks learn immitate the cyclical pattern through recurrent connections
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        sum = 0
        for i in range(TRIALS):
            fitness = 4
            net = maps.MapNetwork.create(genome, config, MAP_SIZE)
            trial_in, trial_out = shuffle_lists(xor_inputs, xor_outputs)
            for xi, xo in zip(trial_in, trial_out):
                output = net.activate(xi)
                fitness -= abs(output[0] - xo[0])
            sum += fitness
        genome.fitness = sum/TRIALS

def run(config_file):
    # Load configuration.
    config = neat.Config(maps.MapGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(100))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = maps.MapNetwork.create(winner, config, MAP_SIZE)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    #The following visualization is not accurate
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    pickle.dump(winner_net, open("mapsw.bin", "wb"))

def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-maps')
    run(config_path)

if __name__ == '__main__':
    main()


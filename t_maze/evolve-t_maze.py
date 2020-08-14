import neat
import t_maze
import gym
import os

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_network(net)

def eval_network(network):

    env = gym.make('MiniGrid-TMaze-v0').unwrapped
    sum = 0
    num_episodes = 2000
    for i_episode in range(num_episodes):
        if i_episode % 500 == 0:
            reward_pos = num_episodes // 500 % 2
            observation = env.reset(reward_pos)
        else:
            observation = env.reset()
        done = False
        while not done:
            output = network.activate(observation)[0]
            action = float_to_action(output)
            observation, reward, done, info = env.step(action)

        sum += reward
    env.close()
    return sum/num_episodes

def float_to_action(output):

    if output < -0.33:
        return t_maze.TMazeEnv.Actions.left
    if output > 0.33:
        return t_maze.TMazeEnv.Actions.right
    return t_maze.TMazeEnv.Actions.forward

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nAverage reward per episode:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    print(eval_network())

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
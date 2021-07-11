import pickle
from eval import eval_one_to_one_3x3
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from solve import convert_to_action2, convert_to_action3, convert_to_action4, convert_to_direction
from switch_neat import Agent, create
from deap import creator, base
from deap_neat import DeapSwitchGenome

def get_grid():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", DeapSwitchGenome, fitness=creator.FitnessMax, features = list)
    res = pickle.load(open("final.p", "rb"))
    grid = res['container']
    return grid

def get_agent_onf(size, config_file):
    grid = get_grid()
    ind = grid.solutions[(1,0,1,0,1,0,1,0,1)][0]
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                    DefaultSpeciesSet, DefaultStagnation,
                    config_file)
    net = create(ind, conf)
    if size == 2:
        outf = convert_to_action2
    elif size == 3:
        outf = convert_to_action3
    elif size == 4:
        outf = convert_to_action4
    agent = Agent(net, lambda x: x, outf)
    return agent

def get_best_agent(size, config_file):
    grid = get_grid()
    ind = grid.best
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)
    net = create(ind, conf)
    if size == 2:
        outf = convert_to_action2
    elif size == 3:
        outf = convert_to_action3
    elif size == 4:
        outf = convert_to_action4
    agent = Agent(net, lambda x: x, outf)
    return agent

def get_best_tmaze():
    grid = get_grid()
    ind = grid.best
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  "config/deap-tmaze")
    net = create(ind, conf)
    outf = convert_to_direction
    agent = Agent(net, lambda x: x, outf)
    return agent


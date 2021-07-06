import pickle
from eval import eval_one_to_one_3x3
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from solve import convert_to_action
from switch_neat import Agent, create
from deap import creator, base
from deap_neat import DeapSwitchGenome

def test_network():
    net = pickle.load(open("winner_net.bin", "rb"))
    agent = Agent(net, lambda x: x, convert_to_action)
    c =  1
    while True:
        c+=1
        print("===================")
        print(f"Evaluation {c}:")
        print("===================")
        fitness, bd = eval_one_to_one_3x3(agent, 200 ,40, 20, True)

        print(fitness, bd)
        if fitness > 169:
            break

def get_grid():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", DeapSwitchGenome, fitness=creator.FitnessMax, features = list)
    res = pickle.load(open("final.p", "rb"))
    grid = res['container']
    return grid

def get_agent_onf(f):
    grid = get_grid()
    ind = grid.solutions[f][0]
    config_file = "config/binary-deap"
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                    DefaultSpeciesSet, DefaultStagnation,
                    config_file)
    net = create(ind, conf)
    agent = Agent(net, lambda x: x, convert_to_action)
    return agent

def get_agent_osc():
    return get_agent_onf((1,0,1,0,1,0,1,0,1))

def get_best_agent():
    grid = get_grid()
    ind = grid.best
    config_file = "config/binary-deap"
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)
    net = create(ind, conf)
    agent = Agent(net, lambda x: x, convert_to_action)
    return agent

import pickle
from eval import eval_one_to_one_3x3
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from solve import convert_to_action2, convert_to_action3, convert_to_action4, convert_to_direction
from switch_neat import Agent
import switch_neat
import switch_maps
from deap import creator, base
from deap_neat import DeapSwitchGenome, DeapSwitchMapGenome

def get_grid(resfile = 'final.p'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", DeapSwitchGenome, fitness=creator.FitnessMax, features = list)
    res = pickle.load(open(resfile, "rb"))
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

def get_best_agent(size, config_file, resfile='final.p'):
    grid = get_grid(resfile)
    ind = grid.best
    conf = Config(DeapSwitchMapGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)
    net = switch_maps.create(ind, conf, 3)
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

def verify_best_agent():
    agent = get_best_agent(3, 'config/deap-maps-skinner3','out/3x3_qd_maps/skinner3_final.p')
    score, bd = eval_one_to_one_3x3(agent, 200,40,20,True)
    print(f"score: {score}, bd: {bd}")
    scores = [eval_one_to_one_3x3(agent,200,40) for _ in range(1000)]
    scores.sort()
    print(f"scores: {scores}")
    print(f"{len(list(filter( lambda x: x < 170, scores)))} scores are below 170")

if __name__ == '__main__':
    verify_best_agent()
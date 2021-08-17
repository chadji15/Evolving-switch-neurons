import pickle
from eval import eval_one_to_one_3x3
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from solve import convert_to_action2, convert_to_action3, convert_to_action4, convert_to_direction, solve_one_to_one_3x3
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
    net = switch_neat.create(ind, conf)
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
    net = switch_neat.create(ind, conf)
    outf = convert_to_direction
    agent = Agent(net, lambda x: x, outf)
    return agent

def verify_best_agent():
    agent = get_best_agent(3, 'config/deap-skinner3', 'out/3x3_map_elites/float_desc/skinner3_final.p')
    eps = 24
    randiter = 12
    snapiter = 3
    satfit = 12
    score, bd = eval_one_to_one_3x3(agent, eps,randiter, snapiter, True, 'training')
    print(f"score: {score}, bd: {bd}")
    # scores = [eval_one_to_one_3x3(agent,eps,randiter,snapiter,False,'test') for _ in range(1000)]
    # scores.sort()
    # print(f"scores: {scores}")
    # print(f"{len(list(filter( lambda x: x < satfit, scores)))} scores are below {satfit}")

def dry_run_optimal():
    agent=solve_one_to_one_3x3()
    score, bd = eval_one_to_one_3x3(agent, 36,9,3,True,'test')
    print(score, bd)

def validate_all():
    grid = get_grid('out/3x3_map_elites/float_desc/skinner3_final.p')
    conf = Config(DeapSwitchGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  'config/deap-skinner3')
    outf = convert_to_action3
    max_score = 0
    for item in grid.items:
        net = switch_neat.create(item, conf)
        agent = Agent(net, lambda x: x, outf)
        eps = 24
        randiter = 12
        snapiter = 3
        satfit = 12
        score, bd = eval_one_to_one_3x3(agent, eps,randiter, snapiter, True, 'test')
        if score > max_score:
            besta = agent
            max_score=score
            maxbd = bd
    print("Best score:", max_score)
    print("bd:", maxbd)
    print(f"Episodes: {eps}, randiter: {randiter}, snapiter: {snapiter}, satfit: {satfit}")
    scores = [eval_one_to_one_3x3(besta,eps,randiter,snapiter,False,'test') for _ in range(1000)]
    scores.sort()
    print(f"scores: {scores}")
    print(f"{len(list(filter( lambda x: x < satfit, scores)))} scores are below {satfit}")


    return besta

def test_validate():
    agent = validate_all()
    eps = 200
    randiter = 40
    snapiter = 20
    satfit = 170
    print(f"Episodes: {eps}, randiter: {randiter}, snapiter: {snapiter}, satfit: {satfit}")
    scores = [eval_one_to_one_3x3(agent,eps,randiter,snapiter,False,'test') for _ in range(1000)]
    scores.sort()
    print(f"scores: {scores}")
    print(f"{len(list(filter( lambda x: x < satfit, scores)))} scores are below {satfit}")

def plot_stats(file='skinner3.out'):
    import matplotlib.pyplot as plt
    sizes = []
    avgs = []
    mins = []
    maxs = []
    gens = list(range(0,9001))
    maxsize = 10000
    with open(file, 'r') as fp:
        fp.readline()
        fp.readline()
        for i in range(9001):
            line = fp.readline()
            words = line.split()
            size = words[1]
            size = float(size.split('/')[0])/maxsize
            sizes.append(size)
            avg = float(words[4].strip('[]'))
            avgs.append(avg)
            min = float(words[6].strip('[]'))
            mins.append(min)
            max = float(words[7].strip('[]'))
            maxs.append(max)

    plt.subplot(1,2,1)
    plt.plot(gens,sizes)
    plt.xlabel('Generations')
    plt.ylabel('Coverage')

    plt.subplot(1,2,2)
    plt.plot(gens,mins)
    plt.plot(gens,avgs)
    plt.plot(gens,maxs)
    plt.plot(gens, [48 for _ in range(9001)], '--')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend(['Min', 'Avg', 'Max', 'Satisfactory'])

    plt.suptitle('Direct encoding')
    plt.show()

if __name__ == '__main__':
    plot_stats()
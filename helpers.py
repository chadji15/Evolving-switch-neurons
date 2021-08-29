import pickle
from eval import eval_one_to_one_3x3, eval_one_to_one_4x4
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from solve import convert_to_action2, convert_to_action3, convert_to_action4, convert_to_direction, solve_one_to_one_3x3
from switch_neat import Agent
import switch_neat
import switch_maps
from deap import creator, base
from deap_neat import DeapSwitchGenome, DeapSwitchMapGenome, DeapGuidedMapGenome
import guided_maps
from render_network import draw_net

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
    conf = Config(DeapGuidedMapGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation,
                  config_file)
    net = guided_maps.create(ind, conf, 4)
    if size == 2:
        outf = convert_to_action2
    elif size == 3:
        outf = convert_to_action3
    elif size == 4:
        outf = convert_to_action4
    #inf = lambda x: x
    inf = guided_maps.reorder_inputs
    agent = Agent(net, inf, outf)
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
    agent = get_best_agent(4, 'config/deap-guided-skinner4', 'out/4x4_map_elites/guided_maps/skinner3_final.p')
    eps = 200
    randiter = 50
    snapiter = 4
    satfit = 152
    evalf = eval_one_to_one_4x4
    # score, bd = evalf(agent, eps,randiter, snapiter, True, 'training')
    # print(f"score: {score}, bd: {bd}")
    scores = [evalf(agent,eps,randiter,snapiter,False,'test') for _ in range(100)]
    scores.sort()
    print(f"scores: {scores}")
    print(f"{len(list(filter( lambda x: x < satfit, scores)))} scores are below {satfit}")

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

def find_desc(file='out/3x3_qd_maps/float_desc/skinner3_final.p'):
    grid = get_grid(file)
    for key in grid.features:
        feat = grid.features[key]
        if not feat:
            continue
        feat = feat[0]
        if feat[0] != 0 and feat[1] == 0 and feat[2] != 0 and feat[3] == 0:
            fit = grid.fitness[key]
            print(f"Features: {feat}, Fitness: {fit}")

def count_optimal():
    grid = get_grid('out/3x3_qd_maps/guided_maps/skinner3_final.p')
    sat_fit = 48
    for key, fi in grid.fitness.items():
        if not fi:
            continue
        f = fi[0]
        if f > creator.FitnessMax((sat_fit,)):
            feats = grid.features[key]
            print(f"Individual with descriptor: {feats} has fitness: {f}")

def test_nonzero():
    genome = DeapGuidedMapGenome
    config = 'config/deap-guided-skinner3'
    conf = Config(genome, DefaultReproduction,
                    DefaultSpeciesSet, DefaultStagnation,
                    config)
    size = 3
    inf = guided_maps.reorder_inputs
    outf = convert_to_action3
    grid = get_grid('out/3x3_qd_maps/guided_maps/skinner3_final.p')
    sat_fit = 48
    for key, fi in grid.fitness.items():
        if not fi:
            continue
        f = fi[0]
        feats = grid.features[key]
        if not feats:
            continue
        feats = feats[0]
        if f > creator.FitnessMax((sat_fit,)) and (feats[1] > 0 and feats[3] > 0):
            print(f"Individual with descriptor: {feats} has fitness: {f} on training set")
            ind = grid.solutions[key][0]
            net=guided_maps.create(ind, conf,size)
            agent = Agent(net,inf, outf)
            score = eval_one_to_one_3x3(agent, 60,30,3,False,'test',1)
            print(f"On the test set it scores a fitness of {score}")

def visualize_all_optimal():
    outdir = 'optimal_vizs'
    import os
    try:
        os.mkdir(outdir)
    except OSError as e:
        pass

    genome = DeapGuidedMapGenome
    config = 'config/deap-guided-skinner3'
    conf = Config(genome, DefaultReproduction,
                    DefaultSpeciesSet, DefaultStagnation,
                    config)
    size = 3

    grid = get_grid('out/3x3_qd_maps/guided_maps/skinner3_final.p')
    sat_fit = 48
    fp = open(f"{outdir}/index.txt", 'w')
    from itertools import count
    c = count(start=1, step=1)
    for key, fi in grid.fitness.items():
        if not fi:
            continue
        f = fi[0]
        if f > creator.FitnessMax((sat_fit,)):
            feats = grid.features[key]
            index = next(c)
            ind = grid.solutions[key][0]
            net=guided_maps.create(ind, conf,size)
            draw_net(network=net, filename=f"{outdir}/{index}")
            fp.write(f"{index} \n {ind}\n=> Fitness: {f}, descriptor: {feats}\n")

    fp.close()

def plot_neat_vs_map_elites():
    neatresfile = r'out/guided_maps/27_sets/2.txt'
    mapelitesfile = r'out/3x3_qd_maps/guided_maps/skinner3.out'
    satfit = 48
    mapelitesevals = list(range(0,256000,128))
    mapelitesevals.insert(0,2000)
    from itertools import accumulate
    mapelitesevals = list(accumulate(mapelitesevals))
    mapelitesscores = []
    neatevals = []
    neatscores = []

    with open(neatresfile,'r') as fp:
        for line in fp:
            if line.startswith('Best fitness:'):
                score = line.split()[2]
                score = float(score)
                neatscores.append(score)
            elif line.startswith('Population of'):
                popu = line.split()[2]
                popu = int(popu)
                neatevals.append(popu)
    neatevals.append(neatevals[-1])
    neatevals = list(accumulate(neatevals))
    with open(mapelitesfile, 'r') as fp2:
        fp2.readline()
        fp2.readline()
        line = fp2.readline()
        while not line.startswith('Total elapsed:'):
            score = line.split()[7].strip('[]')
            score = float(score)
            mapelitesscores.append(score)
            line = fp2.readline()

    maxevals = max(neatevals[-1], mapelitesevals[-1])
    import matplotlib.pyplot as plt

    plt.xlabel('Evaluations')
    plt.ylabel('Maximum fitness')
    plt.plot(mapelitesevals, mapelitesscores)
    plt.plot(neatevals, neatscores)
    plt.axhline(satfit)
    plt.legend(['MAP-Elites', 'NEAT', 'Satisfactory'])
    plt.savefig('neatvsmapelits.jpg')

if __name__ == '__main__':
    visualize_all_optimal()
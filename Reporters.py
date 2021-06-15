from neat.statistics import StatisticsReporter
from neat.reporting import BaseReporter
from neat.six_util import iteritems
from switch_neat import *
import random
import datetime

#Return the low quartile of the values in the *values array
def low_quartile(values):
    values = list(values)
    n = len(values)
    if n == 1:
        return values[0]
    values.sort()
    if n < 4:
        return (values[1] + values[0]) / 2.0
    if (n % 4) == 1:
        return values[n//4]
    i = n//4
    return (values[i - 1] + values[i])/ 2

#Return the high quartile of the values in the *values array
def high_quartile(values):
    values = list(values)
    n = len(values)
    if n == 1:
        return values[0]
    values.sort()
    if n < 4:
        return (values[-1] + values[-2]) / 2.0
    if (n % 4) == 1:
        return values[n // 4 * 3]
    i = n // 4 * 3
    return (values[i - 1] + values[i]) / 2

#Inherit the StatisticsReported from the neat package to add some desired functionality
class StatReporterv2(StatisticsReporter):

    def get_fitness_low_quartile(self):
        return self.get_fitness_stat(low_quartile)

    def get_fitness_high_quartile(self):
        return self.get_fitness_stat(high_quartile)

    def get_fitness_best(self):
        return self.get_fitness_stat(max)

#This is a reporter that saves the best network genome to a binary file every generation.
class NetRetriever(BaseReporter):

    def __init__(self):
        self.g = 0

    def end_generation(self, config, population, species_set):
        max_fit = 0
        best_genome = None
        #Find the best genome based on fitness
        for id, genome in iteritems(population):
            if not genome.fitness:
                continue
            if genome.fitness > max_fit:
                max_fit = genome.fitness
                best_genome = genome
        try:
            #Save it to the binary file
            fp = open(f"net_gen_{self.g}.bin", 'wb')
            pickle.dump(create(best_genome, config), fp)
            fp.close()
        except:
            pass
        if self.g > 0:
            #Remove the previous genome to avoid cluttering
            os.remove(f"net_gen_{self.g-1}.bin")
        self.g += 1

class EvaluatorMutator(BaseReporter):

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def end_generation(self, config, population, species_set):
        self.evaluator.params = random.sample(self.evaluator.param_list, self.evaluator.samples)

class ProgressTracker(BaseReporter):

    def __init__(self, log_file):
        self.g=0
        self.log_file = log_file

        with open(self.log_file,'w') as fp:
            fp.write(f"0 {datetime.datetime.now()}\n")
        super().__init__()

    def end_generation(self, config, population, species_set):
        self.g+=1
        if self.g % 100 == 0:
            with open(self.log_file,'a') as fp:
                fp.write(f"{self.g} {datetime.datetime.now()}\n")
                fp.flush()
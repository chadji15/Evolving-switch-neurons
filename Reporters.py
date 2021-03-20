from neat.statistics import StatisticsReporter
from neat.reporting import BaseReporter
from neat.six_util import iteritems
from switch_neat import *

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


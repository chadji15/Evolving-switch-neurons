from neat.statistics import StatisticsReporter

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

class StatReporterv2(StatisticsReporter):

    def get_fitness_low_quartile(self):
        return self.get_fitness_stat(low_quartile)

    def get_fitness_high_quartile(self):
        return self.get_fitness_stat(high_quartile)

    def get_fitness_best(self):
        return self.get_fitness_stat(max)
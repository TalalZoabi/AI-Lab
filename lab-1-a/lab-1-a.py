import random
import string
import numpy as np
import time
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, target_string, pop_size=100, max_generations=100, mutation_rate=0.01, crossover_rate=0.7, fitness_func=None, crossover_method="single"):
        self.target_string = target_string
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_func = fitness_func if fitness_func else self.default_fitness_func
        self.crossover_method = crossover_method
        self.population = self.init_population()
        self.best_individual = None
        self.avg_fitnesses = []
        self.stddev_fitnesses = []
        self.best_fitnesses = []
        self.elapsed_times = []
        self.fitness_distributions = []

    def init_population(self):
        return [''.join(random.choices(string.ascii_letters + ' ', k=len(self.target_string))) for _ in range(self.pop_size)]

    def default_fitness_func(self, individual):
        return sum(1 for a, b in zip(individual, self.target_string) if a == b)

    def select_parents(self):
        weights = [self.fitness_func(ind) for ind in self.population]
        total_fitness = sum(weights)
        probabilities = [w / total_fitness for w in weights]
        parents = random.choices(self.population, probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            if self.crossover_method == "single":
                point = random.randint(1, len(self.target_string) - 1)
                return parent1[:point] + parent2[point:]
            elif self.crossover_method == "two_point":
                point1 = random.randint(1, len(self.target_string) - 1)
                point2 = random.randint(1, len(self.target_string) - 1)
                if point1 > point2:
                    point1, point2 = point2, point1
                return parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            elif self.crossover_method == "uniform":
                return ''.join(random.choice(pair) for pair in zip(parent1, parent2))
        return parent1

    def mutate(self, individual):
        individual = list(individual)
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(string.ascii_letters + ' ')
        return ''.join(individual)

    def evolve(self):
        new_population = []
        for _ in range(self.pop_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def run(self):
        start_time = time.time()
        for generation in range(self.max_generations):
            self.evolve()
            fitnesses = [self.fitness_func(ind) for ind in self.population]
            avg_fitness = np.mean(fitnesses)
            stddev_fitness = np.std(fitnesses)
            best_individual = max(self.population, key=self.fitness_func)
            best_fitness = self.fitness_func(best_individual)
            elapsed_time = time.time() - start_time
            fitness_distribution = np.histogram(fitnesses, bins=np.linspace(0, max(fitnesses), 11))[0]

            self.avg_fitnesses.append(avg_fitness)
            self.stddev_fitnesses.append(stddev_fitness)
            self.best_fitnesses.append(best_fitness)
            self.elapsed_times.append(elapsed_time)
            self.fitness_distributions.append(fitness_distribution)

            if best_individual == self.target_string:
                self.best_individual = best_individual
                break

        self.plot_results()

    def plot_results(self):
        generations = range(len(self.avg_fitnesses))

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(generations, self.avg_fitnesses, label='Average Fitness')
        plt.xlabel('Generations')
        plt.ylabel('Average Fitness')
        plt.title('Average Fitness Over Generations')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(generations, self.stddev_fitnesses, label='Standard Deviation of Fitness')
        plt.xlabel('Generations')
        plt.ylabel('Stddev Fitness')
        plt.title('Standard Deviation of Fitness Over Generations')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(generations, self.best_fitnesses, label='Best Fitness')
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness Over Generations')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(generations, self.elapsed_times, label='Elapsed Time (s)')
        plt.xlabel('Generations')
        plt.ylabel('Elapsed Time (s)')
        plt.title('Elapsed Time Over Generations')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Example Usage
if __name__ == "__main__":
    ga = GeneticAlgorithm(target_string="hello world", crossover_method="two_point")
    ga.run()

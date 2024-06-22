import random
import time
from .DataCollector import DataCollector

class GeneticAlgorithm:
    def __init__(self, config):
        self.problem = config['problem']
        self.initialize_population = config['initialize_population']
        self.fitness_function = config['fitness_function']
        self.parent_selection = config['parent_selection']
        self.crossover_operator = config['crossover_operator']
        self.mutation_operator = config['mutation_operator']
        self.survivor_selection = config['survivor_selection']
        self.population_size = config['population_size']
        self.num_generations = config['num_generations']
        self.mutation_rate = config['mutation_rate']
        self.data_collector = config['data_collector']

        self.best_lifetime_individual = None
        self.best_lifetime_fitness = float('-inf')

    def run(self):
        # Initialize population
        population = self.initialize_population(self.population_size)
        ages = [0] * self.population_size
        
        for generation in range(self.num_generations):
            start_time = time.time()
            fitnesses = [self.fitness_function.evaluate(ind) for ind in population]
            
            # Parent selection
            parents = self.parent_selection.select(population, fitnesses, self.population_size)
            
            # Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[(i+1) % self.population_size]
                offspring.extend(self.crossover_operator.crossover(parent1, parent2))
            
            # Mutation
            offspring = [self.mutation_operator.mutate(ind) if random.random() < self.mutation_rate else ind for ind in offspring]
            
            # Evaluate fitness
            fitnesses.extend([self.fitness_function.evaluate(ind) for ind in offspring])
            
            population.extend(offspring)

            # Update best individual
            best_fitness = max(fitnesses)
            best_individual = population[fitnesses.index(best_fitness)]
            if best_fitness > self.best_lifetime_fitness:
                self.best_lifetime_fitness = best_fitness
                self.best_lifetime_individual = best_individual
            


            # Survivor selection
            population, ages = self.survivor_selection.select(population, fitnesses, self.population_size, ages)
            

            # Collect and report data
            end_time = time.time()
            self.collect_statistics(fitnesses, generation, end_time - start_time)
            
            self.data_collector.collect(fitnesses, end_time - start_time, population)
        
        # return best individual
        fitnesses = [self.fitness_function.evaluate(ind) for ind in population]
        best_individual = population[fitnesses.index(best_fitness)]
        self.data_collector.plot()
        self.problem.display_individual(best_individual, "Best Individual")
        self.problem.display_individual(self.best_lifetime_individual, "Best Lifetime Individual")
        return best_individual

    def collect_statistics(self, fitnesses, generation, runtime):
        mean_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        print(f"Generation {generation}: Mean Fitness = {mean_fitness}, Max Fitness = {max_fitness}, Min Fitness = {min_fitness}, Runtime = {runtime:.4f} seconds")



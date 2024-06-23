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
        self.mutation_strategy = config['mutation_strategy']
        self.data_collector = config['data_collector']

        self.best_lifetime_individual = None
        self.best_lifetime_fitness = float('-inf')
        
        self.plot_results = config['plot_results']
        self.terminal_log = config['terminal_log']

        # Integrate FitnessSharing, Speciation, and Crowding
        self.fitness_sharing = config['fitness_sharing']
        self.speciation = config['speciation']
        self.crowding = config['crowding']


    def run(self):
        # Initialize population
        population = self.initialize_population(self.population_size)
        ages = [0] * self.population_size
        
        for generation in range(self.num_generations):
            start_time = time.time()
            fitnesses = [self.fitness_function.evaluate(ind) for ind in population]
            
            # Apply fitness sharing
            shared_fitness = self.fitness_sharing.apply_sharing(population, fitnesses)
            
            # Perform speciation
            species = self.speciation.threshold_speciation(population)
            
            new_population = population
            new_ages = ages
            for spec in species:
                sub_population = [population[i] for i in spec]
                sub_fitness = [shared_fitness[i] for i in spec]
                
                # Parent selection within species
                parents = self.parent_selection.select(sub_population, sub_fitness, len(sub_population))
                
                # Crossover and mutation
                offspring = []
                for i in range(0, len(parents), 2):
                    parent1 = parents[i]
                    parent2 = parents[(i + 1) % len(parents)]
                    offspring.extend(self.crossover_operator.crossover(parent1, parent2))
                offspring = self.mutate(offspring, {'generation': generation})
                
                # Evaluate offspring fitnesses
                offspring_fitnesses = [self.fitness_function.evaluate(ind) for ind in offspring]

                # Apply crowding
                sub_population.extend(offspring)
                sub_fitness.extend(offspring_fitnesses)
                new_sub_population = self.crowding.apply_crowding(offspring, sub_population, sub_fitness)
                
                new_population.extend(new_sub_population)
                new_ages.extend([0] * len(new_sub_population))  # Reset ages for new individuals

            # Survivor selection
            population, ages = self.survivor_selection.select(new_population, [self.fitness_function.evaluate(ind) for ind in new_population], self.population_size, new_ages)

            # Update best individual
            fitnesses = [self.fitness_function.evaluate(ind) for ind in population]
            best_fitness = max(fitnesses)
            best_individual = population[fitnesses.index(best_fitness)]
            if best_fitness > self.best_lifetime_fitness:
                self.best_lifetime_fitness = best_fitness
                self.best_lifetime_individual = best_individual

            # Collect and report data
            end_time = time.time()
            self.collect_statistics(fitnesses, generation, end_time - start_time)
            self.data_collector.collect(fitnesses, end_time - start_time, population)
        
        # Return best individual
        fitnesses = [self.fitness_function.evaluate(ind) for ind in population]
        best_individual = population[fitnesses.index(max(fitnesses))]
        
        if self.plot_results:
            self.data_collector.plot()
            self.problem.display_individual(best_individual, "Best Individual")
            self.problem.display_individual(self.best_lifetime_individual, "Best Lifetime Individual")

        return best_individual





    def collect_statistics(self, fitnesses, generation, runtime):
        mean_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)

        if self.terminal_log:
            print(f"Generation {generation}: Mean Fitness = {mean_fitness}, Max Fitness = {max_fitness}, Min Fitness = {min_fitness}, Runtime = {runtime:.4f} seconds")

    def mutate(self, offspring, config):
        offspring_fitnesses = [self.fitness_function.evaluate(ind) for ind in offspring]
        config['min_fitness'] = self.fitness_function.min_fitness()
        config['max_fitness'] = self.fitness_function.max_fitness()
        config['best_fitness'] = max(offspring_fitnesses)
        config['avg_fitness'] = sum(offspring_fitnesses) / len(offspring_fitnesses)

        mutated_offspring = []

        for i, ind in enumerate(offspring):
            config['individual_fitness'] = offspring_fitnesses[i]
            if self.mutation_strategy.should_mutate(config):
                mutated_offspring.append(self.mutation_operator.mutate(ind))
            else:
                mutated_offspring.append(ind)

        return mutated_offspring


    def get_results(self):
        return self.data_collector.get_data()


import random
import numpy as np
import matplotlib.pyplot as plt

class BaldwinExperiment:
    def __init__(self, target_length, population_size=1000, generations=100, learning_attempts=1000, mutation_rate=0.01):
        self.target_length = target_length
        self.population_size = population_size
        self.generations = generations
        self.learning_attempts = learning_attempts
        self.mutation_rate = mutation_rate
        self.target = self.generate_target()
        self.population = self.initialize_population()
        self.data = {'incorrect': [], 'correct': [], 'learned': []}

    def generate_target(self):
        return [random.choice([0, 1, '?']) for _ in range(self.target_length)]

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = []
            for i in range(self.target_length):
                if self.target[i] == '?':
                    individual.append(random.choice([0, 1]))
                else:
                    individual.append(random.choice(['?', self.target[i]]))
            population.append(individual)
        return population

    def fitness(self, individual):
        correct = 0
        for i in range(self.target_length):
            if self.target[i] != '?' and individual[i] == self.target[i]:
                correct += 1
        return correct

    def learning_phase(self, individual):
        learned_bits = 0
        for _ in range(self.learning_attempts):
            candidate = individual[:]
            bit_to_flip = random.randint(0, self.target_length - 1)
            if candidate[bit_to_flip] == '?':
                candidate[bit_to_flip] = 1 if self.target[bit_to_flip] == 1 else 0
            else:
                candidate[bit_to_flip] = 1 - candidate[bit_to_flip]
            
            if self.fitness(candidate) > self.fitness(individual):
                individual = candidate
                learned_bits += 1
        return individual, learned_bits

    def mutate(self, individual):
        for i in range(self.target_length):
            if random.random() < self.mutation_rate:
                if self.target[i] == '?':
                    individual[i] = 1 if individual[i] == 0 else 0
                else:
                    individual[i] = '?' if individual[i] == self.target[i] else self.target[i]
        return individual

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.target_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def selection(self, population):
        fitnesses = [self.fitness(ind) for ind in population]
        selected = random.choices(population, weights=fitnesses, k=self.population_size)
        return selected

    def run_experiment(self, with_learning=True):
        for generation in range(self.generations):
            new_population = []
            total_learned_bits = 0
            for individual in self.population:
                if with_learning:
                    individual, learned_bits = self.learning_phase(individual)
                    total_learned_bits += learned_bits
                new_population.append(individual)

            self.population = self.selection(new_population)
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = self.population[i], self.population[(i + 1) % self.population_size]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))

            self.population = offspring[:self.population_size]
            self.collect_data(generation, total_learned_bits)

    def collect_data(self, generation, total_learned_bits):
        incorrect_positions = 0
        correct_positions = 0
        for individual in self.population:
            for i in range(self.target_length):
                if self.target[i] == '?':
                    continue
                if individual[i] == self.target[i]:
                    correct_positions += 1
                else:
                    incorrect_positions += 1

        total_positions = self.population_size * self.target_length
        learned_bits_percentage = (total_learned_bits / (self.population_size * self.learning_attempts)) * 100
        self.data['incorrect'].append((incorrect_positions / total_positions) * 100)
        self.data['correct'].append((correct_positions / total_positions) * 100)
        self.data['learned'].append(learned_bits_percentage)

    def plot_results(self):
        generations = range(self.generations)
        plt.figure(figsize=(12, 8))

        plt.plot(generations, self.data['incorrect'], label='Incorrect Positions (%)')
        plt.plot(generations, self.data['correct'], label='Correct Positions (%)')
        plt.plot(generations, self.data['learned'], label='Learned Bits (%)')

        plt.xlabel('Generations')
        plt.ylabel('Percentage')
        plt.title('Baldwin Effect Experiment')
        plt.legend()
        plt.show()



def sanity_check():
    target_length = 50
    population_size = 1000
    generations = 50
    learning_attempts = 1000
    mutation_rate = 0.01

    # Initialize the experiment with the specified parameters
    experiment = BaldwinExperiment(
        target_length=target_length,
        population_size=population_size,
        generations=generations,
        learning_attempts=learning_attempts,
        mutation_rate=mutation_rate
    )

    print("Target Pattern:", experiment.target)
    
    # Run the experiment with learning phase
    print("Running experiment with learning phase...")
    experiment.run_experiment(with_learning=True)
    experiment.plot_results()

    # Reset the data and population for the experiment without learning phase
    experiment.data = {'incorrect': [], 'correct': [], 'learned': []}
    experiment.population = experiment.initialize_population()

    # Run the experiment without learning phase
    print("Running experiment without learning phase...")
    experiment.run_experiment(with_learning=False)
    experiment.plot_results()

    # Print final results for verification
    print("\nFinal Results with Learning Phase:")
    print("Average Incorrect Positions:", np.mean(experiment.data['incorrect']))
    print("Average Correct Positions:", np.mean(experiment.data['correct']))
    print("Average Learned Bits:", np.mean(experiment.data['learned']))

    # Reset the data for the experiment without learning phase
    experiment.data = {'incorrect': [], 'correct': [], 'learned': []}
    experiment.population = experiment.initialize_population()
    experiment.run_experiment(with_learning=False)

    print("\nFinal Results without Learning Phase:")
    print("Average Incorrect Positions:", np.mean(experiment.data['incorrect']))
    print("Average Correct Positions:", np.mean(experiment.data['correct']))
    print("Average Learned Bits:", np.mean(experiment.data['learned']))

# Run the sanity check
sanity_check()





# Lab Report: Genetic Algorithms

### University of Haifa
Faculty of Social Sciences  
Department of Computer Science  
Artificial Intelligence Laboratory  
Lab 1A: Genetic Algorithms (Self-Organizing Systems) - Part A

**Course:** 203.3630  
**Semester:** Spring 2024  
**Instructor:** Shai Bushinski  
**Due Date:** Friday, May 24, 2024  
**Submission:** Via email to shay@cs.haifa.ac.il  
**Grade Component:** Mandatory  
**Submission Conditions:** The assignment can be submitted in pairs or individually.

---

## Implementation and Discussion

### Genetic Algorithm Implementation

The implementation of the genetic algorithm includes initializing a population, selecting parents, performing crossover and mutation, and evolving the population over generations. The algorithm aims to find a target string using a genetic approach.

#### Code Overview

```python
class GeneticAlgorithm:
    def __init__(self, target_string, ...):
        ...
        self.population = self.init_population()
        ...

    def init_population(self):
        ...

    def select_parents(self):
        ...

    def crossover(self, parent1, parent2):
        ...

    def mutate(self, individual):
        ...

    def evolve(self):
        ...

    def run(self):
        start_time = time.time()
        for generation in range(self.max_generations):
            self.evolve()
            fitnesses = [self.fitness_func(ind) for ind in self.population]
            ...
            self.avg_fitnesses.append(avg_fitness)
            self.stddev_fitnesses.append(stddev_fitness)
            self.best_fitnesses.append(best_fitness)
            self.elapsed_times.append(elapsed_time)
            ...
        self.plot_results()

    def plot_results(self):
        ...
```

### 1. Average Fitness and Standard Deviation

In each generation, the algorithm calculates and reports the average fitness and standard deviation of the population.

**Implementation:**

```python
class GeneticAlgorithm:
    ...
    def run(self):
        start_time = time.time()
        for generation in range(self.max_generations):
            ...
            fitnesses = [self.fitness_func(ind) for ind in self.population]
            avg_fitness = np.mean(fitnesses)
            stddev_fitness = np.std(fitnesses)
            ...
            self.avg_fitnesses.append(avg_fitness)
            self.stddev_fitnesses.append(stddev_fitness)
            ...
        self.plot_results()
    ...
```

### 2. Runtime and Elapsed Time

The algorithm calculates and reports the runtime for each generation and the total elapsed time.

**Implementation:**

```python
class GeneticAlgorithm:
    ...
    def run(self):
        start_time = time.time()
        for generation in range(self.max_generations):
            ...
            elapsed_time = time.time() - start_time
            self.elapsed_times.append(elapsed_time)
            ...
        self.plot_results()
    ...
```

### 3. Fitness Distribution

The algorithm reports the fitness distribution of the population for each generation.

**Implementation:**

```python
class GeneticAlgorithm:
    ...
    def run(self):
        ...
            fitness_distribution = np.histogram(fitnesses, bins=np.linspace(0, max(fitnesses), 11))[0]
            self.fitness_distributions.append(fitness_distribution)
            ...
        self.plot_results()
    ...
```

### 4. Crossover Operators

The algorithm supports three crossover operators: single-point, two-point, and uniform crossover. 

**Implementation:**

```python
class GeneticAlgorithm:
    ...
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            if self.crossover_method == "single":
                ...
            elif self.crossover_method == "two_point":
                ...
            elif self.crossover_method == "uniform":
                ...
        return parent1
    ...
```

### 5. "Bullseye" Heuristic

A new heuristic, "Bullseye," is implemented, which rewards correct characters even if they are not in the correct position.

**Implementation:**

```python
class GeneticAlgorithm:
    ...
    def bullseye_fitness_func(self, individual):
        return sum(3 if a == b else 1 if a in self.target_string else 0 for a, b in zip(individual, self.target_string))
    ...
```

### 6. Heuristic Comparison

We compare the original heuristic and the "Bullseye" heuristic.

**Analysis:**
- The original heuristic measures exact matches.
- The "Bullseye" heuristic rewards partial matches, potentially leading to faster convergence.

**Implementation:**

```python
if __name__ == "__main__":
    ga_original = GeneticAlgorithm(target_string="hello world")
    ga_original.run()
    
    ga_bullseye = GeneticAlgorithm(target_string="hello world", fitness_func=lambda ind: ...)
    ga_bullseye.run()
```

### 7. Exploration and Exploitation

**Exploration:** 
- Mutation introduces random changes, promoting diversity.
- Random crossover points create new individuals exploring different parts of the solution space.

**Exploitation:**
- Fitness function selects the best individuals.
- Selection based on fitness ensures better solutions reproduce.

### 8. Configurations Comparison

We compare the algorithm's performance under different configurations: only crossover, only mutation, and both.

**Implementation:**

```python
if __name__ == "__main__":
    ...
    ga_crossover_only = GeneticAlgorithm(target_string="hello world", mutation_rate=0.0)
    ga_crossover_only.run()

    ga_mutation_only = GeneticAlgorithm(target_string="hello world", crossover_rate=0.0)
    ga_mutation_only.run()

    ga_both = GeneticAlgorithm(target_string="hello world")
    ga_both.run()
    ...
```

### 9. Simulation and Heuristic Performance

We run simulations and analyze the performance of the two heuristics under the preferred configuration.

**Implementation:**

```python
if __name__ == "__main__":
    ...
    ga_original = GeneticAlgorithm(target_string="hello world")
    ga_original.run()
    
    ga_bullseye = GeneticAlgorithm(target_string="hello world", fitness_func=lambda ind: ...)
    ga_bullseye.run()
    ...
```

### Results and Analysis

After running the algorithm, we plot the results using matplotlib to visualize the performance metrics over generations.

**Plotting Results:**

```python
class GeneticAlgorithm:
    ...
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
```

### Conclusion

This lab successfully implements and extends a genetic algorithm to solve a string matching problem. By adding features such as average fitness, standard deviation, runtime measurement, and fitness distribution, we gain valuable insights into the algorithm's performance. The "Bullseye" heuristic proves to be an effective enhancement, and our exploration of different configurations highlights the importance of balancing exploration and exploitation in genetic algorithms.


### Lab Report for Assignment 1B: Genetic Algorithm Implementation

#### Section 1: Modularity

**Requirement**: The framework must support multiple problem types, fitness functions, and genetic operators.

**Implementation**:
To achieve modularity, we designed a base `Problem` class from which specific problem classes inherit. This allows each problem to implement its own initialization, fitness evaluation, and display methods.

```python
class Problem:
    def initialize_population(self):
        raise NotImplementedError

    def evaluate_fitness(self, candidate):
        raise NotImplementedError

    def display_individual(self, individual):
        raise NotImplementedError
```

Specific problem implementations like `StringMatchingProblem`, `SudokuProblem`, and `BinPackingProblem` inherit from the `Problem` class and define their unique behaviors:

```python
class StringMatchingProblem(Problem):
    def __init__(self, target_string):
        self.target_string = target_string

    def initialize_population(self, population_size):
        population = []
        for _ in range(population_size):
            individual = ''.join(random.choices(self.target_string, k=len(self.target_string)))
            population.append(individual)
        return population

    def evaluate_fitness(self, individual):
        matches = sum(1 for i, j in zip(individual, self.target_string) if i == j)
        return matches
    
    def display_individual(self, individual):
        ...
```

**Explanation**:
The base `Problem` class provides a template for specific problem implementations. This approach ensures that new problems can be added by simply inheriting from `Problem` and implementing the required methods.

---

#### Section 2: Customizability

**Requirement**: Users should be able to specify configurations such as population size, number of generations, crossover, and mutation rates.

**Implementation**:
The `read_config` function reads user inputs to set up the genetic algorithm's configuration. This includes selecting the problem type, specifying the target string for string matching, and setting parameters like population size and mutation rate.

```python
def read_config():
    config = {}
    problem_type = input("Enter the problem type (1: String Matching, 2: Sudoku, 3: Bin Packing): ")
    if problem_type == '1':
        config['problem_type'] = 'string_matching'
        config['problem'] = StringMatchingProblem(target_string=input("Enter the target string: "))
    elif problem_type == '2':
        config['problem_type'] = 'sudoku'
        config['problem'] = SudokuProblem(initial_board=parse_sudoku_board())
    elif problem_type == '3':
        config['problem_type'] = 'bin_packing'
        config['problem'] = BinPackingProblem(items=parse_items(), bin_capacity=parse_bin_capacity())
    ...
    config['population_size'] = 100
    config['num_generations'] = 100
    return config

config = read_config()
```

**Explanation**:
By prompting the user for inputs, the framework can be customized to different problems and configurations. This ensures flexibility and adaptability.

---

#### Section 3: Flexibility

**Requirement**: The framework must handle various types of genetic operators, including problem-specific ones.

**Implementation**:
Genetic operators are implemented as separate classes for both mutation and crossover operations. Each operator is tailored to specific problems.

**Mutation Operators**:
```python
class MutationOperator:
    def mutate(self, candidate):
        raise NotImplementedError

class SwapMutation(MutationOperator):
    def mutate(self, candidate):
        candidate = list(candidate)
        idx1, idx2 = random.sample(range(len(candidate)), 2)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        return ''.join(candidate)

class SudokuSwapMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        ...
```

**Crossover Operators**:
```python
class CrossoverOperator:
    def crossover(self, parent1, parent2):
        raise NotImplementedError

class SinglePointCrossover(CrossoverOperator):
    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return [offspring1, offspring2]

class SudokuCXCrossover(CrossoverOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def cycle_crossover(self, parent1_genome, parent2_genome):
        ...
```

**Explanation**:
By defining genetic operators as separate classes, the framework allows for easy extension and customization of genetic operations. This design supports flexibility and problem-specific adaptations.

---

#### Section 4: Data Collection

**Requirement**: Collect and report data on fitness, runtime, and genetic diversity.

**Implementation**:
A `DataCollector` class is used to gather statistics throughout the genetic algorithm's execution. It tracks fitness values, runtime, and genetic diversity, and checks for convergence.

```python
class DataCollector:
    def __init__(self, global_min_fitness, global_max_fitness, distance_method, distance_unit, convergence_threshold=0.01, convergence_window=10):
        self.global_min_fitness = global_min_fitness
        self.global_max_fitness = global_max_fitness
        self.distance_method = distance_method
        self.distance_unit = distance_unit
        self.fitness_matrix = []
        self.runtimes = []
        self.diversities = []
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.convergence_generation = None

    def collect(self, fitnesses, runtime, population):
        self.fitness_matrix.append(fitnesses)
        self.runtimes.append(runtime)
        self.diversities.append(self.calculate_diversity(population))
        if self.convergence_generation is None:
            self.check_convergence()
    
    def plot(self):
        generations = range(len(self.fitness_matrix))
        avg_fitnesses = [sum(fitnesses) / len(fitnesses) for fitnesses in self.fitness_matrix]
        ...
        plt.plot(generations, avg_fitnesses, label="Average Fitness")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Over Generations')
        plt.legend()
        plt.show()
```

**Explanation**:
The `DataCollector` class provides detailed insights into the algorithm's performance by collecting and plotting data. This aids in analyzing and optimizing the algorithm.

---

#### Conclusion

The genetic algorithm framework meets the specified requirements of modularity, customizability, flexibility, and data collection. The design choices ensure that the framework is adaptable to various optimization problems and user preferences, making it a robust tool for research and practical applications in genetic algorithms.
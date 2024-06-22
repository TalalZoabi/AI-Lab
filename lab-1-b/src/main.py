from .modules.GeneticAlgorithm import GeneticAlgorithm
from .modules.Problem import StringMatchingProblem, SudokuProblem, BinPackingProblem
from .modules.FitnessFunction import MatchFitness
from .modules.ParentSelectionMethod import RWSLinearScaling
from .modules.LinearScaling import ConstantLinearScaling
from .modules.CrossoverOperator import SinglePointCrossover
from .modules.MutationOperator import SinglePointMutation
from .modules.SurvivorSelectionMethod import ElitismSelection
from .modules.DataCollector import DataCollector

from .utils.readers import *

def read_config():
    config = {}
    problem_type = input("Enter the problem type (1: String Matching, 2: Sudoku, 3: Bin Packing): ")

    if problem_type == '1':
        config['problem_type'] = 'string_matching'
    elif problem_type == '2':
        config['problem_type'] = 'sudoku'
    elif problem_type == '3':
        config['problem_type'] = 'bin_packing'

    config['problem'] = read_problem(config)
    config['fitness_function'] = read_fitness_function(config)
    config['parent_selection'] = read_parent_selection(config)
    config['crossover_operator'] = read_crossover(config)
    config['mutation_operator'] = read_mutation(config)
    config['survivor_selection'] = read_survivor_selection(config)
    config['data_collector'] = read_data_collector(config)

    population_size = input("Enter the population size (default 100): ")
    config['population_size'] = int(population_size) if population_size else 100

    num_generations = input("Enter the number of generations (default 100): ")
    config['num_generations'] = int(num_generations) if num_generations else 100

    config['initialize_population'] = lambda size: config['problem'].initialize_population(size)

    return config




config = read_config()

ga = GeneticAlgorithm(config)
best_ind = ga.run()

print(f"Best individual: {best_ind}")



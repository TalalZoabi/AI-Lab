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




config1 = {
    'name': 'Config 1',
    'problem': StringMatchingProblem(target_string='HELLO WORLD'),
    'initialize_population': initialize_population_function,
    'fitness_function': fitness_function,
    'parent_selection': tournament_selection,
    'crossover_operator': single_point_crossover,
    'mutation_operator': bit_flip_mutation,
    'survivor_selection': elitism_selection,
    'population_size': 100,
    'num_generations': 100,
    'mutation_strategy': BasicMutation(0.01),
    'data_collector': DataCollector()
}


ga = GeneticAlgorithm(config)
best_ind = ga.run()

print(f"Best individual: {best_ind}")



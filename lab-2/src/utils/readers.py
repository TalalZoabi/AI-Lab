from ..modules.CrossoverOperator import SinglePointCrossover, TwoPointCrossover, UniformCrossover, SudokuSubgridCrossover, SudokuRowColumnCrossover, SudokuCXCrossover, BinPackingCXCrossover, SudokuPMXCrossover, BinPackingPMXCrossover
from ..modules.DataCollector import DataCollector
from ..modules.Problem import StringMatchingProblem, SudokuProblem, BinPackingProblem
from ..modules.FitnessFunction import MatchFitness, SudokuFitness, CombinedBinPackingFitness
from ..modules.MutationOperator import SwapMutation\
                , InversionMutation\
                ,SinglePointMutation\
                , SudokuSwapMutation\
                , SudokuInversionMutation\
                , SudokuScrambleMutation\
                , SudokuRowColumnSwapMutation\
                , SudokuRandomResettingMutation\
                , SudokuDisplacementMutation\
                , BinPackingInsertionMutation\
                , BinPackingSwapMutation\
                , BinPackingInversionMutation\
                , BinPackingScrambleMutation
from ..modules.LinearScaling import ConstantLinearScaling, DynamicLinearScaling
from ..modules.ParentSelectionMethod import RWSLinearScaling, SUSLinearScaling, RWSRankingSelection, TournamentSelection
from ..modules.SurvivorSelectionMethod import ElitismSelection, TournamentSelection as survivor_tournament_selection, AgingSelection
from ..modules.MutationStrategy import BasicMutation, NonUniformMutation, AdaptiveMutation, TriggeredHyperMutation, SelfAdaptiveMutation
from ..modules.FitnessSharing import BasicFitnessSharing
from ..modules.Speciation import ThresholdSpeciation, KMeansSpeciation, SilhouetteKMeansSpeciation
from ..modules.Crowding import DeterministicCrowding, NonDeterministicCrowding

from .parsers import parse_bin_packing_config, parse_sudoku_configurations



def read_crossover(config):
    if 'crossover_method' not in config:
        crossover_method = input("Enter the crossover method (1: Single Point, 2: Two Point, 3: Uniform, 4: Sudoku Subgrid, 5: Sudoku Row/Column, 6: Sudoku CX, 7: Bin Packing CX, 8: Sudoku PMX, 9: Bin Packing PMX): ")
    else:
        crossover_method = config['crossover_method']

    if crossover_method == '1':
        return SinglePointCrossover()
    elif crossover_method == '2':
        return TwoPointCrossover()
    elif crossover_method == '3':
        return UniformCrossover()
    elif crossover_method == '4':
        return SudokuSubgridCrossover(config['fixed_positions'])
    elif crossover_method == '5':
        return SudokuRowColumnCrossover(config['fixed_positions'])
    elif crossover_method == '6':
        return SudokuCXCrossover(config['fixed_positions'])
    elif crossover_method == '7':
        return BinPackingCXCrossover()
    elif crossover_method == '8':
        return SudokuPMXCrossover(config['fixed_positions'])
    
    return BinPackingPMXCrossover()

def read_data_collector(config):
    global_min_fitness = config['fitness_function'].min_fitness()
    global_max_fitness = config['fitness_function'].max_fitness()
    distance_method = config['problem'].distance
    distance_unit = config['problem'].distance_unit()


    if 'convergence_threshold' not in config:
        convergence_threshold = input("Enter the convergence threshold (or leave blank for default 0.01):")
        if not convergence_threshold:
            convergence_threshold = 0.01
        else:
            convergence_threshold = float(convergence_threshold)
    else:
        convergence_threshold = config['convergence_threshold']


    if 'convergence_window' not in config:
        convergence_window = input("Enter the convergence window (or leave blank for default 10):")
        if not convergence_window:
            convergence_window = 10
        else:
            convergence_window = int(convergence_window)
    else:
        convergence_window = config['convergence_window']

    return DataCollector(global_min_fitness, global_max_fitness, distance_method, distance_unit, convergence_threshold, convergence_window)

def read_fitness_function(config):
    if config['problem_type'] == 'string_matching':
        if 'chosen_fitness' not in config:
            chosen_fitness = input("Enter the fitness function (1: MatchFitness): ")
        else:
            chosen_fitness = config['chosen_fitness']
        
        if chosen_fitness == '1':
            return MatchFitness(config['problem'].target_string)
        else:
            raise ValueError("Invalid fitness function specified.")

    elif config['problem_type'] == 'sudoku':
        if 'chosen_fitness' not in config:
            chosen_fitness = input("Enter the fitness function (1: SudokuFitness): ")
        else:
            chosen_fitness = config['chosen_fitness']

        if chosen_fitness == '1':
            return SudokuFitness()
        else:
            raise ValueError("Invalid fitness function specified.")
    elif config['problem_type'] == 'bin_packing':
        if 'chosen_fitness' not in config:
            chosen_fitness = input("Enter the fitness function (1: CombinedBinPackingFitness): ")
        else: 
            chosen_fitness = config['chosen_fitness']
        
        if chosen_fitness == '1':
            if 'weight_bins_used' not in config: 
                weight_bins_used = input("Enter the weight for bins used (default 1.0): ")
                weight_bins_used = float(weight_bins_used) if weight_bins_used else 1.0
            else:
                weight_bins_used = config['weight_bins_used']
            
            if 'weight_total_waste' not in config:
                weight_total_waste = input("Enter the weight for total waste (default 1.0): ")
                weight_total_waste = float(weight_total_waste) if weight_total_waste else 1.0
            else:
                weight_total_waste = config['weight_total_waste']
            
            if 'weight_load_balance' not in config:
                weight_load_balance = input("Enter the weight for load balance (default 1.0): ")
                weight_load_balance = float(weight_load_balance) if weight_load_balance else 1.0
            else:
                weight_load_balance = config['weight_load_balance']
            
            return CombinedBinPackingFitness(bin_capacity=config['bin_capacity'], 
                                             weight_bins_used=weight_bins_used, 
                                             weight_total_waste=weight_total_waste, 
                                             weight_load_balance=weight_load_balance)
    else:
        raise ValueError("Invalid problem type specified.")

def read_mutation_operator(config):
    if config['problem_type'] == 'string_matching':
        if 'mutation_operator_type' not in config:
            mutation_operator_type = input("Enter the mutation operator type (1: Swap 2. Inversion 3. Single Point ): ")
        else:
            mutation_operator_type = config['mutation_operator_type']
        
        if mutation_operator_type == '1':
            return SwapMutation()
        elif mutation_operator_type == '2':
            return InversionMutation()
        elif mutation_operator_type == '3':
            return SinglePointMutation()
        else:
            raise ValueError("Invalid mutation type specified.")
    
    elif config['problem_type'] == 'sudoku':
        if 'mutation_operator_type' not in config:
            mutation_operator_type = input("Enter the mutation type: \n1. Sudoku Swap\n2. Sudoku Inversion\n3. Sudoku Scramble\n4. Sudoku Row Column Swap\n5. Sudoku Random Resetting\n6. SudokuDisplacement\n: ")
        else:
            mutation_operator_type = config['mutation_operator_type']

        if mutation_operator_type == '1':
            return SudokuSwapMutation(config['fixed_positions'])
        elif mutation_operator_type == '2':
            return SudokuInversionMutation(config['fixed_positions'])
        elif mutation_operator_type == '3':
            return SudokuScrambleMutation(config['fixed_positions'])
        elif mutation_operator_type == '4':
            return SudokuRowColumnSwapMutation(config['fixed_positions'])
        elif mutation_operator_type == '5':
            return SudokuRandomResettingMutation(config['fixed_positions'])
        elif mutation_operator_type == '6':
            return SudokuDisplacementMutation(config['fixed_positions'])
        else:
            raise ValueError("Invalid mutation type specified.")
    
    elif config['problem_type'] == 'bin_packing':
        if 'mutation_operator_type' not in config:
            mutation_operator_type = input("Enter the mutation type: \n1: Swap\n2. Inversion\n3. Bin Packing Swap\n4. Bin Packing Inversion\n5. Bin Packing Scramble\n 6. Bin Packing Insertion\n: ")
        else:
            mutation_operator_type = config['mutation_operator_type']
        
        if mutation_operator_type == '1':
            return SwapMutation()
        elif mutation_operator_type == '2':
            return InversionMutation()
        elif mutation_operator_type == '3':
            return BinPackingSwapMutation()
        elif mutation_operator_type == '4':
            return BinPackingInversionMutation()
        elif mutation_operator_type == '5':
            return BinPackingScrambleMutation()
        elif mutation_operator_type == '6':
            return BinPackingInsertionMutation()
        else:
            raise ValueError("Invalid mutation type specified.")

def read_mutation_strategy(config):
    if 'mutation_strategy' not in config:
          mutation_strategy = input("Enter mutation strategy: \n1. BasicMutation\n2. NonUniformMutation\n3. AdaptiveMutation\n4. TriggeredHyperMutation\n5. SelfAdaptiveMutation\n: ").strip()
    else:
        mutation_strategy = config['mutation_strategy']
        
    if mutation_strategy == '1':
        if 'mutation_prob' not in config:
            mutation_prob = float(input("Enter mutation probability: ").strip())
        else:
            mutation_prob = config['mutation_prob']    
        return BasicMutation(mutation_prob)
    
    elif mutation_strategy == '2':
        if 'mutation_initial_prob' not in config: 
            mutation_initial_prob = float(input("Enter initial mutation probability: ").strip())
        else:
            mutation_initial_prob = config['mutation_initial_prob']
        
        if 'mutation_decay_rate' not in config:
            mutation_decay_rate = float(input("Enter decay rate: ").strip())
        else:
            mutation_decay_rate = config['mutation_decay_rate']
        return NonUniformMutation(mutation_initial_prob, mutation_decay_rate)
    
    elif mutation_strategy == '3':
        if 'mutation_base_prob' not in config:
            mutation_base_prob = float(input("Enter base mutation probability: ").strip())
        else:
            mutation_base_prob = config['mutation_base_prob']
        return AdaptiveMutation(mutation_base_prob)  
    
    elif mutation_strategy == '4':
        if 'mutation_base_prob' not in config:
            mutation_base_prob = float(input("Enter base mutation probability: ").strip())
        else:
            mutation_base_prob = config['mutation_base_prob']
        
        if 'thm_threshold' not in config:
            thm_threshold = float(input("Enter threshold: ").strip())
        else:
            thm_threshold = config['thm_threshold']
        
        if 'mutation_max_prob' not in config:
            mutation_max_prob = float(input("Enter maximum mutation probability: ").strip())
        else:
            mutation_max_prob = config['mutation_max_prob']
        
        return TriggeredHyperMutation(mutation_base_prob, thm_threshold, mutation_max_prob)
    
    elif mutation_strategy == '5':
        return SelfAdaptiveMutation()
    else:
        raise ValueError(f"Unknown mutation strategy: {mutation_strategy}")    

def read_linear_scaling(config):
    if 'scaling_strategy' not in config:
        scaling_strategy = input("Enter the scaling strategy (1: Constant Factor Scaling, 2: Dynamic Scaling): ")
    else:
        scaling_strategy = config['scaling_strategy']
    
    if scaling_strategy == '1':
        if 'k' not in config:
            k = input("Enter the scaling factor k (default 2.0): ")
            if not k:
                k = 2.0
            else:
                k = float(k)
        else:
            k = config['k']

        return ConstantLinearScaling(k)
    
    elif scaling_strategy == '2':
        if 'initial_k' not in config: 
            initial_k = input("Enter the initial k value (default 2.0): ")
            if not initial_k:
                initial_k = 2.0
            else:
                initial_k = float(initial_k)
        else:
            initial_k = config['initial_k']

        if 'increment_k' not in config: 
            increment_k = input("Enter the increment value (default 0.1): ")
            if not increment_k:
                increment_k = 0.1
            else:
                increment_k = float(increment_k)
        else:
            increment_k = config['increment_k']

        if 'max_k' not in config:
            max_k = input("Enter the maximum k value (default 3.0): ")
            if not max_k:
                max_k = 3.0
            else:
                max_k = float(max_k)
        else:
            max_k = config['max_k']

        return DynamicLinearScaling(initial_k, increment_k, max_k)
    else:
        raise ValueError("Invalid scaling strategy specified.")

def read_parent_selection(config):
    if 'parent_selection_method' not in config:
        parent_selection_method = input("Enter the parent selection method (1: RWS Linear Scaling, 2: SUS Linear Scaling, 3: RWS Ranking, 4: Tournament): ")
    else:
        parent_selection_method = config['parent_selection_method']


    if parent_selection_method == '1':
        linear_strategy = read_linear_scaling(config)
        return RWSLinearScaling(linear_strategy)

    elif parent_selection_method == '2':
        linear_strategy = read_linear_scaling(config)
        return SUSLinearScaling(linear_strategy)

    elif parent_selection_method == '3':
        linear_strategy = read_linear_scaling(config)
        return RWSRankingSelection(linear_strategy)
    
    elif parent_selection_method == '4':
        if 'tournament_size' not in config:
            tournament_size = input("Enter the tournament size (default 2): ")
            if not tournament_size:
                tournament_size = 2
            else:
                tournament_size = int(tournament_size)
        else:
            tournament_size = config['tournament_size']

        if 'tournament_prob_best' not in config:
            tournament_prob_best = input("Enter tournament best probability: (default 1): ")
            if not tournament_prob_best:
                tournament_prob_best = 1.0
            else:
                tournament_prob_best = float(tournament_prob_best)
        else:
            tournament_prob_best = config['tournament_prob_best']

        return TournamentSelection(tournament_size, tournament_prob_best)

    else:
        raise ValueError("Invalid parent selection method specified.")

def read_survivor_selection(config):
    if 'survivor_selection_method' not in config:
        survivor_selection_method = input("Enter the survivor selection method (1: Elitism, 2: Tournament, 3: Aging): ")
    else:
        survivor_selection_method = config['survivor_selection_method']


    if survivor_selection_method == '1':
        return ElitismSelection()
    
    elif survivor_selection_method == '2':
        if 'survivor_tournament_size' not in config:
            survivor_tournament_size = input("Enter the tournament size (default 2): ")
            if not survivor_tournament_size:
                survivor_tournament_size = 2
            else:
                survivor_tournament_size = int(survivor_tournament_size)
        else:
            survivor_tournament_size = config['survivor_tournament_size']
        
        return survivor_tournament_selection(survivor_tournament_size)
    
    elif survivor_selection_method == '3':
        if 'max_age' not in config:
            max_age = input("Enter the maximum age (default 10): ")
            if not max_age:
                max_age = 10
            else:
                max_age = int(max_age)
        else:
            max_age = config['max_age']

        return AgingSelection(max_age)
    
    else:
        raise ValueError("Invalid survivor selection method specified.")

def read_problem(config):
    if config['problem_type'] == 'string_matching':
        if 'target_string' not in config:
            target_string = input("Enter the target string: ")
        else:
            target_string = config['target_string']
        
        return StringMatchingProblem(target_string)
    
    elif config['problem_type'] == 'sudoku':
        sudoku_problems = parse_sudoku_configurations("./data/sudoku_problems.txt")
        if 'chosen_problem' not in config:
            chosen_problem = input("Enter the index of the problem to solve: ")
        else:
            chosen_problem = config['chosen_problem']

        # check if the index is valid
        if not 0 <= int(chosen_problem) < len(sudoku_problems):
            raise ValueError("Invalid problem index")
        
        initial_board = sudoku_problems[int(chosen_problem)]
        config['fixed_positions'] = initial_board
        return SudokuProblem(initial_board)
    
    elif config['problem_type'] == 'bin_packing':
        bin_packing_problems = parse_bin_packing_config("./data/bin_packing_problems.txt")
        if 'chosen_problem' not in config:
            chosen_problem = input("Enter the index of the problem to solve: ")
        else:
            chosen_problem = config['chosen_problem']

        # check if the index is valid
        if not 0 <= int(chosen_problem) < len(bin_packing_problems):
            raise ValueError("Invalid problem index")

        items, bin_capacity = bin_packing_problems[int(chosen_problem)]
        return BinPackingProblem(items, bin_capacity)
    else:
        raise ValueError("Invalid problem type")


def read_fitness_sharing(config):
    config['fitness_sharing_type'] = config.get('fitness_sharing_type', input("Enter fitness sharing type (1: basic): ") or '1')
    
    if config['fitness_sharing_type'] == '1':  # basic
        config['sigma_share'] = config.get('sigma_share', float(input("Enter sigma_share for basic fitness sharing: ") or 0.1))
        config['alpha'] = config.get('alpha', float(input("Enter alpha for basic fitness sharing (default 1.0): ") or 1.0))
        return BasicFitnessSharing(config['sigma_share'], config['distance_func'], config['alpha'])
    else:
        raise ValueError(f"Unknown fitness sharing type: {config['fitness_sharing_type']}")






def read_speciation(config):
    config['speciation_type'] = config.get('speciation_type', input("Enter speciation type (1: threshold 2: k-means 3: silhouette k-means): ") or '1')

    if config['speciation_type'] == '1':  # Threshold
        config['similarity_threshold'] = config.get('similarity_threshold', float(input("Enter similarity threshold for threshold speciation: ") or 0.5))
        return ThresholdSpeciation(config['similarity_threshold'], config['distance_func'])

    elif config['speciation_type'] == '2':  # K-Means
        config['num_clusters'] = config.get('num_clusters', int(input("Enter the number of clusters for K-Means speciation: ") or 3))
        return KMeansSpeciation(config['num_clusters'], config['distance_func'])

    elif config['speciation_type'] == '3':  # Silhouette K-Means
        config['min_clusters'] = config.get('min_clusters', int(input("Enter the minimum number of clusters for Silhouette K-Means speciation: ") or 2))
        config['max_clusters'] = config.get('max_clusters', int(input("Enter the maximum number of clusters for Silhouette K-Means speciation: ") or 10))
        return SilhouetteKMeansSpeciation(config['min_clusters'], config['max_clusters'], config['distance_func'])

    else:
        raise ValueError(f"Unknown speciation type: {config['speciation_type']}")



def read_crowding(config):
    config['crowding_type'] = config.get('crowding_type', input("Enter crowding type (1: deterministic, 2: non_deterministic): ") or '1')
    
    if config['crowding_type'] == '1':  # deterministic
        return DeterministicCrowding(config['distance_func'])
    elif config['crowding_type'] == '2':  # non_deterministic
        return NonDeterministicCrowding(config['distance_func'])
    else:
        raise ValueError(f"Unknown crowding type: {config['crowding_type']}")





def read_config(config):
    if "problem_type" not in config:
        problem_type = input("Enter the problem type (1: String Matching, 2: Sudoku, 3: Bin Packing): ")
    else:
        problem_type = config['problem_type']

    if problem_type == '1':
        config['problem_type'] = 'string_matching'
    elif problem_type == '2':
        config['problem_type'] = 'sudoku'
    elif problem_type == '3':
        config['problem_type'] = 'bin_packing'

    config['problem'] = read_problem(config)
    config['distance_func'] = config['problem'].distance

    config['fitness_function'] = read_fitness_function(config)
    config['parent_selection'] = read_parent_selection(config)
    config['crossover_operator'] = read_crossover(config)
    config['mutation_operator'] = read_mutation_operator(config)
    config['mutation_strategy'] = read_mutation_strategy(config)
    config['survivor_selection'] = read_survivor_selection(config)
    config['data_collector'] = read_data_collector(config)
    config['fitness_sharing'] = read_fitness_sharing(config)
    config['speciation'] = read_speciation(config)
    config['crowding'] = read_crowding(config)

    if 'population_size' not in config:
        population_size = input("Enter the population size (default 100): ")
        config['population_size'] = int(population_size) if population_size else 100

    if 'num_generations' not in config:
        num_generations = input("Enter the number of generations (default 100): ")
        config['num_generations'] = int(num_generations) if num_generations else 100

    config['initialize_population'] = lambda size: config['problem'].initialize_population(size)

    return config




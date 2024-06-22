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


from .parsers import parse_bin_packing_config, parse_sudoku_configurations



def read_crossover(config):
    crossover_method = input("Enter the crossover method (1: Single Point, 2: Two Point, 3: Uniform, 4: Sudoku Subgrid, 5: Sudoku Row/Column, 6: Sudoku CX, 7: Bin Packing CX, 8: Sudoku PMX, 9: Bin Packing PMX): ")

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

    convergence_threshold = input("Enter the convergence threshold (or leave blank for default 0.01):")
    if not convergence_threshold:
        convergence_threshold = 0.01
    else:
        convergence_threshold = float(convergence_threshold)

    convergence_window = input("Enter the convergence window (or leave blank for default 10):")
    if not convergence_window:
        convergence_window = 10
    else:
        convergence_window = int(convergence_window)

    return DataCollector(global_min_fitness, global_max_fitness, distance_method, distance_unit, convergence_threshold, convergence_window)

def read_fitness_function(config):
    if config['problem_type'] == 'string_matching':
        chosen_fitness = input("Enter the fitness function (1: MatchFitness): ")
        if chosen_fitness == '1':
            return MatchFitness(config['problem'].target_string)
        else:
            raise ValueError("Invalid fitness function specified.")

    elif config['problem_type'] == 'sudoku':
        chosen_fitness = input("Enter the fitness function (1: SudokuFitness): ")
        if chosen_fitness == '1':
            return SudokuFitness()
        else:
            raise ValueError("Invalid fitness function specified.")
    elif config['problem_type'] == 'bin_packing':
        chosen_fitness = input("Enter the fitness function (1: CombinedBinPackingFitness): ")
        if chosen_fitness == '1':
            weight_bins_used = input("Enter the weight for bins used (default 1.0): ")
            weight_bins_used = float(weight_bins_used) if weight_bins_used else 1.0
            weight_total_waste = input("Enter the weight for total waste (default 1.0): ")
            weight_total_waste = float(weight_total_waste) if weight_total_waste else 1.0
            weight_load_balance = input("Enter the weight for load balance (default 1.0): ")
            weight_load_balance = float(weight_load_balance) if weight_load_balance else 1.0
            return CombinedBinPackingFitness(bin_capacity=config['bin_capacity'], 
                                             weight_bins_used=weight_bins_used, 
                                             weight_total_waste=weight_total_waste, 
                                             weight_load_balance=weight_load_balance)
    else:
        raise ValueError("Invalid problem type specified.")



def read_mutation(config):
    mutation_rate = input("Enter the mutation rate (default 0.01): ")
    if not mutation_rate:
        mutation_rate = 0.01
    else:
        mutation_rate = float(mutation_rate)

    config['mutation_rate'] = mutation_rate

    if config['problem_type'] == 'string_matching':
        mutation_type = input("Enter the mutation type (1: Swap 2. Inversion 3. Single Point ): ")
        if mutation_type == '1':
            return SwapMutation()
        elif mutation_type == '2':
            return InversionMutation()
        elif mutation_type == '3':
            return SinglePointMutation()
        else:
            raise ValueError("Invalid mutation type specified.")
    
    elif config['problem_type'] == 'sudoku':
        mutation_type = input("Enter the mutation type: \n1. Sudoku Swap\n2. Sudoku Inversion\n3. Sudoku Scramble\n4. Sudoku Row Column Swap\n5. Sudoku Random Resetting\n6. SudokuDisplacement\n: ")
        if mutation_type == '1':
            return SudokuSwapMutation(config['fixed_positions'])
        elif mutation_type == '2':
            return SudokuInversionMutation(config['fixed_positions'])
        elif mutation_type == '3':
            return SudokuScrambleMutation(config['fixed_positions'])
        elif mutation_type == '4':
            return SudokuRowColumnSwapMutation(config['fixed_positions'])
        elif mutation_type == '5':
            return SudokuRandomResettingMutation(config['fixed_positions'])
        elif mutation_type == '6':
            return SudokuDisplacementMutation(config['fixed_positions'])
        else:
            raise ValueError("Invalid mutation type specified.")
    
    elif config['problem_type'] == 'bin_packing':
        mutation_type = input("Enter the mutation type: \n1: Swap\n2. Inversion\n3. Bin Packing Swap\n4. Bin Packing Inversion\n5. Bin Packing Scramble\n 6. Bin Packing Insertion\n: ")
        if mutation_type == '1':
            return SwapMutation()
        elif mutation_type == '2':
            return InversionMutation()
        elif mutation_type == '3':
            return BinPackingSwapMutation()
        elif mutation_type == '4':
            return BinPackingInversionMutation()
        elif mutation_type == '5':
            return BinPackingScrambleMutation()
        elif mutation_type == '6':
            return BinPackingInsertionMutation()
        else:
            raise ValueError("Invalid mutation type specified.")


def read_linear_scaling(config):
    scaling_strategy = input("Enter the scaling strategy (1: Constant Factor Scaling, 2: Dynamic Scaling): ")
    if scaling_strategy == '1':
        k = input("Enter the scaling factor k (default 2.0): ")
        if not k:
            k = 2.0
        else:
            k = float(k)
        return ConstantLinearScaling(k)
    
    elif scaling_strategy == '2':
        initial_k = input("Enter the initial k value (default 2.0): ")
        if not initial_k:
            initial_k = 2.0
        else:
            initial_k = float(initial_k)
        increment = input("Enter the increment value (default 0.1): ")
        if not increment:
            increment = 0.1
        else:
            increment = float(increment)
        max_k = input("Enter the maximum k value (default 3.0): ")
        if not max_k:
            max_k = 3.0
        else:
            max_k = float(max_k)
            return DynamicLinearScaling(initial_k, increment, max_k)
        
    else:
        raise ValueError("Invalid scaling strategy specified.")

def read_parent_selection(config):
    parent_selection_method = input("Enter the parent selection method (1: RWS Linear Scaling, 2: SUS Linear Scaling, 3: RWS Ranking, 4: Tournament): ")

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
        tournament_size = input("Enter the tournament size (default 2): ")
        if not tournament_size:
            tournament_size = 2
        else:
            tournament_size = int(tournament_size)
        return TournamentSelection(tournament_size)

    else:
        raise ValueError("Invalid parent selection method specified.")



def read_survivor_selection(config):
    survivor_selection_method = input("Enter the survivor selection method (1: Elitism, 2: Tournament, 3: Aging): ")

    if survivor_selection_method == '1':
        return ElitismSelection()
    
    elif survivor_selection_method == '2':
        tournament_size = input("Enter the tournament size (default 2): ")
        if not tournament_size:
            tournament_size = 2
        else:
            tournament_size = int(tournament_size)
        return survivor_tournament_selection(tournament_size)
    
    elif survivor_selection_method == '3':
        max_age = input("Enter the maximum age (default 10): ")
        if not max_age:
            max_age = 10
        else:
            max_age = int(max_age)
        return AgingSelection(max_age)
    
    else:
        raise ValueError("Invalid survivor selection method specified.")










def read_problem(config):
    if config['problem_type'] == 'string_matching':
        target_string = input("Enter the target string: ")
        return StringMatchingProblem(target_string)
    
    elif config['problem_type'] == 'sudoku':
        sudoku_problems = parse_sudoku_configurations("./sudoku_problems.txt")
        chosen_problem = input("Enter the index of the problem to solve: ")

        # check if the index is valid
        if not 0 <= int(chosen_problem) < len(sudoku_problems):
            raise ValueError("Invalid problem index")
        
        initial_board = sudoku_problems[int(chosen_problem)]
        return SudokuProblem(initial_board)
    
    elif config['problem_type'] == 'bin_packing':
        bin_packing_problems = parse_bin_packing_config("./bin_packing_problems.txt")
        chosen_problem = input("Enter the index of the problem to solve: ")

        # check if the index is valid
        if not 0 <= int(chosen_problem) < len(bin_packing_problems):
            raise ValueError("Invalid problem index")

        items, bin_capacity = bin_packing_problems[int(chosen_problem)]
        return BinPackingProblem(items, bin_capacity)
    else:
        raise ValueError("Invalid problem type")












from .modules.GAComparison import GAComparison
from .utils.readers import read_config


configs = [
    {
        'name': "config1",
        'problem_type': 'string_matching',
        'target_string': 'HELLO WORLD',
        'chosen_fitness': '1',  # MatchFitness
        'parent_selection_method': '1',  # Tournament Selection
        'tournament_size': 3,
        'crossover_method': '1',  # Single Point Crossover
        'mutation_operator_type': '1',  # Swap Mutation
        'mutation_strategy': '1',  # Basic Mutation
        'mutation_prob': 0.01,
        'survivor_selection_method': '1',  # Elitism Selection
        'population_size': 100,
        'num_generations': 100,
        'convergence_threshold': 0.01,
        'convergence_window': 10,
        'tournament_prob_best': 1.0,
        'plot_results': False,
        'terminal_log': False,
        'scaling_strategy': '1',
        'k': 2.0
    },
    {
        'name': "config2",
        'problem_type': 'string_matching',
        'target_string': 'HELLO WORLD',
        'chosen_fitness': '1',  # MatchFitness
        'parent_selection_method': '2',  # Roulette Wheel Selection
        'crossover_method': '2',  # Two Point Crossover
        'mutation_operator_type': '2',  # Inversion Mutation
        'mutation_strategy': '2',  # Non-Uniform Mutation
        'mutation_initial_prob': 0.05,
        'mutation_decay_rate': 0.01,
        'survivor_selection_method': '1',  # Elitism Selection
        'population_size': 100,
        'num_generations': 100,
        'convergence_threshold': 0.01,
        'convergence_window': 10,
        'tournament_prob_best': 1.0,
        'plot_results': False,
        'terminal_log': False,
        'scaling_strategy': '1',
        'k': 2.0
    },
    {
        'name': "config3",
        'problem_type': 'string_matching',
        'target_string': 'HELLO WORLD',
        'chosen_fitness': '1',  # MatchFitness
        'parent_selection_method': '3',  # RWS Ranking Selection
        'crossover_method': '3',  # Uniform Crossover
        'mutation_operator_type': '3',  # Single Point Mutation
        'mutation_strategy': '3',  # Adaptive Mutation
        'mutation_base_prob': 0.05,
        'survivor_selection_method': '1',  # Elitism Selection
        'population_size': 100,
        'num_generations': 100,
        'convergence_threshold': 0.01,
        'convergence_window': 10,
        'tournament_prob_best': 1.0,
        'plot_results': False,
        'terminal_log': False,
        'scaling_strategy': '1',
        'k': 2.0
    },
    {
        'name': "config4",
        'problem_type': 'string_matching',
        'target_string': 'HELLO WORLD',
        'chosen_fitness': '1',  # MatchFitness
        'parent_selection_method': '4',  # Tournament Selection
        'tournament_size': 3,
        'crossover_method': '1',  # Single Point Crossover
        'mutation_operator_type': '1',  # Swap Mutation
        'mutation_strategy': '4',  # Triggered Hyper Mutation
        'mutation_base_prob': 0.1,
        'thm_threshold': 0.01,
        'mutation_max_prob': 0.9,
        'survivor_selection_method': '1',  # Elitism Selection
        'population_size': 100,
        'num_generations': 100,
        'convergence_threshold': 0.01,
        'convergence_window': 10,
        'tournament_prob_best': 1.0,
        'plot_results': False,
        'terminal_log': False,
        'scaling_strategy': '1',
        'k': 2.0
    },
    {
        'name': "config5",
        'problem_type': 'string_matching',
        'target_string': 'HELLO WORLD',
        'chosen_fitness': '1',  # MatchFitness
        'parent_selection_method': '4',  # Tournament Selection
        'tournament_size': 3,
        'crossover_method': '2',  # Two Point Crossover
        'mutation_operator_type': '2',  # Inversion Mutation
        'mutation_strategy': '5',  # Self-Adaptive Mutation
        'survivor_selection_method': '1',  # Elitism Selection
        'population_size': 100,
        'num_generations': 100,
        'convergence_threshold': 0.01,
        'convergence_window': 10,
        'tournament_prob_best': 1.0,
        'plot_results': False,
        'terminal_log': False,
        'scaling_strategy': '1',
        'k': 2.0
    }
]



for i, config in enumerate(configs):
    configs[i] = read_config(config)


comparitor = GAComparison(configs)

comparitor.run()
comparitor.plot_results()


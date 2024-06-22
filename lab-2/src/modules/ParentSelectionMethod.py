from abc import ABC, abstractmethod
import random

class ParentSelectionMethod(ABC):
    @abstractmethod
    def select(self, population, fitnesses, num_parents):
        """
        Abstract method to select parents from the population based on their fitness.
        
        :param population: The current population of individuals.
        :param fitnesses: A list of fitness values corresponding to the population.
        :param num_parents: The number of parents to select.
        :return: A list of selected parents.
        """
        pass



class RWSLinearScaling(ParentSelectionMethod):
    def __init__(self, scaling_strategy):
        self.scaling_strategy = scaling_strategy

    def select(self, population, fitnesses, num_parents):
        scaled_fitnesses = self.scaling_strategy.scale(fitnesses)
        total_fitness = sum(scaled_fitnesses)
        selection_probs = [f / total_fitness for f in scaled_fitnesses]
        
        selected_parents = random.choices(population, weights=selection_probs, k=num_parents)
        return selected_parents


class SUSLinearScaling(ParentSelectionMethod):
    def __init__(self, scaling_strategy):
        self.scaling_strategy = scaling_strategy

    def select(self, population, fitnesses, num_parents):
        scaled_fitnesses = self.scaling_strategy.scale(fitnesses)
        total_fitness = sum(scaled_fitnesses)
        selection_probs = [f / total_fitness for f in scaled_fitnesses]
        
        pointers = [(i + random.random()) / num_parents for i in range(num_parents)]
        selected_parents = []
        cumulative_sum = 0
        j = 0
        for i, individual in enumerate(population):
            cumulative_sum += selection_probs[i]
            while j < num_parents and cumulative_sum > pointers[j]:
                selected_parents.append(individual)
                j += 1
        return selected_parents

class RWSRankingSelection(ParentSelectionMethod):
    def __init__(self, scaling_strategy):
        self.scaling_strategy = scaling_strategy

    def select(self, population, fitnesses, num_parents):
        # Rank the individuals based on fitness
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        ranks = list(range(1, len(population) + 1))
        
        # Apply linear scaling to ranks
        scaled_ranks = self.scaling_strategy.scale(ranks)
        total_rank = sum(scaled_ranks)
        selection_probs = [rank / total_rank for rank in scaled_ranks]
        
        # Select parents based on scaled ranks
        selected_parents = random.choices(sorted_population, weights=selection_probs, k=num_parents)
        return selected_parents



class TournamentSelection(ParentSelectionMethod):
    def __init__(self, scaling_strategy, tournament_size, probability_best=1.0):
        self.scaling_strategy = scaling_strategy
        self.tournament_size = tournament_size
        self.probability_best = probability_best

    def select(self, population, fitnesses, num_parents):
        scaled_fitnesses = self.scaling_strategy.scale(fitnesses)
        selected_parents = []
        for _ in range(num_parents):
            tournament = random.sample(list(zip(population, scaled_fitnesses)), self.tournament_size)
            tournament.sort(key=lambda x: x[1], reverse=True)
            if random.random() < self.probability_best:
                selected_parents.append(tournament[0][0])
            else:
                selected_parents.append(tournament[random.randint(1, self.tournament_size - 1)][0])
        return selected_parents



class ElitistSelection(ParentSelectionMethod):
    def __init__(self):
        pass

    def select(self, population, fitnesses, num_parents):
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        selected_parents = sorted_population[:num_parents]
        return selected_parents




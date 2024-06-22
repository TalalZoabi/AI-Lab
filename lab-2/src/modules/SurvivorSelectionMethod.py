from abc import ABC, abstractmethod
import random

class SurvivorSelectionMethod(ABC):
    @abstractmethod
    def select(self, population, fitnesses, num_survivors):
        """
        Abstract method to select survivors from the population based on their fitness.
        
        :param population: The current population of individuals.
        :param fitnesses: A list of fitness values corresponding to the population.
        :param num_survivors: The number of survivors to select.
        :return: A list of selected survivors.
        """
        pass



class ElitismSelection(SurvivorSelectionMethod):
    def select(self, population, fitnesses, num_survivors, ages=None):
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        return sorted_population[:num_survivors], ages

class TournamentSelection(SurvivorSelectionMethod):
    def __init__(self, tournament_size):
        self.tournament_size = tournament_size

    def select(self, population, fitnesses, num_survivors):
        selected_survivors = []
        for _ in range(num_survivors):
            tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected_survivors.append(winner)
        return selected_survivors


class AgingSelection(SurvivorSelectionMethod):
    def __init__(self, max_age):
        self.max_age = max_age

    def select(self, population, fitnesses, num_survivors, ages):
        # Separate individuals into young and adult groups
        young_individuals = [(pop, fit, age) for pop, fit, age in zip(population, fitnesses, ages) if age == 0]
        adult_individuals = [(pop, fit, age) for pop, fit, age in zip(population, fitnesses, ages) if 0 < age <= self.max_age]

        # Sort adults by fitness in descending order
        adult_individuals.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize survivors with all young individuals
        survivors = young_individuals
        
        # Add adult individuals until reaching the desired number of survivors
        survivors += adult_individuals[:num_survivors - len(survivors)]
        
        # Extract the selected population and update ages
        selected_population = [x[0] for x in survivors[:num_survivors]]
        new_ages = [age + 1 for pop, age in zip(population, ages)]

        return selected_population, new_ages


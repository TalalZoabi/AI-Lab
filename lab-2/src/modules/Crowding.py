import random
import numpy as np

class Crowding:
    def __init__(self, distance_func):
        self.distance_func = distance_func

    def apply_crowding(self, offspring, parents, fitness):
        raise NotImplementedError("This method should be implemented by subclasses.")

class DeterministicCrowding(Crowding):
    def apply_crowding(self, offspring, parents, fitness):
        new_population = []
        for o in offspring:
            similar_parent = min(parents, key=lambda p: self.distance_func(o, p))
            if fitness[o] > fitness[similar_parent]:
                new_population.append(o)
            else:
                new_population.append(similar_parent)
        return new_population


class NonDeterministicCrowding(Crowding):
    def __init__(self, distance_func, temp=1.0):
        super().__init__(distance_func)
        self.T = temp


    def apply_crowding(self, offspring, parents, fitness):
        new_population = []
        for o in offspring:
            similar_parent = min(parents, key=lambda p: self.distance_func(o, p))
            if fitness[o] > fitness[similar_parent]:
                replacement_prob = 1 / (1 + np.exp(-(fitness[o] - fitness[similar_parent]) / self.T))
                if random.random() < replacement_prob:
                    new_population.append(o)
                else:
                    new_population.append(similar_parent)
            else:
                new_population.append(similar_parent)
        return new_population


import random 


class MutationStrategy:
    def should_mutate(self, **kwargs):
        raise NotImplementedError("Mutation strategy must implement the should_mutate method.")


import random

class BasicMutation(MutationStrategy):
    def __init__(self, mutation_prob):
        self.mutation_prob = mutation_prob

    def should_mutate(self, **kwargs):
        return random.random() < self.mutation_prob

# Sanity check
mutation_prob = 0.5
strategy = BasicMutation(mutation_prob)
print("Should mutate:", strategy.should_mutate())


class NonUniformMutation(MutationStrategy):
    def __init__(self, initial_prob, decay_rate):
        self.initial_prob = initial_prob
        self.decay_rate = decay_rate

    def should_mutate(self, generation):
        mutation_prob = self.initial_prob * (1 - self.decay_rate * generation)
        return random.random() < mutation_prob

# Sanity check
initial_prob = 0.5
decay_rate = 0.01
generation = 10
strategy = NonUniformMutation(initial_prob, decay_rate)
print("Should mutate:", strategy.should_mutate( generation=generation))

class AdaptiveMutation(MutationStrategy):
    def __init__(self, base_prob):
        self.base_prob = base_prob

    def should_mutate(self, individual_fitness, avg_fitness, **kwargs):
        mutation_prob = self.base_prob * (1 - (individual_fitness / avg_fitness))
        return random.random() < mutation_prob





class TriggeredHyperMutation(MutationStrategy):
    def __init__(self, base_prob, threshold, max_prob):
        self.base_prob = base_prob
        self.threshold = threshold
        self.max_prob = max_prob
        self.hyper_mutation = False
        self.prev_best_fitness = None

    def should_mutate(self, best_fitness, **kwargs):
        if self.prev_best_fitness is None:
            self.prev_best_fitness = best_fitness
            return False
        if not self.hyper_mutation:
            # Check for stagnation
            if best_fitness - self.prev_best_fitness < self.threshold:
                self.hyper_mutation = True
                self.current_prob = self.max_prob
            else:
                self.current_prob = self.base_prob
        else:
            # Check if significant improvement has been made
            if best_fitness - self.prev_best_fitness >= self.threshold:
                self.hyper_mutation = False
                self.current_prob = self.base_prob

        self.prev_best_fitness = best_fitness

        return random.random() < self.current_prob


class AdaptiveMutation(MutationStrategy):
    def __init__(self, base_prob):
        self.base_prob = base_prob

    def should_mutate(self, individual_fitness, avg_fitness, **kwargs):
        mutation_prob = self.base_prob * (1 - (individual_fitness / avg_fitness))
        return random.random() < mutation_prob


class SelfAdaptiveMutation(MutationStrategy):
    def should_mutate(self, individual_fitness, min_fitness, max_fitness, **kwargs):
        relative_fitness = (individual_fitness - min_fitness) / (max_fitness - min_fitness)
        mutation_prob = max(0, 1 - relative_fitness)
        return random.random() < mutation_prob



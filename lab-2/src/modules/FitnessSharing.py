
class FitnessSharing:
    def __init__(self, sigma_share, distance_func, alpha=1.0):
        self.sigma_share = sigma_share
        self.alpha = alpha
        self.distance_func = distance_func

    def apply_sharing(self, population, raw_fitness):
        raise NotImplementedError("This method should be implemented by subclasses.")


class BasicFitnessSharing(FitnessSharing):
    def apply_sharing(self, population, raw_fitness):
        shared_fitness = raw_fitness.copy()
        for i, ind_i in enumerate(population):
            sharing_sum = sum(self.sharing_function(ind_i, ind_j) for ind_j in population)
            shared_fitness[i] = raw_fitness[i] / sharing_sum
        return shared_fitness

    def sharing_function(self, ind_i, ind_j):
        distance = self.distance_func(ind_i, ind_j)
        if distance < self.sigma_share:
            return 1 - (distance / self.sigma_share) ** self.alpha
        return 0




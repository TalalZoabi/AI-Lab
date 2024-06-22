from abc import ABC, abstractmethod

class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, individual):
        """
        Abstract method to evaluate the fitness of an individual.
        
        :param individual: The individual solution to evaluate.
        :return: The fitness score of the individual.
        """
        pass
    
    @abstractmethod
    def max_fitness(self, *args, **kwargs):
        """
        Abstract method to return the maximum possible fitness score.
        
        :return: The maximum possible fitness score.
        """
        pass

    @abstractmethod
    def min_fitness(self, *args, **kwargs):
        """
        Abstract method to return the minimum possible fitness score.
        
        :return: The minimum possible fitness score.
        """
        pass

class MatchFitness(FitnessFunction):
    def __init__(self, target_string):
        self.target_string = target_string

    def evaluate(self, individual):
        if len(individual) != len(self.target_string):
            raise ValueError("Individual length must match target string length.")
        
        matches = sum(1 for i, j in zip(individual, self.target_string) if i == j)
        return matches
    
    def max_fitness(self, *args, **kwargs):
        return len(self.target_string)
    
    def min_fitness(self, *args, **kwargs):
        return 0
    

class BullseyeFitness(FitnessFunction):
    def __init__(self, target_string, bonus=100):
        self.target_string = target_string
        self.bonus = bonus

    def evaluate(self, individual):
        if len(individual) != len(self.target_string):
            raise ValueError("Individual length must match target string length.")
        
        if individual == self.target_string:
            return len(self.target_string) + self.bonus

        target_chars = list(self.target_string)
        matches = 0
        for char in individual:
            if char in target_chars:
                matches += 1
                target_chars.remove(char)
        
        return matches
    
    def max_fitness(self, *args, **kwargs):
        return len(self.target_string) + self.bonus
    
    def min_fitness(self, *args, **kwargs):
        return 0

class SudokuFitness(FitnessFunction):
    def __init__(self):
        pass

    def evaluate(self, individual):
        grid = self.convert_to_grid(individual)
        row_conflicts = self.count_row_conflicts(grid)
        column_conflicts = self.count_column_conflicts(grid)
        subgrid_conflicts = self.count_subgrid_conflicts(grid)
        
        total_conflicts = row_conflicts + column_conflicts + subgrid_conflicts
        max_conflicts = 3 * 9 * (9 - 1)  # Maximum possible conflicts
        
        fitness = max_conflicts - total_conflicts
        return fitness

    def convert_to_grid(self, individual):
        return [individual[i*9:(i+1)*9] for i in range(9)]

    def count_row_conflicts(self, grid):
        conflicts = 0
        for row in grid:
            conflicts += len(row) - len(set(row))
        return conflicts

    def count_column_conflicts(self, grid):
        conflicts = 0
        for col in range(9):
            column = [grid[row][col] for row in range(9)]
            conflicts += len(column) - len(set(column))
        return conflicts

    def count_subgrid_conflicts(self, grid):
        conflicts = 0
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                subgrid = [grid[r][c] for r in range(row, row + 3) for c in range(col, col + 3)]
                conflicts += len(subgrid) - len(set(subgrid))
        return conflicts
    
    def max_fitness(self, *args, **kwargs):
        return 3 * 9 * (9 - 1)
    
    def min_fitness(self, *args, **kwargs):
        return 0


class CombinedBinPackingFitness(FitnessFunction):
    def __init__(self, bin_capacity, weight_bins_used=1.0, weight_total_waste=1.0, weight_load_balance=1.0):
        self.bin_capacity = bin_capacity
        self.weight_bins_used = weight_bins_used
        self.weight_total_waste = weight_total_waste
        self.weight_load_balance = weight_load_balance

    def evaluate(self, individual):
        bins = self.pack_bins(individual)
        num_bins_used = len(bins)
        total_waste = sum(self.bin_capacity - sum(bin) for bin in bins)
        bin_loads = [sum(bin) for bin in bins]
        load_balance = max(bin_loads) - min(bin_loads) if bin_loads else 0

        # Combine the factors into a single fitness score
        fitness = (self.weight_bins_used * num_bins_used +
                   self.weight_total_waste * total_waste +
                   self.weight_load_balance * load_balance)

        # Since we want to minimize these factors, return the negative fitness value for maximization
        return -fitness

    def pack_bins(self, individual):
        bins = []
        for item in individual:
            placed = False
            for bin in bins:
                if sum(bin) + item <= self.bin_capacity:
                    bin.append(item)
                    placed = True
                    break
            if not placed:
                bins.append([item])
        return bins
    
    def min_fitness(self, *args, **kwargs):
        # The worst-case fitness (max number of bins and maximum waste)
        items = kwargs.get('items', [])
        max_bins_used = len(items)  # Each item in its own bin
        max_total_waste = max_bins_used * self.bin_capacity - sum(items)
        return -(self.weight_bins_used * max_bins_used + self.weight_total_waste * max_total_waste + self.weight_load_balance * (self.bin_capacity - min(items)))

    def max_fitness(self, *args, **kwargs):
        # The best-case fitness (minimum number of bins and minimum waste)
        items = kwargs.get('items', [])
        min_bins_used = sum(items) // self.bin_capacity + (1 if sum(items) % self.bin_capacity > 0 else 0)
        min_total_waste = self.bin_capacity - min(items)
        return -(self.weight_bins_used * min_bins_used + self.weight_total_waste * min_total_waste + self.weight_load_balance * 0)





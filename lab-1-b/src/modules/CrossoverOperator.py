from abc import ABC, abstractmethod
import random

class CrossoverOperator(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        """
        Abstract method to crossover two parents to produce offspring.
        :param parent1: The first parent individual.
        :param parent2: The second parent individual.
        :return: A list of offspring individuals.
        """
        pass


class SinglePointCrossover(CrossoverOperator):
    def crossover(self, parent1, parent2):
        import random
        point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return [offspring1, offspring2]




class TwoPointCrossover(CrossoverOperator):
    def crossover(self, parent1, parent2):
        point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return [offspring1, offspring2]



class UniformCrossover(CrossoverOperator):
    def crossover(self, parent1, parent2):
        offspring1 = []
        offspring2 = []
        for i in range(len(parent1)):
            if random.random() > 0.5:
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])
            else:
                offspring1.append(parent2[i])
                offspring2.append(parent1[i])
        return [''.join(offspring1), ''.join(offspring2)]





class SudokuSubgridCrossover(CrossoverOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def crossover(self, parent1, parent2):
        grid1 = self.convert_to_grid(parent1)
        grid2 = self.convert_to_grid(parent2)
        offspring1 = [row[:] for row in grid1]
        offspring2 = [row[:] for row in grid2]

        subgrid_idx = random.randint(0, 8)
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3

        for i in range(3):
            for j in range(3):
                if not self.fixed_positions[row_start + i][col_start + j]:
                    offspring1[row_start + i][col_start + j] = grid2[row_start + i][col_start + j]
                    offspring2[row_start + i][col_start + j] = grid1[row_start + i][col_start + j]

        return [self.convert_to_list(offspring1), self.convert_to_list(offspring2)]

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]




class SudokuRowColumnCrossover(CrossoverOperator):
    def crossover(self, parent1, parent2):
        grid1 = self.convert_to_grid(parent1)
        grid2 = self.convert_to_grid(parent2)
        offspring1 = [row[:] for row in grid1]
        offspring2 = [row[:] for row in grid2]

        if random.choice([True, False]):
            # Row crossover
            row_idx = random.randint(0, 8)
            for col in range(9):
                if not self.fixed_positions[row_idx][col]:
                    offspring1[row_idx][col] = grid2[row_idx][col]
                    offspring2[row_idx][col] = grid1[row_idx][col]
        else:
            # Column crossover
            col_idx = random.randint(0, 8)
            for row in range(9):
                if not self.fixed_positions[row][col_idx]:
                    offspring1[row][col_idx] = grid2[row][col_idx]
                    offspring2[row][col_idx] = grid1[row][col_idx]

        return [self.convert_to_list(offspring1), self.convert_to_list(offspring2)]

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]



class SudokuCXCrossover(CrossoverOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def cycle_crossover(self, parent1_genome, parent2_genome):
        size = len(parent1_genome)
        child_genome = [None] * size
        cycles = [0] * size
        cycle_num = 1
        
        # Loop through each gene to identify cycles
        for i in range(size):
            if cycles[i] == 0:
                start = i
                while cycles[start] == 0:
                    cycles[start] = cycle_num
                    start = parent1_genome.index(parent2_genome[start])
                cycle_num += 1
        
        # Fill the child genome using the identified cycles
        for i in range(size):
            if cycles[i] % 2 == 1:
                child_genome[i] = parent1_genome[i]
            else:
                child_genome[i] = parent2_genome[i]
        
        return child_genome
    
    
    def crossover(self, parent1, parent2):
        offspring1 = self.cycle_crossover(parent1, parent2)
        offspring2 = self.cycle_crossover(parent2, parent1)
        return [offspring1, offspring2]

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]



class BinPackingCXCrossover(CrossoverOperator):
    def cycle_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        cycles = [0] * size
        cycle_num = 1

        for i in range(size):
            if cycles[i] == 0:
                start = i
                while cycles[start] == 0:
                    cycles[start] = cycle_num
                    start_value = parent1[start]
                    if start_value in parent2:
                        start = parent2.index(start_value)
                    else:
                        break
                cycle_num += 1

        for i in range(size):
            if cycles[i] % 2 == 1:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        # Fill in any remaining None values with items from the other parent
        for i in range(size):
            if child[i] is None:
                if parent2[i] not in child:
                    child[i] = parent2[i]
                else:
                    child[i] = parent1[i]

        return child


    def crossover(self, parent1, parent2):
        offspring1 = self.cycle_crossover(parent1, parent2)
        offspring2 = self.cycle_crossover(parent2, parent1)
        return [offspring1, offspring2]


class SudokuPMXCrossover(CrossoverOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def pmx_crossover(self, parent1_genome, parent2_genome):
        import random
        size = len(parent1_genome)
        child_genome = [None] * size
        start, end = sorted(random.sample(range(size), 2))

        for i in range(start, end):
            if not self.fixed_positions[i // 9][i % 9]:
                child_genome[i] = parent1_genome[i]

        for i in range(start, end):
            if not self.fixed_positions[i // 9][i % 9]:
                if parent2_genome[i] not in child_genome[start:end]:
                    pos = i
                    while start <= pos < end:
                        pos = parent1_genome.index(parent2_genome[pos])
                    child_genome[pos] = parent2_genome[i]

        for i in range(size):
            if child_genome[i] is None:
                child_genome[i] = parent2_genome[i]

        return child_genome
    
    def crossover(self, parent1, parent2):
        offspring1 = self.pmx_crossover(parent1, parent2)
        offspring2 = self.pmx_crossover(parent2, parent1)
        return [offspring1, offspring2]

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]


class BinPackingPMXCrossover(CrossoverOperator):
    def pmx_crossover(self, parent1, parent2):
        import random
        size = len(parent1)
        child = [None] * size
        start, end = sorted(random.sample(range(size), 2))

        # Copy the segment from parent1 to child
        for i in range(start, end):
            child[i] = parent1[i]

        # Map the rest of the positions from parent2 to child
        for i in range(start, end):
            if parent2[i] not in child[start:end]:
                pos = i
                while start <= pos < end:
                    pos = parent1.index(parent2[pos])
                child[pos] = parent2[i]

        # Fill in the remaining positions from parent2
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]

        return child

    def crossover(self, parent1, parent2):
        offspring1 = self.pmx_crossover(parent1, parent2)
        offspring2 = self.pmx_crossover(parent2, parent1)
        return [offspring1, offspring2]







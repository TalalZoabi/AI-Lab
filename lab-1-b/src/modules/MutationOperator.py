from abc import ABC, abstractmethod
import string
import random

class MutationOperator(ABC):
    @abstractmethod
    def mutate(self, candidate):
        """
        Abstract method to mutate a given candidate.
        :param candidate: An individual solution to be mutated.
        :return: A mutated individual solution.
        """
        pass

class SwapMutation(MutationOperator):
    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert string to list for mutability
        idx1, idx2 = random.sample(range(len(candidate)), 2)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        return ''.join(candidate)  # Convert back to string

class InversionMutation(MutationOperator):
    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert string to list for mutability
        start, end = sorted(random.sample(range(len(candidate)), 2))
        candidate[start:end] = reversed(candidate[start:end])
        return ''.join(candidate)  # Convert back to string


class SinglePointMutation(MutationOperator):
    def mutate(self, candidate):
        candidate = list(candidate)  # Convert string to list for mutability
        idx = random.randint(0, len(candidate) - 1)
        candidate[idx] = random.choice(string.printable)
        return ''.join(candidate)  # Convert back to string



class SudokuSwapMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        grid = self.convert_to_grid(candidate)
        
        subgrid_idx = random.randint(0, 8)
        subgrid = self.get_subgrid(grid, subgrid_idx)
        fixed_subgrid_positions = self.get_fixed_subgrid_positions(subgrid_idx)
        
        if len(fixed_subgrid_positions) < 2:
            return candidate  # No mutation if less than two mutable positions
        
        while True:
            idx1, idx2 = random.sample(range(9), 2)
            if not fixed_subgrid_positions[idx1] and not fixed_subgrid_positions[idx2]:
                subgrid[idx1], subgrid[idx2] = subgrid[idx2], subgrid[idx1]
                break
        
        self.set_subgrid(grid, subgrid_idx, subgrid)
        return self.convert_to_list(grid)

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]

    def get_subgrid(self, grid, subgrid_idx):
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3
        subgrid = []
        for i in range(3):
            subgrid.extend(grid[row_start + i][col_start:col_start + 3])
        return subgrid

    def set_subgrid(self, grid, subgrid_idx, subgrid):
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3
        idx = 0
        for i in range(3):
            for j in range(3):
                grid[row_start + i][col_start + j] = subgrid[idx]
                idx += 1
    
    def get_fixed_subgrid_positions(self, subgrid_idx):
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3
        fixed_subgrid_positions = []
        for i in range(3):
            for j in range(3):
                fixed_subgrid_positions.append(self.fixed_positions[row_start + i][col_start + j])
        return fixed_subgrid_positions


class SudokuInversionMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        grid = self.convert_to_grid(candidate)
        
        if random.choice([True, False]):
            # Row inversion
            row = random.randint(0, 8)
            if self.can_mutate_row(row):
                start, end = sorted(random.sample(range(9), 2))
                grid[row][start:end] = self.reverse_except_fixed(grid[row][start:end], self.fixed_positions[row][start:end])
        else:
            # Column inversion
            col = random.randint(0, 8)
            if self.can_mutate_column(col):
                col_values = [grid[row][col] for row in range(9)]
                start, end = sorted(random.sample(range(9), 2))
                col_values[start:end] = self.reverse_except_fixed(col_values[start:end], [self.fixed_positions[row][col] for row in range(9)][start:end])
                for row in range(9):
                    grid[row][col] = col_values[row]

        return self.convert_to_list(grid)

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]

    def can_mutate_row(self, row):
        return sum(self.fixed_positions[row]) < 8

    def can_mutate_column(self, col):
        return sum(self.fixed_positions[row][col] for row in range(9)) < 8

    def reverse_except_fixed(self, values, fixed):
        non_fixed_values = [v for v, f in zip(values, fixed) if not f]
        non_fixed_values.reverse()
        result = []
        non_fixed_idx = 0
        for f in fixed:
            if f:
                result.append(values[fixed.index(f)])
            else:
                result.append(non_fixed_values[non_fixed_idx])
                non_fixed_idx += 1
        return result


class SudokuScrambleMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        grid = self.convert_to_grid(candidate)
        
        subgrid_idx = random.randint(0, 8)
        subgrid = self.get_subgrid(grid, subgrid_idx)
        fixed_subgrid_positions = self.get_fixed_subgrid_positions(subgrid_idx)
        
        mutable_positions = [i for i in range(9) if not fixed_subgrid_positions[i]]
        if len(mutable_positions) < 2:
            return candidate  # No mutation if less than two mutable positions
        
        values_to_shuffle = [subgrid[i] for i in mutable_positions]
        random.shuffle(values_to_shuffle)
        for idx, pos in enumerate(mutable_positions):
            subgrid[pos] = values_to_shuffle[idx]
        
        self.set_subgrid(grid, subgrid_idx, subgrid)
        return self.convert_to_list(grid)

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]

    def get_subgrid(self, grid, subgrid_idx):
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3
        subgrid = []
        for i in range(3):
            subgrid.extend(grid[row_start + i][col_start:col_start + 3])
        return subgrid

    def set_subgrid(self, grid, subgrid_idx, subgrid):
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3
        idx = 0
        for i in range(3):
            for j in range(3):
                grid[row_start + i][col_start + j] = subgrid[idx]
                idx += 1

    def get_fixed_subgrid_positions(self, subgrid_idx):
        row_start = (subgrid_idx // 3) * 3
        col_start = (subgrid_idx % 3) * 3
        fixed_subgrid_positions = []
        for i in range(3):
            for j in range(3):
                fixed_subgrid_positions.append(self.fixed_positions[row_start + i][col_start + j])
        return fixed_subgrid_positions

class SudokuRowColumnSwapMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        grid = self.convert_to_grid(candidate)
        
        if random.choice([True, False]):
            # Row swap within a band
            band = random.randint(0, 2) * 3
            row1, row2 = random.sample(range(band, band + 3), 2)
            if not self.is_row_fixed(row1) and not self.is_row_fixed(row2):
                grid[row1], grid[row2] = grid[row2], grid[row1]
        else:
            # Column swap within a stack
            stack = random.randint(0, 2) * 3
            col1, col2 = random.sample(range(stack, stack + 3), 2)
            if not self.is_column_fixed(col1) and not self.is_column_fixed(col2):
                for row in grid:
                    row[col1], row[col2] = row[col2], row[col1]
        
        return self.convert_to_list(grid)

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]

    def is_row_fixed(self, row):
        return any(self.fixed_positions[row])

    def is_column_fixed(self, col):
        return any(self.fixed_positions[row][col] for row in range(9))


class SudokuRandomResettingMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        grid = self.convert_to_grid(candidate)
        
        while True:
            row, col = random.randint(0, 8), random.randint(0, 8)
            if not self.fixed_positions[row][col]:
                grid[row][col] = random.randint(1, 9)
                break

        return self.convert_to_list(grid)

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]

class SudokuDisplacementMutation(MutationOperator):
    def __init__(self, fixed_positions):
        self.fixed_positions = fixed_positions

    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        grid = self.convert_to_grid(candidate)

        if random.choice([True, False]):
            # Row displacement
            row = random.randint(0, 8)
            if self.can_mutate_row(row):
                start, end = sorted(random.sample(range(9), 2))
                segment = grid[row][start:end]
                grid[row][start:end] = [0] * (end - start)
                new_pos = random.randint(0, 9 - len(segment))
                grid[row][new_pos:new_pos] = segment
        else:
            # Column displacement
            col = random.randint(0, 8)
            if self.can_mutate_column(col):
                start, end = sorted(random.sample(range(9), 2))
                segment = [grid[row][col] for row in range(start, end)]
                for row in range(start, end):
                    grid[row][col] = 0
                new_pos = random.randint(0, 9 - len(segment))
                for i in range(len(segment)):
                    grid[new_pos + i][col] = segment[i]

        return self.convert_to_list(grid)

    def convert_to_grid(self, candidate):
        return [candidate[i*9:(i+1)*9] for i in range(9)]

    def convert_to_list(self, grid):
        return [cell for row in grid for cell in row]

    def can_mutate_row(self, row):
        return sum(self.fixed_positions[row]) < 8

    def can_mutate_column(self, col):
        return sum(self.fixed_positions[row][col] for row in range(9)) < 8



class BinPackingSwapMutation(MutationOperator):
    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        idx1, idx2 = random.sample(range(len(candidate)), 2)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        return candidate



class BinPackingInversionMutation(MutationOperator):
    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        start, end = sorted(random.sample(range(len(candidate)), 2))
        candidate[start:end] = reversed(candidate[start:end])
        return candidate


class BinPackingScrambleMutation(MutationOperator):
    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        start, end = sorted(random.sample(range(len(candidate)), 2))
        sublist = candidate[start:end]
        random.shuffle(sublist)
        candidate[start:end] = sublist
        return candidate



class BinPackingInsertionMutation(MutationOperator):
    def mutate(self, candidate):
        import random
        candidate = list(candidate)  # Convert to list for mutability
        idx1, idx2 = random.sample(range(len(candidate)), 2)
        item = candidate.pop(idx1)
        candidate.insert(idx2, item)
        return candidate



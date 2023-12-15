import copy
import random


class Matrix:
    def __init__(self, rows, columns) -> None:
        self.rows = rows
        self.cols = columns
        self.data = [[0] * self.cols for _ in range(self.rows)]

    @classmethod
    def multiply(self, num: int, matrix: "Matrix") -> "Matrix":
        """Multiplies scalar value"""
        new_matrix = copy.deepcopy(matrix)

        for row in range(matrix.rows):
            for col in range(matrix.cols):
                new_matrix.data[row][col] *= num

        return new_matrix

    @classmethod
    def dot_product(self, matrix_a: "Matrix", matrix_b: "Matrix") -> "Matrix":
        """Dot Product of two matrices"""
        if matrix_a.cols != matrix_b.rows:
            raise ValueError("Invalid dimensions")

        new_matrix = Matrix(matrix_a.rows, matrix_b.cols)
        new_matrix.data = [[0] * matrix_b.cols for _ in range(matrix_a.rows)]

        for row in range(matrix_a.rows):
            for col in range(matrix_b.cols):
                dot_product = 0
                for col_a in range(matrix_a.cols):
                    dot_product += matrix_a.data[row][col_a] * matrix_b.data[col_a][col]
                new_matrix.data[row][col] = dot_product

        return new_matrix

    def zeroize(self) -> None:
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = 0

    def randomize(self) -> None:
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = random.random()

    def add(self, matrix: "Matrix") -> None:
        """add"""
        if not all([self.rows == matrix.rows, self.cols == matrix.cols]):
            raise ValueError("Unable to add matrices with different dimensions")

        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] += matrix.data[row][col]

    def substract(self, matrix: "Matrix") -> None:
        """substract"""
        if not all([self.rows == matrix.rows, self.cols == matrix.cols]):
            raise ValueError("Unable to subtract matrices with different dimensions")

        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] -= matrix.data[row][col]

    def map_func(self, func: any) -> None:
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = func(self.data[row][col])

    def __str__(self) -> None:
        dashes = "-" * self.cols * 3
        string = f"{dashes}\n"
        for row in range(self.rows):
            string += f"{self.data[row]}"
            string += "\n"
        string += f"{dashes}"
        return string

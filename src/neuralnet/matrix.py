import copy
import random
from typing import Union


class Matrix:
    def __init__(self, rows=None, columns=None, matrix: "Matrix" = None) -> None:
        if matrix:
            self.rows = matrix.rows
            self.cols = matrix.cols
            self.data = matrix.data
        elif rows and columns:
            self.rows = rows
            self.cols = columns
            self.data = [[0] * self.cols for _ in range(self.rows)]
        else:
            raise AttributeError("Matrix must have rows and columns or a Matrix")

    @classmethod
    def multiply(self, num: float, matrix: "Matrix") -> "Matrix":
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
            raise ValueError("Invalid dimensions, Matrix A cols must match "
                             f"matrix B rows: {matrix_a.cols=}, {matrix_b.rows=}")

        new_matrix = Matrix(matrix_a.rows, matrix_b.cols)
        new_matrix.data = [[0] * matrix_b.cols for _ in range(matrix_a.rows)]

        for row in range(matrix_a.rows):
            for col in range(matrix_b.cols):
                dot_product = 0
                for col_a in range(matrix_a.cols):
                    dot_product += matrix_a.data[row][col_a] * matrix_b.data[col_a][col]
                new_matrix.data[row][col] = dot_product

        return new_matrix

    @classmethod
    def from_list(self, a_list: list) -> "Matrix":
        matrix = Matrix(len(a_list), 1)

        for i, val in enumerate(a_list):
            matrix.data[i][0] = val

        return matrix

    @classmethod
    def to_list(self, matrix: "Matrix") -> list:
        _list = []

        for row in matrix.data:
            _list.extend(row)

        return _list

    @classmethod
    def transpose(self, old_matrix: "Matrix") -> "Matrix":
        new_matrix = Matrix(old_matrix.cols, old_matrix.rows)
        for row in range(old_matrix.rows):
            for col in range(old_matrix.cols):
                new_matrix.data[col][row] = old_matrix.data[row][col]
        return new_matrix

    @classmethod
    def subtract(self, matrix_a: "Matrix", matrix_b: "Matrix") -> None:
        """substract"""
        if not all([matrix_a.rows == matrix_b.rows, matrix_a.cols == matrix_b.cols]):
            raise ValueError("Unable to subtract matrices with different dimensions")

        new_matrix = Matrix(matrix_a.rows, matrix_a.cols)

        for row in range(matrix_a.rows):
            for col in range(matrix_a.cols):
                new_matrix.data[row][col] = matrix_a.data[row][col] - matrix_b.data[row][col]

        return new_matrix

    def mult(self, value: Union[float, "Matrix"]) -> None:
        """Multiplies scalar value"""

        if isinstance(value, float):
            for row in range(self.rows):
                for col in range(self.cols):
                    self.data[row][col] *= value
        else:
            if all([self.rows != value.rows, self.cols != value.cols]):
                raise ValueError("Rows and Columns must match for elementwise multiplication "
                                 f"{self.rows=}, {value.rows=}, {self.cols=}, {value.cols=}")
            for row in range(self.rows):
                for col in range(self.cols):
                    self.data[row][col] *= value.data[row][col]

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

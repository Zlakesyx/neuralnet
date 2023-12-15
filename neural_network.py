import copy
import math

from matrix import Matrix


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """Constructs NeuralNetwork"""
        # TODO add additional hidden layers
        self.ih_matrix = Matrix(hidden_nodes, input_nodes)
        self.ho_matrix = Matrix(output_nodes, hidden_nodes)
        self.h_bias = Matrix(hidden_nodes, 1)
        self.o_bias = Matrix(output_nodes, 1)
        self.ih_matrix.randomize()
        self.ho_matrix.randomize()

        self.training_rate = 0.1

    def train(self, inputs: list, expected: list) -> None:
        """
        Trains network based on given inputs and expected results
        TODO Finish this function
        """
        inputs_matrix = Matrix.from_list(inputs)
        hidden_matrix = Matrix.dot_product(self.ih_matrix, inputs_matrix)
        hidden_matrix.add(self.h_bias)
        hidden_matrix.map_func(self.sigmoid)

        output_matrix = Matrix.dot_product(self.ho_matrix, hidden_matrix)
        output_matrix.add(self.o_bias)
        output_matrix.map_func(self.sigmoid)

    def feed_forward(self, inputs: list) -> None:
        """Feed forward process"""
        inputs_matrix = Matrix.from_list(inputs)
        hidden_matrix = Matrix.dot_product(self.ih_matrix, inputs_matrix)
        hidden_matrix.add(self.h_bias)
        hidden_matrix.map_func(self.sigmoid)

        output_matrix = Matrix.dot_product(self.ho_matrix, hidden_matrix)
        output_matrix.add(self.o_bias)
        output_matrix.map_func(self.sigmoid)

    def predict(self) -> None:
        """Predicts the output"""
        print(self.ho_matrix)

    def sigmoid(self, x: float) -> None:
        """Sigmoid activation function"""
        return 1 / (1 + math.e ** (-x))


nn = NeuralNetwork(2, 3, 1)
training = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]],
]

for i in range(len(training)):
    nn.train(training[i][0], training[i][1])

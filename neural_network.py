import copy
import math

from matrix import Matrix


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """Constructs NeuralNetwork"""
        # TODO add additional hidden layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.ih_matrix = Matrix(input_nodes, hidden_nodes)
        self.ho_matrix = Matrix(hidden_nodes, output_nodes)
        self.ih_matrix.randomize()
        self.ho_matrix.randomize()

        self.training_rate = 0.1
        self.bias = 1

    def train(self, inputs: list, expected: list) -> None:
        """Trains network based on given inputs and expected results"""

    def feed_forward(self, inputs: list) -> None:
        """Feed forward process"""
        summed_ih = Matrix(self.ih_matrix.rows, self.ih_matrix.cols)
        for in_data in inputs:
            # TODO include bias somehow?
            summed_ih.add(Matrix.multiply(in_data, self.ih_matrix))

        self.ih_matrix = copy.deepcopy(summed_ih)
        print(self.ih_matrix)
        # [map(self.sigmoid, x) for x in self.ih_matrix.data]
        self.ih_matrix.map_func(self.sigmoid)
        print(self.ih_matrix)

        self.ho_matrix = Matrix.dot_product(self.ih_matrix, self.ho_matrix)
        # [map(self.sigmoid, x) for x in self.ho_matrix.data]
        self.ho_matrix.map_func(self.sigmoid)

        self.predict()

    def predict(self) -> None:
        """Predicts the output"""
        print(self.ho_matrix)

    def sigmoid(self, x: float) -> None:
        """Sigmoid activation function"""
        # TODO double check bias is here
        return 1 / (1 + math.e ** (-x + self.bias))


nn = NeuralNetwork(2, 3, 1)
nn.feed_forward([1, 0])

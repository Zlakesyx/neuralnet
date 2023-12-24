import math
import random

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

        self.learning_rate = 0.08

    def train(self, inputs: list, targets: list) -> None:
        """
        Trains network based on given inputs and expected results with
        backpropagation
        """
        # Feed Forward
        inputs_matrix = Matrix.from_list(inputs)
        hidden_matrix = Matrix.dot_product(self.ih_matrix, inputs_matrix)
        hidden_matrix.add(self.h_bias)
        hidden_matrix.map_func(self.sigmoid)
        output_matrix = Matrix.dot_product(self.ho_matrix, hidden_matrix)
        output_matrix.add(self.o_bias)
        output_matrix.map_func(self.sigmoid)

        # Backpropagation
        targets = Matrix.from_list(targets)
        # Calculate output gradient
        # learning rate * output_errors * dt_sigmoid * hidden transposed
        output_errors = Matrix.subtract(targets, output_matrix)
        output_gradients = Matrix(matrix=output_matrix)
        output_gradients.map_func(self.dt_sigmoid)
        output_gradients.mult(output_errors)
        output_gradients.mult(self.learning_rate)
        # Calculate ho delta weights
        hidden_matrix_t = Matrix.transpose(hidden_matrix)
        dt_ho_weights = Matrix.dot_product(output_gradients, hidden_matrix_t)
        self.ho_matrix.add(dt_ho_weights)
        self.o_bias.add(output_gradients)

        # Calculate Hidden gradient
        # learning rate * hidden_errors * dt_sigmoid * inputs transposed
        ho_matrix_t = Matrix.transpose(self.ho_matrix)
        hidden_errors = Matrix.dot_product(ho_matrix_t, output_errors)
        hidden_gradients = Matrix(matrix=hidden_matrix)
        hidden_gradients.map_func(self.dt_sigmoid)
        hidden_gradients.mult(hidden_errors)
        hidden_gradients.mult(self.learning_rate)
        # Calculate ih delta weights
        inputs_matrix_t = Matrix.transpose(inputs_matrix)
        dt_ih_weights = Matrix.dot_product(hidden_gradients, inputs_matrix_t)
        self.ih_matrix.add(dt_ih_weights)
        self.h_bias.add(hidden_gradients)

    def feed_forward(self, inputs: list) -> list[list[float]]:
        """Feed forward process"""
        inputs_matrix = Matrix.from_list(inputs)
        hidden_matrix = Matrix.dot_product(self.ih_matrix, inputs_matrix)
        hidden_matrix.add(self.h_bias)
        hidden_matrix.map_func(self.sigmoid)

        output_matrix = Matrix.dot_product(self.ho_matrix, hidden_matrix)
        output_matrix.add(self.o_bias)
        output_matrix.map_func(self.sigmoid)

        return Matrix.to_list(output_matrix)

    def predict(self) -> None:
        """Predicts the output"""
        print(self.ho_matrix)

    def sigmoid(self, x: float) -> None:
        """Sigmoid activation function"""
        return 1 / (1 + math.e ** (-x))

    def dt_sigmoid(self, x: float) -> None:
        """
        Assumes x has already been run through sigmoid and returns the rest
        of its derivative calculation
        """
        # Actual derivative: self.sigmoid(x) * (1 - self.sigmoid(x))
        return x * (1 - self.sigmoid(x))


#nn = NeuralNetwork(2, 2, 1)
#training = [
#    {
#        "input": [0, 0],
#        "target": [0]
#    },
#    {
#        "input": [0, 1],
#        "target": [1]
#    },
#    {
#        "input": [1, 0],
#        "target": [1]
#    },
#    {
#        "input": [1, 1],
#        "target": [0]
#    },
#]
#
#
#for _ in range(50000):
#    test = random.choice(training)
#    nn.train(test["input"], test["target"])
#
#correct = 0
#total = 0
#
#for _ in range(100):
#    test = random.choice(training)
#    prediction = nn.feed_forward(test["input"])
#    if prediction[0] >= test["target"][0] - 0.1:
#        correct += 1
#    total += 1
#
#print(f"{correct / total * 100}%")

#print(nn.feed_forward([0, 0]))
#print(nn.feed_forward([0, 1]))
#print(nn.feed_forward([1, 0]))
#print(nn.feed_forward([1, 1]))

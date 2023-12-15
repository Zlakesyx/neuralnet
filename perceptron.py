import random
import pygame

# For every input multiply by weight
# Sum all weighted inputs
# Compute output based on sum passed through activation function

pygame.init()
WIDTH = 800
HEIGHT = 800
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))


class Point:
    def __init__(self, x: float = None, y: float = None) -> None:
        if x and y:
            self.x = x
            self.y = y
        else:
            self.x = random.uniform(-1, 1)
            self.y = random.uniform(-1, 1)

        self.bias = 1
        self.line_y = f(self.x)
        self.label = 1 if self.y >= self.line_y else -1

    def draw(self, color) -> None:
        """Draws the point"""

        mapped_x = self.get_mapped_x()
        mapped_y = self.get_mapped_y()

        if self.y > self.line_y:
            pygame.draw.circle(SCREEN, "white", (mapped_x, mapped_y), 8)
            pygame.draw.circle(SCREEN, "black", (mapped_x, mapped_y), 7)

        pygame.draw.circle(SCREEN, color, (mapped_x, mapped_y), 4)

    def get_mapped_x(self) -> float:
        input_start = -1
        input_end = 1
        output_start_x = 0
        output_end_x = WIDTH
        x_slope = 1.0 * (output_end_x - output_start_x) / (input_end - input_start)

        return x_slope * (self.x - input_start) + output_start_x

    def get_mapped_y(self) -> float:
        input_start = -1
        input_end = 1
        output_start_y = HEIGHT
        output_end_y = 0

        y_slope = 1.0 * (output_end_y - output_start_y) / (input_end - input_start)

        return y_slope * (self.y - input_start) + output_start_y


class Perceptron:
    def __init__(self, n: int) -> None:
        """Constructs Perceptron to solve linear problems"""
        self.weights = [random.uniform(-1, 1) for _ in range(n)]
        self.learning_rate = 0.09

    def train(self, inputs: list[float], target: int) -> None:
        """Trains network based on given inputs and expected results"""
        prediction = self.feed_forward(inputs)
        error = target - prediction

        for i, _ in enumerate(self.weights):
            dt_weight = error * inputs[i]
            self.weights[i] += dt_weight * self.learning_rate

    def feed_forward(self, inputs: list[float]) -> int:
        """Feed forward process"""
        output = 0

        for i, weight in enumerate(self.weights):
            output += inputs[i] * weight

        return self.activation(output)

    def activation(self, value: float) -> int:
        """Activation function"""
        if value >= 0:
            return 1
        else:
            return -1

    def guess_Y(self, x: float) -> float:
        """Guesses the output of a line"""
        # w0x * w1y + w2b = 0
        # w1y = -w0x - w2b
        # y = (-w0x - w2b) / w1
        return (-self.weights[0] * x - self.weights[2]) / self.weights[1]


def f(x: float) -> float:
    """y = mx + b"""
    return -0.9 * x + 0.1


def main() -> None:
    running = True
    clock = pygame.time.Clock()
    training_num = 200
    training_index = 0
    training = [Point() for _ in range(training_num)]
    brain = Perceptron(3)  # 2 weights plus bias weight

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        SCREEN.fill((20, 10, 20))

        pygame.draw.line(SCREEN, "white", (WIDTH / 2, 0), (WIDTH / 2, HEIGHT))
        pygame.draw.line(SCREEN, "white", (0, HEIGHT / 2), (WIDTH, HEIGHT / 2))

        p1 = Point(-1.0, f(-1.0))
        p2 = Point(1.0, f(1.0))
        p3 = Point(-1.0, brain.guess_Y(-1))
        p4 = Point(1.0, brain.guess_Y(1))
        pygame.draw.line(SCREEN,
            "yellow",
            (p1.get_mapped_x(), p1.get_mapped_y()),
            (p2.get_mapped_x(), p2.get_mapped_y())
        )
        pygame.draw.line(SCREEN,
            "yellow",
            (p3.get_mapped_x(), p3.get_mapped_y()),
            (p4.get_mapped_x(), p4.get_mapped_y())
        )

        for point in training:
            prediction = brain.feed_forward([point.x, point.y, point.bias])
            if prediction == point.label:
                color = (50, 220, 100)
            else:
                color = (220, 100, 50)
            point.draw(color)

        training_point = training[training_index]
        brain.train([training_point.x, training_point.y, training_point.bias], training_point.label)
        training_index += 1
        if training_index == training_num:
            training_index = 0

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()


if __name__ == "__main__":
    main()

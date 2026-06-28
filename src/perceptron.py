class Perceptron:

    def __init__(self, learning_rate=0.1):
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activation(self, value):
        return 1 if value >= 0 else 0

    def predict(self, inputs):
        total = self.bias

        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        return self.activation(total)

    def train(self, training_data, epochs=20):

        for _ in range(epochs):
            for inputs, expected in training_data:

                prediction = self.predict(inputs)

                error = expected - prediction

                for i in range(len(self.weights)):
                    self.weights[i] += (
                        self.learning_rate * error * inputs[i]
                    )

                self.bias += (
                    self.learning_rate * error
                )

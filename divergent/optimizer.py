from divergent.neuralnet import NeuralNetwork

class Optimizer: 
    """
    We use an optimizer to adjust the parameters 
    of our network based on the gradients computed 
    during backpropagation. 
    """
    def step(self, neural_network: NeuralNetwork) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, neural_network: NeuralNetwork) -> None:
        for paramter, gradient in neural_network.parameters_and_gradients():
            paramter -= self.learning_rate * gradient
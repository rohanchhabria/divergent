import numpy as np 
from divergent.train import train
from divergent.neuralnet import NeuralNetwork
from divergent.layers import Linear, Tanh


"""
The canonical example of a function that can't be 
learnt with a simple linear function model is XOR. 
"""

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

neural_network = NeuralNetwork([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(neural_network, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = neural_network.forward(x)
    print(f'{x} | {predicted} | {y}')
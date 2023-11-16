from divergent.tesnor import Tensor
from divergent.neuralnet import NeuralNetwork
from divergent.loss import Loss, TSE
from divergent.optimizer import Optimizer, SGD
from divergent.data import DataIterator, BatchIterator


"""
Here's a function that can train a neural network. 
"""

def train(neural_network: NeuralNetwork,
          inputs: Tensor,
          targets: Tensor, 
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = TSE(),
          optimizer: Optimizer = SGD()) -> None:
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0 
        for batch in iterator(inputs, targets):
            predicted = neural_network.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            gradient = loss.gradient(predicted, batch.targets)
            neural_network.backward(gradient)
            optimizer.step(neural_network)
        print(f'{epoch} | {epoch_loss}')
            
            
from divergent.tesnor import Tensor
from divergent.layers import Layer
from typing import Tuple, Sequence, Iterator

class NeuralNetwork:
    """
    A NeuralNetwork is just a collection of layers.
    It behaves a lot like a layer itself, although 
    we're not going to make it one. 
    """
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs 
        
    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient 
    
    def parameters_and_gradients(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, parameter in layer.parameters.items():
                gradient = layer.gradients[name]
                yield parameter, gradient
                
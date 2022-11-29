import numpy as np 
from divergent.tesnor import Tensor
from typing import Dict, Callable

class Layer:
    """
    Our neural networks will be made up of layers. 
    Each layer needs to pass it's inputs forward
    and propogate gradients backward. 

    For example, a neural network might look like 
        * inputs -> Linear -> Tanh -> Linear -> output
    """
    def __init__(self) -> None:
        self.parameters: Dict[str, Tensor] = {}
        self.gradients: Dict[str, Tensor] = {}
        
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs.
        """
        raise NotImplementedError
    
    def backward(self, gradient: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer.
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes output b/w input and weights 
    matrix multiplication with inclusion of bias.
        * (inputs @ weight) + bias
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        self.parameters['weight'] = np.random.randn(input_size, output_size)
        self.parameters['bias'] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ weight + bias
        """
        self.inputs = inputs         
        return inputs @ self.parameters['weight'] + self.parameters['bias']
    
    def backward(self, gradient: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b 
        and dy/db = f'(x) * a
        and dy/dc = f'(x)
        
        if y = f(x) and x = a @ b + c [a^b]
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.gradients['weight'] = self.inputs.T @ gradient
        self.gradients['bias'] = np.sum(gradient, axis=0)
        return gradient @ self.parameters['weight'].T

Function = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to it's inputs.
    """
    def __init__(self, function: Function, function_prime: Function) -> None:
        super().__init__()
        self.function = function
        self.function_prime = function_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.function(inputs)
    
    def backward(self, gradient: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.function_prime(self.inputs) * gradient
            
def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - (y ** 2)

class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)
        
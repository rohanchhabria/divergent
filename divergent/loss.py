import numpy as np
from divergent.tesnor import Tensor

class Loss:
    """
    A loss function measures how good our predictions are,
    we can use this to adjust the parameters of our network. 
    """
    def __init__(self) -> None:
        pass
    
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class TSE(Loss):
    """
    MSE is Mean Squared Error, altough we are just going 
    to compute Total Squared Error hence TSE. 
    """
    def __init__(self) -> None:
        pass
    
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
        
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

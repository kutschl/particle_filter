from abc import ABC, abstractmethod
import numpy as np

class DownsamplingInterface(ABC):
    
    indentifier: str
    
    @abstractmethod
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __call__(self, ranges: np.ndarray) -> np.ndarray:
        return ranges


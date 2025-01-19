from abc import ABC, abstractmethod
import numpy as np

class DownsamplingInterface(ABC):
    
    id: str
    sampled_idx: np.ndarray = None
    
    @abstractmethod
    def __init__(self):
        pass

    def get_sampled_idx(self) -> np.ndarray:
        return self.sampled_idx



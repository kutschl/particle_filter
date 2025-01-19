import numpy as np 
from .downsampling_interface import DownsamplingInterface

class UniformDownsampling(DownsamplingInterface):
    id = 'UNI'
    
    def __init__(self, num_beams: int, num_uniform_beams: int):
        self.sampled_idx = np.linspace(0, num_beams - 1, num_uniform_beams, dtype=int)

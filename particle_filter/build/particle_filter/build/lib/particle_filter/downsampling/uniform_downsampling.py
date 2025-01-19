import numpy as np 
from .downsampling_interface import DownsamplingInterface

class UniformDownsampling(DownsamplingInterface):
    # TODO: description of uniform downsampling and explain approach

    identifier='UNI'
    
    def __init__(self, range_min: float, range_max: float, num_uniform_beams: int):
        """
        :param range_min: Minimale Reichweite, unterhalb derer Strahlen ausgeschlossen werden.
        :param range_max: Maximale Reichweite, oberhalb derer Strahlen ausgeschlossen werden.
        :param num_desired_beams: Anzahl der gewünschten Strahlen nach Downsampling.
        """
        self.range_min_ = range_min
        self.range_max_ = range_max
        self.num_uniform_beams_ = num_uniform_beams
    
    def __call__(self, ranges: np.ndarray) -> np.ndarray:
        """
        Verarbeitet den LaserScan: Filtert ungültige Strahlen und führt gleichmäßiges Downsampling durch.

        :param ranges: Liste oder Array der eingehenden LaserScan-Daten (Reichweiten).
        :return: Gefilterte und gleichmäßig reduzierte Reichweiten.
        """
        
        # Filter out all ranges that fall outside the specified range_min and range_max parameters
        filtered_indices = np.where((ranges >= self.range_min_) & (ranges <= self.range_max_))[0]
        
        print(type(filtered_indices))
        # Return empty array 
        num_valid_beams = len(filtered_indices)
        if num_valid_beams == 0:
            return []  # Keine gültigen Daten vorhanden

        # Uniform downsampling
        sampled_indices = np.linspace(0, num_valid_beams - 1, self.num_uniform_beams_, dtype=int)
        
        # Export downsampled range measurement
        return ranges[filtered_indices[sampled_indices]]
        
        
    


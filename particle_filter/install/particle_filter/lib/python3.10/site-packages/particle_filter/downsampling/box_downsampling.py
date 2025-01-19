import numpy as np 
from .downsampling_interface import DownsamplingInterface

class BoxDownsampling(DownsamplingInterface):
    # TODO: description of box downsampling and reference to ETHZ and explain approach

    identifier='BOX'
    
    def __init__(self, box_aspect_ratio: float, num_boxed_beams: float, num_beams: int, angle_min: float, angle_max: float):
        
        beam_angles = np.linspace(angle_min, angle_max, num_beams)

        mid_idx = num_beams//2
        sparse_idxs = [mid_idx]

        # Box
        a = box_aspect_ratio
        beam_proj = 2*a*np.array([np.cos(beam_angles), np.sin(beam_angles)])       

        # Compute the points of intersection along a uniform corridor of given aspect ratio
        beam_intersections = np.zeros((2, num_beams))
        box_corners = [(a, 1), (a, -1), (-a, -1), (-a, 1)]
        for idx in range(len(box_corners)):
            x1, y1 = box_corners[idx]
            x2, y2 = box_corners[0] if idx == 3 else box_corners[idx+1]
            for i in range(num_beams):
                x4 = beam_proj[0, i]
                y4 = beam_proj[1, i]

                den = (x1-x2)*(-y4)-(y1-y2)*(-x4)
                if den == 0:
                    continue    # parallel lines

                t = ((x1)*(-y4)-(y1)*(-x4))/den
                u = ((x1)*(y1-y2)-(y1)*(x1-x2))/den

                px = u*x4
                py = u*y4
                if 0 <= t <= 1.0 and 0 <= u <= 1.0:
                    beam_intersections[0, i] = px
                    beam_intersections[1, i] = py

        # Compute the distances for uniform spacing
        dx = np.diff(beam_intersections[0, :])
        dy = np.diff(beam_intersections[1, :])
        dist = np.sqrt(dx**2 + dy**2)
        total_dist = np.sum(dist)
        dist_amt = total_dist/(num_boxed_beams-1)

        # Calc half of the evenly-spaced interval first, then the other half
        idx = mid_idx + 1
        num_boxed_beams2 = num_boxed_beams//2 + 1
        acc = 0
        while len(sparse_idxs) <= num_boxed_beams2:
            acc += dist[idx]
            if acc >= dist_amt:
                acc = 0
                sparse_idxs.append(idx-1)
            idx += 1

            if idx == num_beams-1:
                sparse_idxs.append(num_beams-1)
                break

        mirrored_half = []
        for idx in sparse_idxs[1:]:
            new_idx = 2*sparse_idxs[0]-idx
            mirrored_half.insert(0, new_idx)
        sparse_idxs = mirrored_half + sparse_idxs

        # Export box sampled lidar indices 
        self.lidar_sample_idxs_ = np.array(sparse_idxs)
        
    def __call__(self, ranges: np.ndarray) -> np.ndarray:
        return ranges[self.lidar_sample_idxs_]
        
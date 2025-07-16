import numpy as np
import torch
def get_adaptive_bins(num_bins):
    """Returns indices for binning an equal number of datapoints per bin."""
    # if np.size(logits, axis=0) == 0:
    #     return np.linspace(0, 1, num_bins+1)[:-1]
    edge_indices = np.linspace(0.0, 1.0, num_bins, endpoint=False)
    return torch.tensor(edge_indices)
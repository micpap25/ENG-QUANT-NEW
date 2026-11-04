# IPM Utility Functions
import numpy as np

# Ratio Test
def ratio(var_vec, delta_var_vec):
    decreasing = delta_var_vec < 0
    if not np.any(decreasing):
        return 1.0
    return min(1.0, np.min(-var_vec[decreasing] / delta_var_vec[decreasing]))
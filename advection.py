import numpy as np
def advection(x_p, v_p, delta_t):
    p_new = np.add(x_p, delta_t * v_p)
    return p_new
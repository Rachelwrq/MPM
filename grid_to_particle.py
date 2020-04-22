from util import *
import numpy as np

def grid_to_particle(x_p, set1, set2, u, v, h, n, e_x, e_y):
    v_p = np.zeros((1,2))
    c_p = np.zeros((1,2))
    for i in range(n):
      w_x = get_weight(x_p, set1[i], h)
      w_y = get_weight(x_p, set2[i], h)
      tmp = np.add(w_x * u[i] * e_x, w_y * v[i] * e_y)
      v_p = np.add(v_p, tmp)
    
      w1_grad = get_weight_grad(x_p, set1[i], h)
      w2_grad = get_weight_grad(x_p, set2[i], h)
      c_p = np.add(c_p, w1_grad * u[i])
      c_p = np.add(c_p, w2_grad * v[i])
    
    return v_p, c_p
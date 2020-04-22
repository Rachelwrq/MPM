from util import *

def particle_to_grid(m_p, v_p, c_p, x_i, x_p, e, h, n):
    # n is the number of particles
    sum_m = 0
    sum_momentum = 0
    for i in range(n):
      w_ip = get_weight(x_p[i], x_i, h)
      m_i = w_ip * m_p[i]
      momentum = m_p[i] * w_ip * (np.matmul(e, v_p[i]) + np.matmul(c_p[i], (x_i - x_p[i])))

      sum_m += m_i
      sum_momentum += momentum
    if sum_m == 0:
        return sum_m, 0
    return sum_m, sum_momentum / sum_m
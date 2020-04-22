import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util import *
from grid_to_particle import *
from particle_to_grid import *
from advection import *
import time


def MPM_algo(m_p, x_pi, v_pi, set1, set2, n, h=0.5, delta_t=0.1):
    e_x = np.array((1,0))
    num = 110
    m1_i = np.zeros(num)
    v1_i = np.zeros(num)
    e_y = np.array((0,1))
    m2_i = np.zeros(num)
    v2_i = np.zeros(num)
    
    x_p = np.copy(x_pi)
    v_p = np.copy(v_pi)

    c_p = np.zeros((len(x_p),2))
    D_t, D = get_D()
    P1, P1_t = get_P1()
    P2, P2_t = get_P2()
    vp_new = np.zeros((len(x_p),2))
    cp_new = np.zeros((len(x_p),2))
    new_position = np.zeros((len(x_p),2))
    start_time = time.time()
    fig, ax = plt.subplots()
    plt.scatter(x_p[:,0], x_p[:,1], s = 20)
    # plt.scatter(set2_x, set2_y, color = "red")
    ax.set_xticks(np.arange(0,5.5,0.5))
    ax.set_yticks(np.arange(0,5.5,0.5))
    plt.grid()
    plt.savefig('0.jpg')
    plt.close()
    v_max = 0
    v_min = 0
    for j in range(n):
      v_p = gravity_step(v_p, delta_t)
      for i in range(len(set1)):
        m1_i[i], v1_i[i] = particle_to_grid(m_p, v_p, c_p, set1[i], x_p, e_x, h, len(x_p))
      for i in range(len(set2)):
        m2_i[i], v2_i[i] = particle_to_grid(m_p, v_p, c_p, set2[i], x_p, e_y, h, len(x_p))

      v_i = np.concatenate((v1_i, v2_i))

      b = -np.matmul(P2_t, np.matmul(D_t, np.matmul(P1, np.matmul(P1_t, v_i)))) / delta_t
      L = np.matmul(P2_t, np.matmul(D_t, np.matmul(P1, np.matmul(P1_t, np.matmul(D, P2)))))

      pc = np.linalg.solve(L, b)
      p = np.matmul(P2, pc)
      
      v_new = np.matmul(P1, np.matmul(P1_t, v_i)) + delta_t * np.matmul(P1, np.matmul(P1_t, np.matmul(D, p)))
      # v_new = np.matmul(P1, np.matmul(P1_t, v_i)) + delta_t * np.matmul(D, p)
      # print(v_new)

      v_new1 = v_new[:110]
      v_new2 = v_new[110:]
      for i in range(len(x_p)):
        vp_new[i], cp_new[i] = grid_to_particle(x_p[i], set1, set2, v_new1, v_new2, h, len(set1), e_x, e_y)
        new_position[i] = advection(x_p[i], vp_new[i], delta_t)
      
      
      x_p = np.copy(new_position)
      v_p = np.copy(vp_new)
      c_p = np.copy(cp_new)

      fig, ax = plt.subplots()
      plt.scatter(x_p[:,0], x_p[:,1], s = s)
      # plt.scatter(set2_x, set2_y, color = "red")
      ax.set_xticks(np.arange(0,5.5,0.5))
      ax.set_yticks(np.arange(0,5.5,0.5))
      plt.grid()
      plt.savefig('{}.jpg'.format(j+1))
      plt.close()
      v_max = np.amax(v_p)
      v_min = np.amin(v_p)
    end_time = time.time()
    print("Time used: ", end_time - start_time)
    print(v_max, v_min)
    return
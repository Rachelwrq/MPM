import numpy as np

# Interpolation function
def N(x):
    x = abs(x)

    # use cubic kernel
    if x < 1:
        return 0.5 * x ** 3 - x ** 2 + 2 / 3
    elif x < 2:
        return ((2 - x) ** 3) / 6
    else:
        return 0

def dN_dx(x):
    x = abs(x)

    if x < 1:
        return 1.5 * x ** 2 - 2 * x
    elif x < 2:
        return - ((2-x) ** 2) / 2
    else:
        return 0


def get_weight(x_p, x_i, h):
    x = N((x_p[0] - x_i[0]) / h)
    y = N((x_p[1] - x_i[1]) / h)
    return x * y

def get_weight_grad(x_p, x_i, h):
    x = dN_dx((x_p[0] - x_i[0]) / h) * N((x_p[1] - x_i[1]) / h)
    y = N((x_p[0] - x_i[0]) / h) * dN_dx((x_p[1] - x_i[1]) / h)
    result = np.array((x, y))
    return result

# Divergence matrix
def get_D():
  delta_x = 0.5
  delta_y = 0.5

  c = 100
  hor = 10
  p = 110 + 110
  row = 10
  col = 11

  x_co = 1/delta_x
  y_co = 1/delta_y

  x_comp = np.zeros(110)
  x_comp[-2] = -x_co
  x_comp[-1] = x_co
  # print(x_comp)

  y_comp = np.zeros(110)
  y_comp[-1] = -y_co
  y_comp[hor-1] = y_co

  A = np.zeros(shape=(c,p))
  printing = False
  for i in range(c):
    if i % row == 0:
      x_comp = np.roll(x_comp, 2)
    else:
      x_comp = np.roll(x_comp, 1)
    
    y_comp = np.roll(y_comp, 1)
    A[i] = (np.concatenate((x_comp, y_comp)))
  return A, A.T

def get_P1():
  del_1 = 0
  del_2 = 10
  P_t = np.identity(220)
  # print(len(P_t))
  del_list = []
  for i in range(110):
    if i == del_1:
      del_list.append(i)
      del_1 += 11
    if i == del_2:
      del_list.append(i)
      del_2 += 11

  P_t = np.delete(P_t, del_list, 0)

  # print(len(P_t))
  P_t = np.delete(P_t, slice(90, 100), 0)
  # print(len(P_t))
  P_t = np.delete(P_t, slice(180, 190), 0)
  # print(np.shape(P_t))

  P = P_t.T
  # print(np.shape(P))
  return P, P_t

def get_P2():
  P2 = np.identity(99)
  ttt = np.zeros(99)
  P2 = np.insert(P2, 0, ttt, axis=0)
  return P2, P2.T

def gravity_step(v_p, delta_t):
  g = np.empty(shape=(len(v_p),2))
  g.fill(-9.8*delta_t)
  g[:,0].fill(0)
  vvv = np.add(v_p, g)
  return vvv
from functools import partial
import equinox as eqx
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import optax
from scipy.interpolate import griddata
from tqdm import tqdm
import numpy

def generate_initial_condition(key, N, A):
  """generate an initial function that is 2pi-periodic and has zero mean in the interval [-pi,pi]

  Args:
      key (jax.randon.PRNGKey): a random key
      N (int): number of sine terms to include in the Fourier series
      A (float): variance of the Gaussian from which the coefficients are sampled

  Returns:
      callable: a function representing the initial condition
  """
  # generate random coefficients
  coefficients = A * jax.random.normal(key, (N,))

  def __initial_condition(x):
    sine_terms = np.zeros_like(x)
    for n in range(N):
      sine_terms += coefficients[n] * np.sin((n + 1) * x)
    # set small values to zero
    # sine_terms = np.where(np.abs(sine_terms) < 1e-14, 0, sine_terms)
    return sine_terms

  return __initial_condition

def solve_burgers(key, number_of_sensors, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period, kappa):
  """solve the 1D Burgers' equation u_t + uu_x = k * uu_x with a given initial condition and perioic boundary conditions

  Args:
      key (_type_): _description_
      num_sine_terms (_type_): _description_
      sine_amplitude (_type_): _description_
      Nx (_type_): _description_
      Nt (_type_): _description_
      T_lim (_type_): _description_
      period (_type_): _description_
      kappa (_type_): _description_
  """
  xmin, xmax = -period * np.pi, period * np.pi
  tmin, tmax = 0, T_lim
  # generate subkeys
  subkeys = jax.random.split(key, 2)
  # generate initial condition function
  initial_condition_fn = generate_initial_condition(subkeys[0], num_sine_terms, sine_amplitude)

  # create grid
  x = np.linspace(xmin, xmax, Nx)
  t = np.linspace(tmin, tmax, Nt)
  h = x[1] - x[0]
  dt = t[1] - t[0]

  # Compute initial condition
  u0 = initial_condition_fn(x)

  # Finite Difference Approximation Matrices
  D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)  # first derivative approximation matrix
  D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
  D3 = np.eye(Nx - 2)  # enforce BCs
  M = -np.diag(D1 @ (kappa * np.ones_like(x))) @ D1 - 4 * np.diag(kappa * np.ones_like(x)) @ D2
  m_bond = 8 * h**2 / dt * D3 + M[1:-1, 1:-1]
  c = 8 * h**2 / dt * D3 - M[1:-1, 1:-1]

  u = np.zeros((Nx, Nt))
  u = u.at[:, 0].set(u0)

  def body_fn(i, u):
    u_x = D1 @ u[:, i]
    nonlinear_term = u[1:-1, i] * u_x[1:-1]
    b2 = c @ u[1:-1, i].T - nonlinear_term * h**2 / 2
    u = u.at[1:-1, i + 1].set(np.linalg.solve(m_bond, b2))
    return u

  s = jax.lax.fori_loop(0, Nt - 1, body_fn, u)  # PDE solution over Nx x Nt grid
  # Input sensor locations and measurements
  # xx = np.linspace(xmin, xmax, number_of_sensors)
  xx_r = jax.random.uniform(key, shape=(number_of_sensors,), minval=xmin, maxval=xmax)
  u_r = initial_condition_fn(xx_r)
  xx = np.linspace(xmin, xmax, number_of_sensors)
  u = initial_condition_fn(xx)
  return (x, t, s), (xx_r, u_r, u, u0)


def solve_burgers_uniform(key, number_of_sensors, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period, kappa):
  """solve the 1D Burgers' equation u_t + uu_x = k * uu_x with a given initial condition and perioic boundary conditions

  Args:
      key (_type_): _description_
      num_sine_terms (_type_): _description_
      sine_amplitude (_type_): _description_
      Nx (_type_): _description_
      Nt (_type_): _description_
      T_lim (_type_): _description_
      period (_type_): _description_
      kappa (_type_): _description_
  """
  xmin, xmax = -period * np.pi, period * np.pi
  tmin, tmax = 0, T_lim
  # generate subkeys
  subkeys = jax.random.split(key, 2)
  # generate initial condition function
  initial_condition_fn = generate_initial_condition(subkeys[0], num_sine_terms, sine_amplitude)

  # create grid
  x = np.linspace(xmin, xmax, Nx)
  t = np.linspace(tmin, tmax, Nt)
  h = x[1] - x[0]
  dt = t[1] - t[0]

  # Compute initial condition
  u0 = initial_condition_fn(x)

  # Finite Difference Approximation Matrices
  D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)  # first derivative approximation matrix
  D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
  D3 = np.eye(Nx - 2)  # enforce BCs
  M = -np.diag(D1 @ (kappa * np.ones_like(x))) @ D1 - 4 * np.diag(kappa * np.ones_like(x)) @ D2
  m_bond = 8 * h**2 / dt * D3 + M[1:-1, 1:-1]
  c = 8 * h**2 / dt * D3 - M[1:-1, 1:-1]

  u = np.zeros((Nx, Nt))
  u = u.at[:, 0].set(u0)

  def body_fn(i, u):
    u_x = D1 @ u[:, i]
    nonlinear_term = u[1:-1, i] * u_x[1:-1]
    b2 = c @ u[1:-1, i].T - nonlinear_term * h**2 / 2
    u = u.at[1:-1, i + 1].set(np.linalg.solve(m_bond, b2))
    return u

  s = jax.lax.fori_loop(0, Nt - 1, body_fn, u)  # PDE solution over Nx x Nt grid
  # Input sensor locations and measurements
  xx_r = np.linspace(xmin, xmax, number_of_sensors)
  u_r = initial_condition_fn(xx_r)
  xx = np.linspace(xmin, xmax, number_of_sensors)
  u = initial_condition_fn(xx)
  return (x, t, s), (xx_r, u_r, u, u0)

# generate training data correspondingto one input sample
def generate_one_training_data(key, num_of_sensors, num_query_points, num_sine_terms, sine_amplitude, Nx, Nt, T_lim,
                               period, kappa):
  # numerical solution
  (x, t, s), (x_obs, u_r, u, u0) = solve_burgers(key, num_of_sensors, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period,
                                     kappa)  # u'shape: [num_of_sensors, ]
  # generate subkeys
  subkeys = jax.random.split(key, 2)
  # sample input data, copy u for num_query_points times
  x_obs = np.tile(x_obs, (num_query_points, 1))
  u_r_train = np.tile(u_r, (num_query_points, 1))  # u_train'shape: [num_query_points, num_of_sensors]
  u_train = np.tile(u, (num_query_points, 1))  # u_train'shape: [num_query_points, num_of_sensors]
  # sample query points, random select num_query_points points from Nx and Nt points
  x_query_idx = jax.random.choice(subkeys[0], np.arange(Nx), shape=(num_query_points, 1))
  x_query = x[x_query_idx]  # x_query'shape: [num_query_points, 1]
  t_query_idx = jax.random.choice(subkeys[1], np.arange(Nt), shape=(num_query_points, 1))
  t_query = t[t_query_idx]  # t_query'shape: [num_query_points, 1]
  y_train = np.hstack((x_query, t_query))  # y_train'shape: [num_query_points, 2]
  # outputs data
  s_train = s[x_query_idx, t_query_idx]  # s_train'shape: [num_query_points, 1]
  return x_obs, u_r_train, u_train, y_train, s_train

def generate_one_test_data(key, num_of_sensors, num_query_points, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period,
                           kappa):
  (x, t, s), (x_obs, u_r, u, u0) = solve_burgers(key, num_of_sensors, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period, kappa)
  XX, TT = np.meshgrid(x, t)
  x_obs = np.tile(x_obs, (Nx * Nt, 1))
  u_r = np.tile(u_r, (Nx * Nt, 1))
  u_test = np.tile(u, (Nx * Nt, 1))  # copy u for Nx * Nt times
  y_test = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None]))  # flatten x, t
  s_test = s.T.flatten() # flatten默认逐行展开，这里需要先转至再展开，才能维持对应关系
  return x_obs, u_r, u_test, y_test, s_test

def generate_one_test_data_uniform(key, num_of_sensors, num_query_points, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period,
                           kappa):
  (x, t, s), (x_obs, u_r, u, u0) = solve_burgers_uniform(key, num_of_sensors, num_sine_terms, sine_amplitude, Nx, Nt, T_lim, period, kappa)
  XX, TT = np.meshgrid(x, t)
  x_obs = np.tile(x_obs, (Nx * Nt, 1))
  u_r = np.tile(u_r, (Nx * Nt, 1))
  u_test = np.tile(u, (Nx * Nt, 1))  # copy u for Nx * Nt times
  y_test = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None]))  # flatten x, t
  s_test = s.T.flatten() # flatten默认逐行展开，这里需要先转至再展开，才能维持对应关系
  return x_obs, u_r, u_test, y_test, s_test

def generate_training_data(N, P, m):
    key = jax.random.PRNGKey(2)
    subkeys = jax.random.split(key, N)
    gen_train_data = partial(generate_one_training_data,
                             num_of_sensors=m,
                             num_query_points=P,
                             num_sine_terms=3,
                             sine_amplitude=0.2,
                             Nx=100,
                             Nt=100,
                             T_lim=5,
                             period=1,
                             kappa=0.01)
    x_obs, u_r_train, u_train, y_train, s_train = jax.vmap(gen_train_data)(subkeys)
    print("training info: ", u_train.shape, y_train.shape, s_train.shape)
    u_train = numpy.array(u_train).reshape(N * P, -1)
    y_train = numpy.array(y_train).reshape(N * P, -1)
    s_train = numpy.array(s_train).reshape(N * P, -1)
    u_r_train = numpy.array(u_r_train).reshape(N * P, -1)
    x_obs = numpy.array(x_obs).reshape(N * P, -1)
    u_r_train = numpy.stack([x_obs, u_r_train], axis=1)
    return u_r_train, u_train, y_train, s_train

def generate_test_data(test_key, N, P, m):
  # test_key = jax.random.PRNGKey(3)
  subkeys_test = jax.random.split(test_key, N)
  gen_test_data = partial(generate_one_test_data,
                          num_of_sensors=m,
                          num_query_points=P,
                          num_sine_terms=3,
                          sine_amplitude=0.2,
                          Nx=100,
                          Nt=100,
                          T_lim=5,
                          period=1,
                          kappa=0.01)
  x_obs, u_r_test, u_test, y_test, s_test = jax.vmap(gen_test_data)(subkeys_test)
  # print("test info: ", u_test.shape, y_test.shape, s_test.shape)
  u_test = numpy.array(u_test).reshape(N * P, -1)
  y_test = numpy.array(y_test).reshape(N * P, -1)
  s_test = numpy.array(s_test).reshape(N * P, -1)
  u_r_test = numpy.array(u_r_test).reshape(N * P, -1)
  x_obs = numpy.array(x_obs).reshape(N * P, -1)
  u_r_test = numpy.stack([x_obs, u_r_test], axis=1)
  return u_r_test, u_test, y_test, s_test

def generate_test_data_uniform(test_key, N, P, m):
  # test_key = jax.random.PRNGKey(3)
  subkeys_test = jax.random.split(test_key, N)
  gen_test_data = partial(generate_one_test_data,
                          num_of_sensors=m,
                          num_query_points=P,
                          num_sine_terms=3,
                          sine_amplitude=0.2,
                          Nx=100,
                          Nt=100,
                          T_lim=5,
                          period=1,
                          kappa=0.01)
  x_obs, u_r_test, u_test, y_test, s_test = jax.vmap(gen_test_data)(subkeys_test)
  # print("test info: ", u_test.shape, y_test.shape, s_test.shape)
  u_test = numpy.array(u_test).reshape(N * P, -1)
  y_test = numpy.array(y_test).reshape(N * P, -1)
  s_test = numpy.array(s_test).reshape(N * P, -1)
  u_r_test = numpy.array(u_r_test).reshape(N * P, -1)
  x_obs = numpy.array(x_obs).reshape(N * P, -1)
  u_r_test = numpy.stack([x_obs, u_r_test], axis=1)
  return u_r_test, u_test, y_test, s_test
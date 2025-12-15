import numpy as np
import jax.numpy as jnp
import jax
# from jax.ops import index_update, index
# from jax import lax
# from jax.config import config
from jax import config
from jax.experimental.ode import odeint

length_scale = 0.2

# Define RBF kernel
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs ** 2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


# Geneate training data corresponding to one input sample
def generate_one_training_data(key, m=100, P=1):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    gp_sample = jnp.dot(L, jax.random.normal(key, (N,)))

    # Create a callable interpolation function
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    u = jax.vmap(u_fn, in_axes=(None, 0))(0.0, x)

    x_p = jnp.array(np.random.rand(P, m))
    u_p = jax.vmap(u_fn, in_axes=(None, 0))(0.0, x_p)
    u_p_train = jnp.stack([x_p, u_p], axis=1)


    # Output sensor locations and measurements
    y_train = jax.random.uniform(key, (P,)).sort()
    s_train = odeint(u_fn, 0.0, jnp.hstack((0.0, y_train)))[
              1:]  # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]

    # Tile inputs
    u_train = jnp.tile(u, (P, 1))

    # training data for the residual
    u_r_train = jnp.tile(u, (m, 1))
    y_r_train = x
    s_r_train = u
    return u_p_train, u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


# Geneate test data corresponding to one input sample



# Geneate training data corresponding to N input sample
def generate_training_data(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = jax.random.split(key, N)
    gen_fn = jax.jit(lambda key: generate_one_training_data(key, m, P))
    u_p_train, u_train, y_train, s_train, u_r_train, y_r_train, s_r_train = jax.vmap(gen_fn)(keys)

    u_train = jnp.float32(u_train.reshape(N * P, -1))
    y_train = jnp.float32(y_train.reshape(N * P, -1))
    s_train = jnp.float32(s_train.reshape(N * P, -1))

    u_p_train = jnp.float32(u_p_train.reshape(N * P, 2, -1))

    u_r_train = jnp.float32(u_r_train.reshape(N * m, -1))
    y_r_train = jnp.float32(y_r_train.reshape(N * m, -1))
    s_r_train = jnp.float32(s_r_train.reshape(N * m, -1))

    config.update("jax_enable_x64", False)

    u_train, y_train, s_train = np.array(u_train), np.array(y_train), np.array(s_train)
    u_p_train = np.array(u_p_train)

    return u_p_train, u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


def generate_one_test_data(key, m=100, P=100):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    gp_sample = jnp.dot(L, jax.random.normal(key, (N,)))

    # Create a callable interpolation function
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    # x = jnp.linspace(0, 1, m)
    x = jax.random.uniform(key, shape=(m,), minval=0, maxval=1)

    u = jax.vmap(u_fn, in_axes=(None, 0))(0.0, x)

    # Output sensor locations and measurements
    y = jnp.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)

    u_p = jnp.tile(jnp.stack([x, u], axis=1).T, (P, 1))

    # Tile inputs
    u = jnp.tile(u, (P, 1))

    return u_p, u, y, s

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = jax.random.split(key, N)
    gen_fn = jax.jit(lambda key: generate_one_test_data(key, m, P))
    u_p, u, y, s = jax.vmap(gen_fn)(keys)
    u = jnp.float32(u.reshape(N * P, -1))
    y = jnp.float32(y.reshape(N * P, -1))
    s = jnp.float32(s.reshape(N * P, -1))

    u_p = jnp.float32(u_p.reshape(N * P, 2, -1))

    config.update("jax_enable_x64", False)

    u, y, s = np.array(u), np.array(y), np.array(s)
    u_p = np.array(u_p)
    return u_p, u, y, s



def generate_one_test_data_uniform_obs(key, m=100, P=100):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    gp_sample = jnp.dot(L, jax.random.normal(key, (N,)))

    # Create a callable interpolation function
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    # x = jax.random.uniform(key, shape=(m,), minval=0, maxval=1)

    u = jax.vmap(u_fn, in_axes=(None, 0))(0.0, x)

    # Output sensor locations and measurements
    y = jnp.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)

    u_p = jnp.tile(jnp.stack([x, u], axis=1).T, (P, 1))

    # Tile inputs
    u = jnp.tile(u, (P, 1))

    return u_p, u, y, s

# Geneate test data corresponding to N input sample
def generate_test_data_uniform_obs(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = jax.random.split(key, N)
    gen_fn = jax.jit(lambda key: generate_one_test_data_uniform_obs(key, m, P))
    u_p, u, y, s = jax.vmap(gen_fn)(keys)
    u = jnp.float32(u.reshape(N * P, -1))
    y = jnp.float32(y.reshape(N * P, -1))
    s = jnp.float32(s.reshape(N * P, -1))

    u_p = jnp.float32(u_p.reshape(N * P, 2, -1))

    config.update("jax_enable_x64", False)

    u, y, s = np.array(u), np.array(y), np.array(s)
    u_p = np.array(u_p)
    return u_p, u, y, s
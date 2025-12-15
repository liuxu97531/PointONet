from tqdm import trange
import tqdm
import torch.nn.functional as F
from scipy.interpolate import griddata

from jax import lax
JAX = False
if not JAX:
    import numpy as np
if JAX:
    import jax.numpy as np
    from jax import lax

Nx = 100
Nt = 100
xmin, xmax = 0, 1
tmin, tmax = 0, 1

def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def solve_ADR(Nx, Nt, xmin, xmax, tmin, tmax, k, v, g, dg, u0, f_fn):
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    k = k(x)
    v = v(x)
    f = f_fn(x)

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    u = np.zeros((Nx, Nt))
    if not JAX:
        u[:, 0] = u0(x)
        u[0, :] = -1
        u[-1, :] = -1
    else:
        u = u.at[np.index_exp[:, 0]].set(u0(x))
        u = u.at[np.index_exp[0, :]].set(-1)
        u = u.at[np.index_exp[-1, :]].set(-1)

    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        if not JAX:
            u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
        else:
            u = u.at[np.index_exp[1:-1, i + 1]].set(np.linalg.solve(A, b1 + b2))
        return u

    if not JAX:
        for i in range(Nt - 1):
            body_fn(i, u)
    else:
        u = lax.fori_loop(0, Nt - 1, body_fn, u)
    return x, t, u

def generate_one_test_data(P,m):
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)
    N = 512
    length_scale = 0.2
    gp_params = (1.0, length_scale)
    jitter = 1e-10

    X = np.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = np.dot(L, np.random.randn(N))
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    x, t, UU = solve_ADR(Nx, Nt, xmin, xmax, tmin, tmax, k, v, g, dg, u0, f_fn)

    # xx = np.linspace(xmin, xmax, m)
    xx = (np.random.rand(1, m) * (xmax - xmin) + xmin).reshape(m, )
    xx = np.sort(xx)
    u = f_fn(xx)
    u_test = u

    u = np.stack([xx, u], axis=1).T
    u_p_test = u

    XX, TT = np.meshgrid(x, t)
    y_test = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_test = UU.T.flatten()
    return u_p_test, y_test, s_test, u_test

def generate_data_pointnet(N, P, m):
    pbar = trange(N, disable=True)
    test_data_P = []
    test_data_D = []
    for _ in pbar:
        data_out = generate_one_test_data(P, m)
        test_data_P.append(data_out[:3])
        test_data_D.append(data_out[3:])
    u_test = np.array([data[0] for data in test_data_P]).reshape(N, 2, -1)
    y_test = test_data_P[0][1]
    s_test = np.array([data[2] for data in test_data_P])
    u1_test = np.array([data[0] for data in test_data_D]).reshape(N, -1)
    return u_test, y_test, s_test, u1_test



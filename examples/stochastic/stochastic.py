from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import scipy.linalg as dla
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.stats import gamma
from scipy.stats import norm

import scipy.integrate as integrate
import math
import ufl

# directories
output_dir = "./output/"
mesh_dir = "./mesh/"


# exponential covariance function
def cov_exp(r, rho, sigma=1.0):
    return sigma * np.exp(-math.pi * r * r / 2.0 / rho / rho)

# covariance length


def cov_len(rho, sigma=1.0):
    return integrate.quad(lambda r: cov_exp(r, rho), 0, math.inf)


def solve_covariance_EVP(cov, N, degree=1):
    def setup_FEM(N):
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, 'CG', degree)
        u = TrialFunction(V)
        v = TestFunction(V)
        return mesh, V, u, v
    # construct FEM space
    mesh, V, u, v = setup_FEM(N)

    # dof to vertex map
    dof2vert = dof_to_vertex_map(V)
    # coords will be used for interpolation of covariance kernel
    coords = mesh.coordinates()
    # but we need degree of freedom ordering of coordinates
    coords = coords[dof2vert]

    # assemble mass matrix and convert to scipy
    M = assemble(u * v * dx)
    M = M.array()

    print("size of M: ")
    print(len(M))

    # evaluate covariance matrix
    L = coords.shape[0]
    C = np.zeros([L, L])

    for i in range(L):
        for j in range(L):
            if j <= i:
                v = cov(np.linalg.norm(coords[i] - coords[j]))
                C[i, j] = v
                C[j, i] = v

    # solve eigenvalue problem
    A = np.dot(M, np.dot(C, M))

    # w, v = spla.eigsh(A, k, M)
    w, v = dla.eigh(A, b=M)

    return w, v, mesh, C, M, V


def set_fem_fun(vec, fs):
    retval = Function(fs)
    retval.vector().set_local(vec)
    return retval


w, v, mesh, C, M, V = solve_covariance_EVP(
    lambda r: cov_exp(r, rho=0.2, sigma=1.0), N=10, degree=1)

idx = w.argsort()[::-1]
w = w[idx]
v = v[:, idx]


print("Truncation error")
e = 0
eig = 0
trCM = np.trace(np.dot(C, M))
while 1 - eig / trCM > 0.1:
    eig = eig + w[e]
    e = e + 1
print(e)
print(1 - eig / trCM)


randomField = np.zeros(v[:, 0].shape)

gauss = np.random.normal(loc=0.0, scale=1.0, size=(len(w), 1))


for i in range(e):
    print(w[i])
    randomField = randomField + sqrt(w[i]) * v[:, i] * gauss[i]

for i in range(len(w)):
    randomField[i] = norm.cdf(randomField[i])
    randomField[i] = gamma.ppf(randomField[i], 20, loc=0, scale=0.05)


rF = set_fem_fun(randomField, V)

file = File(output_dir + "2D_Random.pvd")
file << rF

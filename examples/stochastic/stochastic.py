# system imports
from __future__ import division
from dolfin import *
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

import scipy.integrate as integrate
import ufl

from mshr import *
import time

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross
from utils import build_nullspace
from utils import XDMF2PVD
from utils import cov_exp
from utils import cov_len
from utils import set_fem_fun
from utils import trun_order
from utils import solve_covariance_EVP
from utils import order_eig
from utils import nonGauss

# directories
output_dir = "./output/"
mesh_dir = "./mesh/"

# mesh
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

# function space
V = FunctionSpace(mesh, 'CG', 2)

w, v, C, M = solve_covariance_EVP(
    lambda r: cov_exp(r, rho=0.2, sigma=1.0), mesh, V)


w, v = order_eig(w, v)


print("Truncation error")
e, error = trun_order(0.1, C, M, w)
print(e)
print(error)

randomField = nonGauss(w, v, 0, 0.05, e)

rF = set_fem_fun(randomField, V)

file = File(output_dir + "2D_Random.pvd")
file << rF

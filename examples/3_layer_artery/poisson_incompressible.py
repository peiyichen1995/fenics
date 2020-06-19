# system imports
from dolfin import *
from mshr import *
import math
import time
from ufl.classes import *
from ufl.algorithms import *
from numpy import linalg as LA

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross
from utils import build_nullspace
from utils import XDMF2PVD

# system imports
from utils import build_nullspace
from utils import my_cross
from solvers import SolverWithNullSpace
from problems import ProblemWithNullSpace
import ufl
import math
from dolfin import *
from mshr import *
import numpy as np

# my imports
from problems import CustomProblem
from solvers import CustomSolver
from utils import MSH2XDMF, XDMF2PVD, matrix_cofactor


def NeoHookean(c1, F):
    J = det(F)
    C = F.T * F
    C_bar = pow(J, -2 / 3) * C
    return c1 * (tr(C_bar) - 3)


def NeoHookean_imcompressible(mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F, Q):
    J = det(F)
    C = F.T * F
    temp1 = mu1 * tr(C) / pow(J, 2 / 3)
    temp2 = mu2 * pow(sqrt(tr(matrix_cofactor(F).T *
                              matrix_cofactor(F))), 3) / pow(J, 2)
    h = pow(J, beta3) + pow(J, -beta3)
    temp3 = mu3 * h
    temp4 = mu4 / beta4 * \
        exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
                                    inner(F * a1, F * a1) - 1, 0), 2))
    temp5 = mu4 / beta4 * \
        exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
                                    inner(F * a2, F * a2) - 1, 0), 2))
    return temp1 + temp2 + temp3 + temp4 + temp5


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# directories
output_dir = "./output/"
mesh_dir = "./mesh/"

start_time = time.time()

# mesh
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")
n = FacetNormal(mesh)

# mark boundaries
ds = Measure('ds', domain=mesh, subdomain_data=mf)

# function space
V = FunctionSpace(mesh, 'CG', 2)
exit()
VV = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)
Q = FunctionSpace(mesh, "CG", 1)

# functions
v = TrialFunction(VV)
w = TestFunction(VV)
phi1 = Function(V)
phi2 = Function(V)
u = Function(VV)

# read in laplace solutions
phi1_h5 = HDF5File(MPI.comm_world, output_dir + "phi1.h5", "r")
phi2_h5 = HDF5File(MPI.comm_world, output_dir + "phi2.h5", "r")
phi1_h5.read(phi1, "phi1")
phi2_h5.read(phi2, "phi2")
phi1_h5.close()
phi2_h5.close()

# define orthorgonal basis
e3 = grad(phi1)
e1 = grad(phi2)
e2 = my_cross(e3, e1)

# normalize basis
e1 = e1 / sqrt(inner(e1, e1))
e2 = e2 / sqrt(inner(e2, e2))
e3 = e3 / sqrt(inner(e3, e3))

# defin tissue orientation on the spatial varying basis
theta = 46.274 / 180 * math.pi
a1 = math.cos(theta) * e3 + math.sin(theta) * e2
a2 = math.cos(theta) * e3 - math.sin(theta) * e2


# Kinematics
d = u.geometric_dimension()
I = Identity(d)
F = I + grad(u)


# pressure
P = Expression("t", t=0.0, degree=0)

mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
delta = 0.2

psi = NeoHookean_imcompressible(mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F, V)
Pi = psi * dx - dot(-P * n, u) * ds(1)

dPi = derivative(Pi, u, w)
J = derivative(dPi, u, v)
null_space = build_nullspace(VV)


# solve variational problem
problem = ProblemWithNullSpace(J, dPi, [], null_space)
solver = SolverWithNullSpace()


start_time = time.time()
T = 1
num_steps = 1
dt = T / num_steps
t = 0
for n in range(num_steps):
    t += dt
    P.t = t
    print("Time step " + str(n) + ", t = " + str(t))
    solver.solve(problem, u.vector())

    # write solution
    file = File(output_dir + "displacements_step_" + str(n) + ".pvd")
    file << u

print("runnning time")
print(time.time() - start_time)

file1 = open("myfile.txt", "w")  # write mode
file1.write(time.time() - start_time)
file1.close()

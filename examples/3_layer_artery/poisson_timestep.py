# system imports
from dolfin import *
from mshr import *
import math
import time

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
from utils import MSH2XDMF, XDMF2PVD


def NeoHookean(c1, F):
    J = det(F)
    C = F.T * F
    C_bar = pow(J, -2 / 3) * C
    return c1 * (tr(C_bar) - 3)


def NeoHookean_imcompressible(mu, F):

    return mu / 2 * (dot(F, F) - 3)


def Penalty(e1, e2, I3):
    return e1 * (pow(I3, e2) + pow(I3, -e2) - 2)


def Tissue(k1, k2, J4):
    return k1 * \
        (exp(k2 * conditional(gt(J4, 1), pow((J4 - 1), 2), 0)) - 1) / k2 / 2


def define_domain(phi, point, threshold1, threshold2):
    value = phi(point)

    return threshold1 < value < threshold2


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
VV = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)

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
theta = math.pi / 3
a1 = math.cos(theta) * e3 + math.sin(theta) * e2
a2 = math.cos(theta) * e3 - math.sin(theta) * e2


# Define domain of three different layers
eps = DOLFIN_EPS
eps = 0.04
domain0 = AutoSubDomain(lambda x: define_domain(
    phi2, Point(x[0], x[1], x[2]), 0.0 - eps, 2 / 3 + eps))
domain1 = AutoSubDomain(lambda x: define_domain(
    phi2, Point(x[0], x[1], x[2]), 0.0 - eps, 2 / 3 + eps))
domain2 = AutoSubDomain(lambda x: define_domain(
    phi2, Point(x[0], x[1], x[2]), 2 / 3 - eps, 2 + eps))

# Have one function with tags of domains
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(2)

domain0.mark(domains, 1)
domain1.mark(domains, 2)
domain2.mark(domains, 3)
# mark domains
dx = Measure('dx', domain=mesh, subdomain_data=domains)

# Save sub domains to VTK files
file = File(mesh_dir + "subdomains.pvd")
file << domains


# Kinematics
d = u.geometric_dimension()
I = Identity(d)
F = I + grad(u)
C = variable(F.T * F)
M1 = outer(a1, a1)
M2 = outer(a2, a2)
J4_1 = tr(C * M1)
J4_2 = tr(C * M2)
I1 = tr(C)
I2 = 1 / 2 * (tr(C) * tr(C) - tr(C * C))
I3 = det(C)

# model parameters and material properties
c1_media = 120
c1_adventitia = 120

e1_media = 6
e2_media = 5
e1_adventitia = 7
e2_adventitia = 8.5

k1_media = 192.86
k2_media = 2626.84
k1_adventitia = 0.00368
k2_adventitia = 51.15

psi_NH_media = NeoHookean(c1_media, F)
psi_NH_adventitia = NeoHookean(c1_adventitia, F)
psi_P_media = Penalty(e1_media, e2_media, I3)
psi_P_adventitia = Penalty(e1_adventitia, e2_adventitia, I3)
psi_ti_1_media = Tissue(k1_media, k2_media, J4_1)
psi_ti_2_media = Tissue(k1_media, k2_media, J4_2)
psi_ti_1_adventitia = Tissue(k1_adventitia, k2_adventitia, J4_1)
psi_ti_2_adventitia = Tissue(k1_adventitia, k2_adventitia, J4_2)

psi_media = psi_NH_media + psi_P_media + psi_ti_1_media + psi_ti_2_media
psi_adventitia = psi_NH_adventitia + psi_P_adventitia + \
    psi_ti_1_adventitia + psi_ti_2_adventitia


# pressure
# P = Constant(1)
P = Expression("t", t=0.0, degree=0)

Pi = psi_media * dx(1) + psi_media * dx(2) + psi_adventitia * \
    dx(3) - dot(-P * n, u) * ds(1)
dPi = derivative(Pi, u, w)
J = derivative(dPi, u, v)
null_space = build_nullspace(VV)


# solve variational problem
problem = ProblemWithNullSpace(J, dPi, [], null_space)
solver = SolverWithNullSpace()

start_time = time.time()
T = 17
num_steps = 17
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

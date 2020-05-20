# system imports
from dolfin import *
from mshr import *
import math

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross
from utils import build_nullspace

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

# mesh
mesh = Mesh(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)

# mark boundaries
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary_inner = CompiledSubDomain(
    "near(sqrt(x[0]*x[0]+x[1]*x[1]), side, 0.1) && on_boundary", side=1.0)
boundary_inner.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

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
e1 = sqrt(inner(e1, e1))
e2 = sqrt(inner(e2, e2))
e3 = sqrt(inner(e3, e3))

# defin tissue orientation on the spatial varying basis
theta = math.pi / 6
a1 = math.cos(theta) * e3 + math.sin(theta) * e2
a2 = math.cos(theta) * e3 - math.sin(theta) * e2

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
eta1 = 141
eta2 = 160
eta3 = 3100
delta = 2 * eta1 + 4 * eta2 + 2 * eta3
e1 = 0.005
e2 = 10
k1 = 100000
k2 = 0.04

# strain energy functionals
psi_MR = eta1 * I1 + eta2 * I2 + eta3 * I3 - delta * ln(sqrt(I3))
psi_P = e1 * (pow(I3, e2) + pow(I3, -e2) - 2)
psi_ti_1 = k1 * \
    (exp(k2 * conditional(gt(J4_1, 1), pow((J4_1 - 1), 2), 0)) - 1) / k2 / 2
psi_ti_2 = k1 * \
    (exp(k2 * conditional(gt(J4_2, 1), pow((J4_2 - 1), 2), 0)) - 1) / k2 / 2
psi = psi_MR + psi_P + psi_ti_1 + psi_ti_2

# pressure
P = Constant(0.1)

# define variational problem
Pi = psi * dx - dot(-P * n, u) * ds(1)
dPi = derivative(Pi, u, w)
J = derivative(dPi, u, v)
null_space = build_nullspace(VV)

# solve variational problem
problem = ProblemWithNullSpace(J, dPi, [], null_space)
solver = SolverWithNullSpace()
solver.solve(problem, u.vector())

# write solution
file = File(output_dir + "displacements.pvd")
file << u
PK2 = 2.0 * diff(psi, C)
PK2Project = project(PK2, VVV)
file = XDMFFile(output_dir + "PK2.xdmf")
file.write(PK2Project, 0)

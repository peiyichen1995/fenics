from dolfin import *
import math
import ufl

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross
from utils import build_nullspace
from problems import CustomProblem
from solvers import CustomSolver

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

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

# Create mesh and define function space
mesh = Mesh(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)

# function space
V = FunctionSpace(mesh, 'CG', 2)
VV = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)


# Mark boundary subdomians
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=30.0)

# Define Dirichlet boundary (x = 0 or x = 1)
l = Expression(('0', '0', '0.0'), element=VV.ufl_element())
r = Expression(('1', '0', '0'), element=VV.ufl_element())

bc_r = DirichletBC(VV, r, right)
bc_l = DirichletBC(VV, l, left)
bcs = [bc_r, bc_l]

# Define functions
v = TrialFunction(VV)            # Incremental displacement
w = TestFunction(VV)             # Test function
u = Function(VV)                 # Displacement from previous iteration
phi1 = Function(V)
phi2 = Function(V)

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

# define tissue orientation on the spatial varying basis
theta = math.pi / 6
a1 = math.cos(theta) * e3 + math.sin(theta) * e2
a2 = math.cos(theta) * e3 - math.sin(theta) * e2

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
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
eta1_0 = 141
eta1_1 = 141
eta1_2 = 141

eta2 = 160
eta3 = 3100

delta_0 = 2 * eta1_0 + 4 * eta2 + 2 * eta3
delta_1 = 2 * eta1_1 + 4 * eta2 + 2 * eta3
delta_2 = 2 * eta1_2 + 4 * eta2 + 2 * eta3

# e1 = 0.1
# e2 = 1

k1 = 6.85
k2 = 754.01

# Stored strain energy density (compressible neo-Hookean model)
# psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
# compressible Mooney-Rivlin model
# psi_MR = eta1 * I1 + eta2 * I2 + eta3 * I3 - delta * ln(sqrt(I3))
psi_MR_0 = eta1_0 * I1 + eta2 * I2 + eta3 * I3 - delta_0 * ln(sqrt(I3))
psi_MR_1 = eta1_1 * I1 + eta2 * I2 + eta3 * I3 - delta_1 * ln(sqrt(I3))
psi_MR_2 = eta1_2 * I1 + eta2 * I2 + eta3 * I3 - delta_2 * ln(sqrt(I3))

# psi_P = e1 * (pow(I3, e2) + pow(I3, -e2) - 2)
psi_ti_1 = k1 * \
    (exp(k2 * conditional(gt(J4_1, 1), pow((J4_1 - 1), 2), 0)) - 1) / k2 / 2
psi_ti_2 = k1 * \
    (exp(k2 * conditional(gt(J4_2, 1), pow((J4_2 - 1), 2), 0)) - 1) / k2 / 2

# psi = psi_MR + psi_ti_1
psi_0 = psi_MR_0 + psi_ti_1 + psi_ti_2
psi_1 = psi_MR_1 + psi_ti_1 + psi_ti_2
psi_2 = psi_MR_2 + psi_ti_1 + psi_ti_2

# pressure
P = Constant(0.1)

# Total potential energy
# Pi = psi * dx  # - dot(-P * n, u) * ds(1)
Pi = psi_0 * dx

# Compute first variation of Pi (directional derivative about u in the direction of v)
dPi = derivative(Pi, u, w)

# Compute Jacobian of F
J = derivative(dPi, u, v)

problem = CustomProblem(J, dPi, bcs)
solver = CustomSolver()
solver.solve(problem, u.vector())

# write solution
file = File(output_dir + "displacements.pvd")
file << u

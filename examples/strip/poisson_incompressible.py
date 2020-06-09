from dolfin import *
import math
import ufl

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross, matrix_cofactor
from utils import build_nullspace
from problems import CustomProblem
from solvers import CustomSolver

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True


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

# Create mesh and define function space
mesh = Mesh(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)

# function space
V = FunctionSpace(mesh, 'CG', 2)
VV = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)


# Mark boundary subdomians
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=10.0)

# Define Dirichlet boundary (x = 0 or x = 1)
l = Expression(('0', '0', '0.0'), element=VV.ufl_element())
# r = Expression(('1', '0', '0'), element=VV.ufl_element())
r = Constant(0.5)

bc_r = DirichletBC(VV.sub(0), r, right)
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


mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02


# Total potential energy
psi = NeoHookean_imcompressible(
    mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F, V)
Pi = psi * dx

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

PK2 = 2.0 * diff(psi, C)
PK2Project = project(PK2, VVV)

file = XDMFFile(output_dir + "PK2Tensor.xdmf")
file.write(PK2Project, 0)

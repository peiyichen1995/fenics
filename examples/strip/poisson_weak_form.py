from solvers import CustomSolver
from problems import CustomProblem
from utils import build_nullspace
from utils import my_cross, matrix_cofactor
from solvers import SolverWithNullSpace
from problems import ProblemWithNullSpace
from dolfin import *
import math
import ufl
from ufl.operators import cell_avg


# my imports

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True


def NeoHookean_imcompressible(mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F):
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


def h_prime(delta, beta3):
    return beta3 * pow(delta, beta3 - 1) - beta3 * pow(delta, -beta3 - 1)

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
# VVV = TensorFunctionSpace(mesh, 'DG', 1)


# Mark boundary subdomians
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=10.0)

# Define Dirichlet boundary (x = 0 or x = 1)
l = Expression(('0', '0', '0.0'), element=VV.ufl_element())
r = Constant(5.0)

bc_l = DirichletBC(VV, l, left)
bc_r = DirichletBC(VV.sub(0), r, right)

bcs = [bc_r, bc_l]

# Define functions
v = TestFunction(VV)             # Test function
u = Function(VV)                 # Displacement from previous iteration
du = TrialFunction(VV)            # Incremental displacement

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
g3 = 9.7227

P = Constant(0.0)

# Total potential energy
psi = NeoHookean_imcompressible(
    mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F)

L = inner(P * det(F) * inv(F.T) * n, v) * ds
theta_bar = cell_avg(det(F))
# a = theta_bar * dx
# A = assemble(a)
p_bar = cell_avg(g3 * h_prime(theta_bar, beta3))
# a = p_bar * dx
# A = Constant(assemble(p_bar * dx))

delta_E = (grad(u).T * grad(v) + grad(v).T * grad(u)) / 2

S_isc = 2.0 * diff(psi, C) - g3 * sqrt(det(F)) * \
    h_prime(sqrt(det(F)), beta3) * inv(C)


# a = inner(S_isc + det(F) * h_prime(det(F), beta3) * inv(C), delta_E) * dx
a = inner(S_isc + det(F) * p_bar * inv(C), delta_E) * dx


# A = 0 * inner(S_isc + det(F) * p_bar * inv(C), delta_E) * dx

# for cell in cells(mesh):
#     theta_bar = cell_avg(det(F))
#     p_bar = cell_avg(g3 * h_prime(theta_bar, beta3))
#     a = cell.volume() * cell_avg(inner(S_isc + det(F) * p_bar * inv(C), delta_E))
#     A += a


F = a - L
J = derivative(F, u, du)
# problem = CustomProblem(J, F, bcs)
# solver = CustomSolver()
# solver.solve(problem, u.vector())


# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, u, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'

solver.solve()


# solve(F == 0, u, bcs, J=J)

# write solution
file = File(output_dir + "displacements.pvd")
file << u

# PK2 = 2.0 * diff(psi, C)
# PK2Project = project(PK2, VVV)
#
# file = XDMFFile(output_dir + "PK2Tensor.xdmf")
# file.write(PK2Project, 0)

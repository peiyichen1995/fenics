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
from utils import my_cross, matrix_cofactor
from utils import build_nullspace_three_field

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
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")
n = FacetNormal(mesh)
gdim = mesh.geometry().dim()

# mark boundaries
ds = Measure('ds', domain=mesh, subdomain_data=mf)

print('Number of nodes: ', mesh.num_vertices())
print('Number of cells: ', mesh.num_cells())

# function space
V_phi = FunctionSpace(mesh, 'CG', 2)
V = VectorElement("CG", mesh.ufl_cell(), 2)
Q = FiniteElement("DG", mesh.ufl_cell(), 0)
mixed_element = MixedElement([V, Q, Q])
W = FunctionSpace(mesh, mixed_element)

w = Function(W)
u, p, d = w.split()

d0e = Expression('1', degree=1)
d0 = interpolate(d0e, W.sub(2).collapse())
assign(d, d0)

u, p, d = split(w)

# functions
phi1 = Function(V_phi)
phi2 = Function(V_phi)

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
alpha = 46.274 / 180 * math.pi

a1 = cos(alpha) * e1 + sin(alpha) * e2
a2 = cos(alpha) * e1 - sin(alpha) * e2

# material parameters
mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
g3 = 9.7227

# pressure
P = Constant(1.0)
I = Identity(gdim)
F = grad(u) + I
J = det(F)
C = variable(F.T * F)
C_bar = pow(J, -2 / 3) * C


psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
psi_ti = mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
                                inner(F * a1, F * a1) - 1, 0), 2)) + mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
                                inner(F * a2, F * a2) - 1, 0), 2))
psi_theta = g3 * (pow(d, beta3) + pow(d, -beta3))
psi_p = p * (J - d)
psi = psi_bar + psi_theta + psi_ti + psi_p
R = derivative(psi * dx - dot(-P * n, u) * ds(1), w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))


null_space = build_nullspace_three_field(W, 0)

# solve variational problem
comm = MPI.comm_world
rank = comm.Get_rank()
if rank == 0:
    start_time = time.time()
# solve(R == 0, w, [], J=J_form,
#       form_compiler_parameters={"keep_diagonal": True})

# problem = ProblemWithNullSpace(J_form, R, [], null_space)
# solver = SolverWithNullSpace()
# solver.solve(problem, w.vector())

problem = NonlinearVariationalProblem(R, w, bcs=[], J=J_form)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.solve()

if rank == 0:
    end_time = time.time()
    print("solved using {0} seconds".format(end_time - start_time))

# write solution
u_sol, p_sol, d_sol = w.split()
file = File(output_dir + ".pvd")
file << p_sol

file = File(output_dir + ".pvd")
file << d_sol

file = File(output_dir + ".pvd")
file << u_sol

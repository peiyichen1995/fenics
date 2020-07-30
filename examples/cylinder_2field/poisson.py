
import time
from dolfin import *
import matplotlib.pyplot as plt
import os
import numpy as np
import math

from utils import my_cross, matrix_cofactor

# system imports
from dolfin import *
from mshr import *
import math
import time

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from problems import CustomProblem
from solvers import CustomSolver
from utils import my_cross
from utils import build_nullspace_three_field
from utils import my_cross, matrix_cofactor

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"


# traction
class Traction(UserExpression):
    def __init__(self):
        super().__init__(self)
        self.t = 0.0

    def eval(self, values, x):
        values[0] = 0 * self.t
        values[1] = 0.0
        values[2] = 0.0

    def value_shape(self):
        return (3,)

# Kinematics


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


def PK1Stress(u, pressure, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2):

    I = Identity(V.mesh().geometry().dim())  # Identity tensor
    F = I + grad(u)          # Deformation gradient
    # C = F.T * F                # Right Cauchy-Green tensor
    C = variable(F.T * F)
    F = variable(F)
    Ic = tr(C)               # Invariants of deformation tensors
    J = det(F)
    NH = NeoHookean_imcompressible(mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F)

    PK1 = diff(NH, F)
    return PK1, (J - 1), (F.T * F)


# Create mesh and define function space ============================================
output_dir = "./output/"
mesh_dir = "./mesh/"

# mesh
mesh = Mesh(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)
gdim = mesh.geometry().dim()

# mark boundaries
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary_inner = CompiledSubDomain(
    "near(sqrt(x[0]*x[0]+x[1]*x[1]), side, 0.1) && on_boundary", side=1.0)
boundary_inner.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ===============================================

V_phi = FunctionSpace(mesh, 'CG', 2)
phi1 = Function(V_phi)
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

a1 = as_vector([cos(alpha), sin(alpha), 0])
a2 = as_vector([cos(alpha), -sin(alpha), 0])


# Create function space
element_v = VectorElement("CG", mesh.ufl_cell(), 2)
element_s = FiniteElement("DG", mesh.ufl_cell(), 0)
mixed_element = MixedElement([element_v, element_s])
V = FunctionSpace(mesh, mixed_element)


# Define test and trial functions
dup = TrialFunction(V)
_u, _p = TestFunctions(V)

_u_p = Function(V)
u, p = split(_u_p)


# material parameters
mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
g3 = 9.7227
P = Constant(1.0)
# # Total potential energy
pkstrs, hydpress, C_s = PK1Stress(
    u, p, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
I = Identity(V.mesh().geometry().dim())
dgF = I + grad(u)
F1 = inner(pkstrs - p * inv(dgF).T, grad(_u)) * dx
F2 = hydpress * _p * dx
# F3 = derivative(- dot(-P * n, u) * ds(1), _u_p, dup)
F = F1 + F2 + inner(P * n, _u) * ds(1)
J = derivative(F, _u_p, dup)

null_space = build_nullspace_three_field(V, 0)

# solve variational problem
comm = MPI.comm_world
rank = comm.Get_rank()
if rank == 0:
    start_time = time.time()

problem = NonlinearVariationalProblem(F, _u_p, bcs=[], J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.solve()

# solve(R == 0, w, [], J=J_form,
#       form_compiler_parameters={"keep_diagonal": True})

# problem = ProblemWithNullSpace(J, F, [], null_space)
# solver = SolverWithNullSpace()
# solver.solve(problem, _u_p.vector())

# problem = CustomProblem(J, F, [])
# solver = CustomSolver()
# solver.solve(problem, _u_p.vector())

if rank == 0:
    end_time = time.time()
    print("solved using {0} seconds".format(end_time - start_time))

# write solution
u_sol, p_sol = _u_p.split()
file = File(output_dir + ".pvd")
file << p_sol

file = File(output_dir + ".pvd")
file << u_sol

# system imports
from dolfin import *
import matplotlib.pyplot as plt
import os
import numpy as np
import math

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
    C_bar = pow(J, -2 / 3) * C
    # temp1 = mu1 * tr(C) / pow(J, 2 / 3)
    temp1 = mu1 * tr(C_bar)
    # temp2 = mu2 * pow(sqrt(tr(matrix_cofactor(F).T *
    #                           matrix_cofactor(F))), 3) / pow(J, 2)
    temp2 = mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
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


def geometry_3d(mesh_dir):
    mesh = Mesh(mesh_dir)

    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    x0 = AutoSubDomain(lambda x: near(x[0], 0))
    x1 = AutoSubDomain(lambda x: near(x[0], 10))
    y0 = AutoSubDomain(lambda x: near(x[1], 0))
    z0 = AutoSubDomain(lambda x: near(x[2], 0))
    x0.mark(boundary_parts, 1)
    y0.mark(boundary_parts, 2)
    z0.mark(boundary_parts, 3)
    x1.mark(boundary_parts, 4)
    return boundary_parts


def h_prime(delta, beta3):
    return beta3 * pow(delta, beta3 - 1) - beta3 * pow(delta, -beta3 - 1)


# Create mesh and define function space ============================================
output_dir = "./output/"
mesh_dir = "./mesh/"

facet_function = geometry_3d(mesh_dir + "mesh.xml")
mesh = facet_function.mesh()
n = FacetNormal(mesh)
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", subdomain_data=facet_function, subdomain_id=4)
print('Number of nodes: ', mesh.num_vertices())
print('Number of cells: ', mesh.num_cells())

# V = VectorFunctionSpace(mesh, "CG", 2)
# Q = FunctionSpace(mesh, "CG", 1)
# W = MixedFunctionSpace([V, Q, Q])

# ===============================================

# V = FunctionSpace(mesh, 'CG', 2)
# phi1 = Function(V)
# phi2 = Function(V)

# read in laplace solutions
# phi1_h5 = HDF5File(MPI.comm_world, output_dir + "phi1.h5", "r")
# phi2_h5 = HDF5File(MPI.comm_world, output_dir + "phi2.h5", "r")
# phi1_h5.read(phi1, "phi1")
# phi2_h5.read(phi2, "phi2")
# phi1_h5.close()
# phi2_h5.close()

# define orthorgonal basis
# e3 = grad(phi1)
# e1 = grad(phi2)
# e2 = my_cross(e3, e1)

# # normalize basis
# e1 = e1 / sqrt(inner(e1, e1))
# e2 = e2 / sqrt(inner(e2, e2))
# e3 = e3 / sqrt(inner(e3, e3))

# define tissue orientation on the spatial varying basis
theta = math.pi / 3
# a1 = math.cos(theta) * e3 + math.sin(theta) * e2
# a2 = math.cos(theta) * e3 - math.sin(theta) * e2

a1 = as_vector([cos(theta), -sin(theta), 0])
a2 = as_vector([cos(theta), sin(theta), 0])
# ===============================================


V = VectorElement("CG", mesh.ufl_cell(), 2)
Q = FiniteElement("CG", mesh.ufl_cell(), 1)
# mixed_element = MixedElement([element_v, element_s, element_s])
mixed_element = MixedElement([V, Q, Q])
W = FunctionSpace(mesh, mixed_element)

w = Function(W)
u, p, d = w.split()

# fa = FunctionAssigner(W.sub(2), Q)
# fa.assign(d, interpolate(Constant(1.0), Q))

d0e = Expression('1', degree=1)
d0 = interpolate(d0e, W.sub(2).collapse())
assign(d, d0)

u, p, d = split(w)

# initial guess
w_n = Function(W)
u_n, p_n, d_n = w_n.split()

d0e_n = Expression('1', degree=1)
d0_n = interpolate(d0e_n, W.sub(2).collapse())
assign(d_n, d0_n)

u_n, p_n, d_n = split(w_n)

# Displacement from previous iteration
b = Constant((0.0, 0.0, 0.0))  # Body force per unit mass
h = Traction()  # Traction force on the boundary


# Define Dirichlet boundary
bc0 = DirichletBC(W.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(W.sub(0).sub(1), Constant(0.), facet_function, 2)
bc2 = DirichletBC(W.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('2.5*time_'), time_=0.1, degree=0)
bc3 = DirichletBC(W.sub(0).sub(0), tDirBC, facet_function, 4)
bcs = [bc0, bc1, bc2, bc3]

# material parameters
mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
g3 = 9.7227

# pressure
P = Constant(0.0)

F = grad(u) + Identity(3)
J = det(F)
C = F.T * F
C_bar = pow(J, -2 / 3) * C

F_n = grad(u_n) + Identity(3)
J_n = det(F_n)
C_n = F_n.T * F_n
C_bar_n = pow(J_n, -2 / 3) * C_n
# =============================================
# C1 = Constant(2.5)
# lmbda = Constant(30.0)

# psi = C1 * (tr(C) - 3) + p * (J - d) + lmbda * (ln(d)**2)

psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
psi_ti = mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
                                inner(F * a1, F * a1) - 1, 0), 2)) + mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
                                inner(F * a2, F * a2) - 1, 0), 2))
psi_theta = g3 * (pow(J, beta3) + pow(J, -beta3))

psi = psi_bar + psi_theta + psi_ti

# ======================================
psi_bar_n = mu1 * tr(C_bar_n) + mu2 * pow(tr(matrix_cofactor(C_bar_n)), 3 / 2)
psi_ti_n = mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F_n * a1, F_n * a1), 1),
                                inner(F_n * a1, F_n * a1) - 1, 0), 2)) + mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F_n * a2, F_n * a2), 1),
                                inner(F_n * a2, F_n * a2) - 1, 0), 2))
psi_theta_n = g3 * (pow(J_n, beta3) + pow(J_n, -beta3))

psi_n = psi_bar_n + psi_theta_n + psi_ti_n

R_n = derivative((psi_n) * dx, w_n, TestFunction(W))
J_form_n = derivative(R_n, w_n, TrialFunction(W))

# Create nonlinear variational problem and solve
problem_n = NonlinearVariationalProblem(R_n, w_n, bcs=bcs, J=J_form_n)
solver_n = NonlinearVariationalSolver(problem_n)
solver_n.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver_n.parameters['newton_solver']['linear_solver'] = 'mumps'

solver_n.solve()
exit()
# ======================================

R = derivative((psi - psi_n) * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))

# solve(R == 0, w, bcs,
#       form_compiler_parameters={"keep_diagonal": True})
#
# exit()

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(R, w, bcs=bcs, J=J_form)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'

# solver.solve()

# Time stepping parameters
dt = 0.2
t, T = 0.1, 10 * dt

# Save solution in VTK format
file_results = XDMFFile("./Results/TestUniaxialLoading/Uniaxial.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


while t <= T:
    print('time: ', t)

    # Increase traction
    h.t = t
    tDirBC.time_ = t

    # solve and save disp
    solver.solve()

    # Save solution
    u_solu, p_solu, d_solu = w.split()
    file = File(output_dir + "theta_step" + str(t) + ".pvd")
    file << d_solu

    # time increment
    t += float(dt)

    assign(w_n, w)

u_plot = w.split()
file = File(output_dir + "displacements.pvd")
file << u_plot

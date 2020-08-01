# system imports
import time
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


def geometry_3d(mesh_dir):
    # mesh = Mesh(mesh_dir)
    # mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10., 3, 0.5), 5, 5, 5)
    mesh = UnitCubeMesh(10, 10, 10)
    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    x0 = AutoSubDomain(lambda x: near(x[0], 0))
    x1 = AutoSubDomain(lambda x: near(x[0], 1))
    x0.mark(boundary_parts, 1)
    x1.mark(boundary_parts, 2)

    y0 = AutoSubDomain(lambda x: near(x[1], 0))
    y1 = AutoSubDomain(lambda x: near(x[1], 1))
    y0.mark(boundary_parts, 3)
    y1.mark(boundary_parts, 4)

    z0 = AutoSubDomain(lambda x: near(x[2], 0))
    z1 = AutoSubDomain(lambda x: near(x[2], 1))
    z0.mark(boundary_parts, 5)
    z1.mark(boundary_parts, 6)

    return mesh, boundary_parts


def cauchy(mu1, mu2, mu4, mu3, beta3, beta4, a1, a2, F, p, theta):
    F = variable(F)
    J = det(F)
    C = F.T * F
    C_bar = pow(J, -2 / 3) * C

    psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
    # psi_ti = mu4 / beta4 * \
    #     exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
    #                                 inner(F * a1, F * a1) - 1, 0), 2)) + mu4 / beta4 * \
    #     exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
    #                                 inner(F * a2, F * a2) - 1, 0), 2))
    psi_theta = mu3 * (pow(theta, beta3) + pow(theta, -beta3))
    psi = psi_bar + psi_theta  # + psi_ti
    PK1 = diff(psi, F)
    return PK1 * F.T / J


# Create mesh and define function space ============================================
output_dir = "./output/"
mesh_dir = "./mesh/"

mesh, facet_function = geometry_3d(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", subdomain_data=facet_function)
print('Number of nodes: ', mesh.num_vertices())
print('Number of cells: ', mesh.num_cells())

# define tissue orientation on the spatial varying basis
theta = 46.274 / 180 * math.pi

a1 = as_vector([cos(theta), sin(theta), 0])
a2 = as_vector([cos(theta), -sin(theta), 0])
# ===============================================


V = VectorElement("CG", mesh.ufl_cell(), 2)
Q = FiniteElement("DG", mesh.ufl_cell(), 0)
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


c = Expression(('0.1*exp(x[1]+x[2])', '0', '0'), degree=2)

# Displacement from previous iteration
b = Expression(('- (exp(x[1] + x[2]) * (9 * sqrt(2) * pow(exp(2 * x[1] + 2 * x[2]) + 150, 1 / 2) + 80)) / 50 - \
    (9 * sqrt(2) * exp(3 * x[1] + 3 * x[2])) / \
    (50 * pow(exp(2 * x[1] + 2 * x[2]) + 150, 1 / 2))', '(exp(2 * x[1] + 2 * x[2]) * (27 * sqrt(2) * exp(2 * x[1] + \
    2 * x[2]) + 80 * pow(exp(2 * x[1] + 2 *x[2]) + 150, 1 / 2) + 2700 * sqrt(2))) / (750 * pow(exp(2 * x[1] + 2 * x[2]) + 150, \
    1 / 2))', '(exp(2 * x[1] + 2 * x[2]) * (27 * sqrt(2) * exp(2 * x[1] + 2 * x[2]) + 80 * pow(exp(2 * x[1] + 2 * x[2]) + 150, 1 / 2) + 2700 * sqrt(2))) / (750 * pow(exp(2 * x[1] + 2 * x[2]) + 150, 1 / 2))'), degree=2)

# h = Traction()  # Traction force on the boundary


# Define Dirichlet boundary
bc_1 = DirichletBC(W.sub(0), c, facet_function, 1)
bc_2 = DirichletBC(W.sub(0), c, facet_function, 2)
bc_3 = DirichletBC(W.sub(0), c, facet_function, 3)
bc_4 = DirichletBC(W.sub(0), c, facet_function, 4)
bc_5 = DirichletBC(W.sub(0), c, facet_function, 5)
bc_6 = DirichletBC(W.sub(0), c, facet_function, 6)
bcs = [bc_1, bc_2, bc_3, bc_4, bc_5, bc_6]

# material parameters
mu1 = 4
mu2 = 3
mu3 = 10
mu4 = 19.285
beta3 = 4
beta4 = 500.02

# pressure
P = Constant(0.0)
I = Identity(gdim)
F = grad(u) + I
J = det(F)
C = variable(F.T * F)
C_bar = pow(J, -2 / 3) * C


# psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(det(C_bar) * inv(C_bar)), 3 / 2)
# psi_ti = mu4 / beta4 * \
#     exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
#                                 inner(F * a1, F * a1) - 1, 0), 2)) + mu4 / beta4 * \
#     exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
#                                 inner(F * a2, F * a2) - 1, 0), 2))
psi_theta = mu3 * (pow(d, beta3) + pow(d, -beta3))
# psi_p = p * (J - d)
psi_p = p * (J - d)
psi = psi_bar + psi_theta + psi_p
R = derivative(psi * dx - dot(b, u) * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))

solve(R == 0, w, bcs, J=J_form,
      form_compiler_parameters={"keep_diagonal": True})

u_sol, p_sol, d_sol = w.split()

print("err:")
u_e = interpolate(c, W.sub(0).collapse())
error = (u_sol - u_e)**2 * dx
L2_err = sqrt(abs(assemble(error)))
print(L2_err)

file = File(output_dir + "MMS_displacement.pvd")
file << u_sol

# solve(R == 0, w, bcs, J=J_form,
#       form_compiler_parameters={"keep_diagonal": True})
exit()

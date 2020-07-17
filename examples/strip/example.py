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

V = FunctionSpace(mesh, 'CG', 2)
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
theta = math.pi / 3
a1 = math.cos(theta) * e3 + math.sin(theta) * e2
a2 = math.cos(theta) * e3 - math.sin(theta) * e2


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

# Displacement from previous iteration
b = Constant((0.0, 0.0, 0.0))  # Body force per unit mass
h = Traction()  # Traction force on the boundary


# Define Dirichlet boundary
bc0 = DirichletBC(W.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(W.sub(0).sub(1), Constant(0.), facet_function, 2)
bc2 = DirichletBC(W.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('2.0*time_'), time_=0.1, degree=0)
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
psi = psi_bar + psi_ti + psi_theta
R = derivative(psi * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))

solve(R == 0, w, bcs,
      form_compiler_parameters={"keep_diagonal": True})
# plot(split(w)[0], interactive=True, mode="displacement")

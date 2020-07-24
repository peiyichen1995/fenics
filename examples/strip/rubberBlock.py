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


def geometry_3d(mesh_dir):
    # mesh = Mesh(mesh_dir)
    mesh = UnitSquareMesh(10, 10)

    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    x0 = AutoSubDomain(lambda x: near(x[0], 0))
    x1 = AutoSubDomain(lambda x: near(x[0], 1))
    y0 = AutoSubDomain(lambda x: near(x[1], 0))
    y1 = AutoSubDomain(lambda x: near(x[1], 1))
    x0.mark(boundary_parts, 1)
    y0.mark(boundary_parts, 2)
    x1.mark(boundary_parts, 3)
    y1.mark(boundary_parts, 4)
    return boundary_parts


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


# Define Dirichlet boundary
bc0 = DirichletBC(W.sub(0).sub(0), Constant(0.), facet_function, 2)
bc1 = DirichletBC(W.sub(0).sub(1), Constant(0.), facet_function, 2)
# bc2 = DirichletBC(W.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('3*time_'), time_=0.1, degree=0)
bc3 = DirichletBC(W.sub(0).sub(1), tDirBC, facet_function, 4)
bcs = [bc1, bc3]

# material parameters
c = 2.5
d = 5
mu = 6.3

# pressure
I = Identity(W.mesh().geometry().dim())
F = grad(u) + I
J = det(F)
C = F.T * F
C_bar = pow(J, -2 / 3) * C

psi_NH = c * (pow(J, 2) - 1) - d * ln(J) - mu * \
    ln(J) + 1 / 2 * mu * (tr(C) - 3)
psi_theta = p * (J - d)
psi = psi_NH + psi_theta
R = derivative(psi * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))


solve(R == 0, w, bcs,
      form_compiler_parameters={"keep_diagonal": True})

u_solu, p_solu, d_solu = w.split()
file = File(output_dir + "2Drubber.pvd")
file << u_solu

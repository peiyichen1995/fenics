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
    mesh = RectangleMesh(Point(0, 0), Point(10, 10), 30, 30)

    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    y0 = AutoSubDomain(lambda x: near(x[1], 0))
    y1 = AutoSubDomain(lambda x: near(x[1], 10))
    y0.mark(boundary_parts, 1)
    y1.mark(boundary_parts, 2)
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
Q = FiniteElement("DG", mesh.ufl_cell(), 1)
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
bc0 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_function, 1)
# bc2 = DirichletBC(W.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('-3*time_'), time_=1, degree=0)
bc1 = DirichletBC(W.sub(0).sub(1), tDirBC, facet_function, 2)
bcs = [bc0, bc1]

# material parameters
Lambda = 10
mu = 6.3
Bulk = Lambda + 2 / 3 * mu

# pressure
I = Identity(gdim)
F = grad(u) + I
J = det(F)
Fh = pow(d / J, 1 / gdim) * F
C = F.T * F
Ch = Fh.T * Fh
C_bar = pow(J, -2 / 3) * C

# psi_NH = 1 / 2 * mu * (tr(C_bar) - 3)
# psi_theta = Bulk / 4 * (d * d - 1) - Bulk / 2 * ln(d)
# psi_theta = Bulk / 2 * (d - 1) * (d - 1)
# psi_theta = Bulk / 2 * (d - 1) * (d - 1)
# psi_p = p * (J - d)
# psi_original = Lambda / 4 * (d * d - 1) - Lambda / 2 * \
# ln(d) - mu * ln(d) + 1 / 2 * mu * (tr(C) - 3)

# psi = psi_original + psi_p
psi = Lambda / 4 * (d * d - 1) - Lambda / 2 * ln(d) - \
    mu * ln(d) + 1 / 2 * mu * (tr(Ch) - 3) + p * (J - d)
# psi = psi_NH + psi_theta + psi_p
R = derivative(psi * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))


solve(R == 0, w, bcs, J=J_form,
      form_compiler_parameters={"keep_diagonal": True})

u_solu, p_solu, d_solu = w.split()
file = File(output_dir + "2Drubber_u.pvd")
file << u_solu

file = File(output_dir + "2Drubber_p.pvd")
file << p_solu

file = File(output_dir + "2Drubber_d.pvd")
file << d_solu

exit()

# Time stepping parameters
dt = 0.1
t, T = 0.0, 10 * dt

# Save solution in VTK format
file_results = XDMFFile("./Results/TestUniaxialLoading/Uniaxial.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

stretch_vec = []
cauchy_stress = []

while t <= T:
    print('time: ', t)

    # increase load
    tDirBC.time_ = t

    # solve and save disp
    solve(R == 0, w, bcs, J=J_form,
          form_compiler_parameters={"keep_diagonal": True})

    # time increment
    t += float(dt)

u_solu, p_solu, d_solu = w.split()
file = File(output_dir + "2Drubberwa.pvd")
file << u_solu

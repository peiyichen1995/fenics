
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
    mesh = RectangleMesh(Point(0, 0), Point(10, 10), 5, 5)

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
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", subdomain_data=facet_function, subdomain_id=4)
print('Number of nodes: ', mesh.num_vertices())
print('Number of cells: ', mesh.num_cells())


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


# Define Dirichlet boundary
bc0 = DirichletBC(V.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), facet_function, 1)
# bc2 = DirichletBC(W.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('3*time_'), time_=0.1, degree=0)
bc2 = DirichletBC(V.sub(0).sub(1), tDirBC, facet_function, 2)
bcs = [bc0, bc1, bc2]

# Displacement from previous iteration
b = Constant((0.0, 0.0))  # Body force per unit mass
h = Constant((0.0, 0.0))  # Traction force on the boundary

# material parameters
c = 2.5
parameter_d = 5
mu = 6.3

# # Total potential energy
I = Identity(V.mesh().geometry().dim())
dgF = I + grad(u)
dgF = variable(dgF)
J = det(dgF)
C = dgF.T * dgF
psi = c * (pow(J, 2) - 1) - parameter_d * \
    ln(J) - mu * ln(J) + 1 / 2 * (tr(C) - 3)
pkstrs = diff(psi, dgF)
F1 = inner(pkstrs - p * inv(dgF).T, grad(_u)) * \
    dx - dot(b, _u) * dx - dot(h, _u) * ds
F2 = (J - 1) * _p * dx
F = F1 + F2
J = derivative(F, _u_p, dup)

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, _u_p, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'


# Time stepping parameters
dt = 0.1
t, T = 0.0, 20 * dt

# Save solution in VTK format
file_results = XDMFFile("./Results/TestUniaxialLoading/Uniaxial.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

solver.solve()

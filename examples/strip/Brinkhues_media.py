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
theta = 43.47 / 180 * math.pi
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

# Displacement from previous iteration
b = Constant((0.0, 0.0, 0.0))  # Body force per unit mass
h = Traction()  # Traction force on the boundary


# Define Dirichlet boundary
bc0 = DirichletBC(W.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(W.sub(0).sub(1), Constant(0.), facet_function, 2)
bc2 = DirichletBC(W.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('2.5*time_'), time_=1, degree=0)
bc3 = DirichletBC(W.sub(0).sub(0), tDirBC, facet_function, 4)
bcs = [bc0, bc1, bc2, bc3]

# material parameters
eta1 = 141
eta2 = 160
eta3 = 3100
delta = 2 * eta1 + 4 * eta2 + 2 * eta3
k1 = 6.85
k2 = 754.01
# pressure
P = Constant(0.0)

F = grad(u) + Identity(3)
J = det(F)
C = F.T * F
C_bar = pow(J, -2 / 3) * C
I1 = tr(C)
I2 = 1 / 2 * (tr(C) * tr(C) - tr(C * C))
I3 = det(C)
M1 = outer(a1, a1)
M2 = outer(a2, a2)
J4_1 = tr(C * M1)
J4_2 = tr(C * M2)

psi_MR = eta1 * I1 + eta2 * I2 + eta3 * I3 - delta * ln(sqrt(I3))
psi_ti = k1 / 2 / k2 * \
    (exp(k2 * conditional(gt(pow(J4_1, 2), 1), pow(pow(J4_1, 2) - 1, 2), 0)) - 1 +
     exp(k2 * conditional(gt(pow(J4_2, 2), 1), pow(pow(J4_2, 2) - 1, 2), 0)) - 1)
psi_theta = p * (J - d)
psi = psi_MR + psi_ti + psi_theta
R = derivative(psi * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))


solve(R == 0, w, bcs,
      form_compiler_parameters={"keep_diagonal": True})

# Create nonlinear variational problem and solve
# problem = NonlinearVariationalProblem(R, w, bcs=bcs, J=J_form)
# solver = NonlinearVariationalSolver(problem)
# solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
# solver.parameters['newton_solver']['linear_solver'] = 'mumps'
#
# solver.solve()
exit()

# Time stepping parameters
dt = 0.1
t, T = 0.0, 10 * dt

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

    # time increment
    t += float(dt)

u_plot = w.split()
file = File(output_dir + "displacements.pvd")
file << u_plot

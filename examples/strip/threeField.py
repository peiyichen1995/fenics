
from dolfin import *
import matplotlib.pyplot as plt
import os
import numpy as np
import math

from utils import my_cross, matrix_cofactor


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


def pk2Stress(u, pressure, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2):

    I = Identity(V.mesh().geometry().dim())  # Identity tensor
    F = I + grad(u)          # Deformation gradient
    # C = F.T * F                # Right Cauchy-Green tensor
    C = variable(F.T * F)
    F = variable(F)
    Ic = tr(C)               # Invariants of deformation tensors
    J = det(F)
    NH = NeoHookean_imcompressible(mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F)
    PK2 = 2.0 * diff(NH, C)
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

# Limit quadrature degree
dx = dx(degree=4)
ds = ds(degree=4)


# Create function space
element_v = VectorElement("CG", mesh.ufl_cell(), 2)
element_s = FiniteElement("DG", mesh.ufl_cell(), 0)
# mixed_element = MixedElement([element_v, element_s, element_s])
mixed_element = MixedElement([element_v, element_s])
V = FunctionSpace(mesh, mixed_element)


# Define test and trial functions
dup = TrialFunction(V)
# _u, _p, _q = TestFunctions(V)
_u, _p = TestFunctions(V)

_u_p = Function(V)
# u, p, q = split(_u_p)
u, p = split(_u_p)


# Create tensor function spaces for stress and stretch output
W_DFnStress = TensorFunctionSpace(mesh, "DG", degree=1)
defGrad = Function(W_DFnStress, name='F')
PK1_stress = Function(W_DFnStress, name='PK1')
C_stress = Function(W_DFnStress, name='Cauchy')

# Displacement from previous iteration
b = Constant((0.0, 0.0, 0.0))  # Body force per unit mass
h = Traction()  # Traction force on the boundary


# Define Dirichlet boundary
bc0 = DirichletBC(V.sub(0).sub(0), Constant(0.), facet_function, 1)
bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), facet_function, 2)
bc2 = DirichletBC(V.sub(0).sub(2), Constant(0.), facet_function, 3)

tDirBC = Expression(('2.0*time_'), time_=0.0, degree=0)
bc3 = DirichletBC(V.sub(0).sub(0), tDirBC, facet_function, 4)
bcs = [bc0, bc1, bc2, bc3]

# material parameters
E, nu = 1e3, 0.5
mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
g3 = 9.7227

# pressure
P = Constant(0.0)

# # Total potential energy
pkstrs, hydpress, C_s = pk2Stress(
    u, p, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
I = Identity(V.mesh().geometry().dim())
dgF = I + grad(u)
C = variable(dgF.T * dgF)

psi = NeoHookean_imcompressible(
    mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, dgF)

S_isc = 2.0 * diff(psi, C) - g3 * sqrt(det(dgF)) * \
    h_prime(sqrt(det(dgF)), beta3) * inv(C)

delta_E = (grad(u).T * grad(_u) + grad(_u).T * grad(u)) / 2

L = inner(P * det(dgF) * inv(dgF.T) * n, _u) * ds


F1 = inner(S_isc + det(dgF) * p * inv(C), delta_E) * dx
# F2 = (q - det(dgF)) * _q * dx
F3 = (p - g3 * h_prime(det(dgF), beta3)) * _p * dx
F = F1 + F3 - L
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

stretch_vec = []
stress_vec = []
stress_ana = []
cauchy_stress = []

while t <= T:
    print('time: ', t)

    # Increase traction
    h.t = t
    tDirBC.time_ = t

    # solve and save disp
    solver.solve()

    # Extract solution components
    u_plot, p_plot = _u_p.split()
    u_plot.rename("u", "displacement")
    p_plot.rename("p", "pressure")

    # get stretch at a point for plotting
    point = (5, 1.5, 0)
    DF = I + grad(u_plot)
    defGrad.assign(project(DF, W_DFnStress))
    stretch_vec.append(defGrad(point)[0])

    # get stress at a point for plotting
    PK1_s, thydpress, C_s = pk2Stress(
        u_plot, p_plot, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
    # PK1_stress.assign(project(PK1_s, W_DFnStress))
    C_stress.assign(project(C_s, W_DFnStress))
    # stress_vec.append(PK1_stress(point)[0])
    cauchy_stress.append(C_stress(point)[0])

    # save xdmf file
    file_results.write(u_plot, t)
    file_results.write(defGrad, t)
    # file_results.write(PK1_stress, t)

    # time increment
    t += float(dt)


# get analytical solution
stretch_vec = np.array(stretch_vec)
# stress_vec = np.array(stress_vec)
cauchy_stress = np.array(cauchy_stress)
# G = E / (2 * (1 + nu))
# c1 = G / 2.0
# for i in range(len(stretch_vec)):
#     pk1_ana = 2 * c1 * (stretch_vec[i] - 1 /
#                         (stretch_vec[i] * stretch_vec[i]))  # PK1
#     pk2_ana = (1 / stretch_vec[i]) * pk1_ana  # PK2
#     stress_ana.append(pk2_ana)
# stress_ana = np.array(stress_ana)

# plot results
f = plt.figure(figsize=(12, 6))
plt.plot(stretch_vec, cauchy_stress, 'r-')
plt.xlabel("stretch")
plt.ylabel("cauchy stress")
plt.savefig('test1.png')
# plt.plot(stretch_vec, stress_ana, 'k.')
# plt.xlabel("stretch")
# plt.ylabel("PK2")
# plt.savefig('test2.png')

file = File(output_dir + "displacements.pvd")
file << u_plot

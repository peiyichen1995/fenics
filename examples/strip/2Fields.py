
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


def geometry_3d(mesh_dir):
    # mesh = Mesh(mesh_dir)
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10., 3, 0.5), 5, 5, 5)

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
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", subdomain_data=facet_function, subdomain_id=4)
print('Number of nodes: ', mesh.num_vertices())
print('Number of cells: ', mesh.num_cells())


# define tissue orientation on the spatial varying basis
# theta = math.pi / 3
#
# a1 = as_vector([cos(theta), sin(theta), 0])
# a2 = as_vector([cos(theta), -sin(theta), 0])

theta = 46.274 / 180 * math.pi

a1 = as_vector([cos(theta), sin(theta), 0])
a2 = as_vector([cos(theta), -sin(theta), 0])

# ===============================================


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
bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), facet_function, 1)
bc2 = DirichletBC(V.sub(0).sub(2), Constant(0.), facet_function, 1)

tDirBC = Expression(('3*time_'), time_=0.0, degree=0)
bc3 = DirichletBC(V.sub(0).sub(0), tDirBC, facet_function, 4)
bcs = [bc0, bc1, bc2, bc3]

# material parameters
mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
g3 = 9.7227

# # Total potential energy
pkstrs, hydpress, C_s = PK1Stress(
    u, p, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
I = Identity(V.mesh().geometry().dim())
dgF = I + grad(u)
F1 = inner(pkstrs - p * inv(dgF).T, grad(_u)) * dx
F2 = hydpress * _p * dx
F = F1 + F2
J = derivative(F, _u_p, dup)

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, _u_p, bcs=bcs, J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'


start_time = time.time()

# Time stepping parameters
dt = 0.05
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
    tDirBC.time_ = t

    # solve and save disp
    solver.solve()

    # Extract solution components
    u_plot, p_plot = _u_p.split()
    u_plot.rename("u", "displacement")
    p_plot.rename("p", "pressure")

    # get stretch at a point for plotting
    point = (5, 1.5, 0.25)
    DF = I + grad(u_plot)
    defGrad.assign(project(DF, W_DFnStress))
    stretch_vec.append(defGrad(point)[0])

    # get stress at a point for plotting
    PK1_s, thydpress, C_s = PK1Stress(
        u_plot, p_plot, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
    # PK1_stress.assign(project(PK1_s, W_DFnStress))
    C_stress.assign(
        project(1 / det(DF) * (PK1_s - p_plot * inv(DF).T) * DF.T, W_DFnStress))
    # stress_vec.append(PK1_stress(point)[0])
    cauchy_stress.append(C_stress(point)[0])

    # save xdmf file
    file_results.write(u_plot, t)
    file_results.write(defGrad, t)
    # file_results.write(PK1_stress, t)

    # time increment
    t += float(dt)
print("dofs: ")
print(assemble(J).size(0))
print("time: ")
print(time.time() - start_time)

# get analytical solution
stretch_vec = np.array(stretch_vec)
# stress_vec = np.array(stress_vec)
cauchy_stress = np.array(cauchy_stress)


# plot results
f = plt.figure(figsize=(12, 6))
plt.plot(stretch_vec, cauchy_stress, 'r-')
plt.xlabel("stretch")
plt.ylabel("cauchy stress")
plt.savefig('penaltyCauchyStress.png')
# plt.plot(stretch_vec, stress_ana, 'k.')
# plt.xlabel("stretch")
# plt.ylabel("PK2")
# plt.savefig('test2.png')

file = File(output_dir + "penaltyDisplacements.pvd")
file << u_plot

for i in range(len(stretch_vec)):
    print(str(1 + 0.015 * i) + "," + str(cauchy_stress[i]))

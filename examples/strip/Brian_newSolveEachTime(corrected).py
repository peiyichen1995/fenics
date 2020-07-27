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


def geometry_3d(mesh_dir):
    # mesh = Mesh(mesh_dir)
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(10., 3, 0.5), 5, 5, 5)

    boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    x0 = AutoSubDomain(lambda x: near(x[0], 0))
    x1 = AutoSubDomain(lambda x: near(x[0], 10))
    x0.mark(boundary_parts, 1)
    x1.mark(boundary_parts, 2)

    return mesh, boundary_parts


def cauchy(mu1, mu2, mu4, g3, beta3, beta4, a1, a2, F, p, theta):
    F = variable(F)
    J = det(F)
    C = F.T * F
    C_bar = pow(J, -2 / 3) * C

    psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
    psi_ti = mu4 / beta4 * \
        exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
                                    inner(F * a1, F * a1) - 1, 0), 2)) + mu4 / beta4 * \
        exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
                                    inner(F * a2, F * a2) - 1, 0), 2))
    psi_theta = g3 * (pow(theta, beta3) + pow(theta, -beta3))
    psi = psi_bar + psi_theta + psi_ti + p * (J - theta)
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
Q = FiniteElement("CG", mesh.ufl_cell(), 1)
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

# Create tensor function spaces for stress and stretch output
W_DFnStress = TensorFunctionSpace(mesh, "DG", degree=1)
defGrad = Function(W_DFnStress, name='F')
PK1_stress = Function(W_DFnStress, name='PK1')
C_stress = Function(W_DFnStress, name='Cauchy')

# Displacement from previous iteration
b = Constant((0.0, 0.0, 0.0))  # Body force per unit mass
h = Traction()  # Traction force on the boundary


# Define Dirichlet boundary
bc_fixed = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), facet_function, 1)

tDirBC = Expression(('3*time_'), time_=0.0, degree=0)
bc_dispaced = DirichletBC(W.sub(0).sub(0), tDirBC, facet_function, 2)
bcs = [bc_fixed, bc_dispaced]

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
I = Identity(gdim)
F = grad(u) + I
J = det(F)
C = variable(F.T * F)
C_bar = pow(J, -2 / 3) * C


psi_bar = mu1 * tr(C_bar) + mu2 * pow(tr(matrix_cofactor(C_bar)), 3 / 2)
psi_ti = mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F * a1, F * a1), 1),
                                inner(F * a1, F * a1) - 1, 0), 2)) + mu4 / beta4 * \
    exp(beta4 * pow(conditional(gt(inner(F * a2, F * a2), 1),
                                inner(F * a2, F * a2) - 1, 0), 2))
psi_theta = g3 * (pow(d, beta3) + pow(d, -beta3))
psi_p = p * (J - d)
psi = psi_bar + psi_theta + psi_ti + psi_p
R = derivative(psi * dx, w, TestFunction(W))
J_form = derivative(R, w, TrialFunction(W))

# solve(R == 0, w, bcs,
#       form_compiler_parameters={"keep_diagonal": True})


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

    # Increase traction
    h.t = t
    tDirBC.time_ = t

    # solve and save disp
    solve(R == 0, w, bcs, J=J_form,
          form_compiler_parameters={"keep_diagonal": True})

    # Save solution
    u_sol, p_sol, d_sol = w.split()
    file = File(output_dir + "p_step" + str(10 * t) + ".pvd")
    file << p_sol

    file = File(output_dir + "theta_step" + str(10 * t) + ".pvd")
    file << d_sol

    file = File(output_dir + "u_step" + str(10 * t) + ".pvd")
    file << u_sol

    # get stretch at a point for plotting
    point = (10, 1.5, 0)
    defGrad.assign(project(I + grad(u_sol), W_DFnStress))
    stretch_vec.append(defGrad(point)[0])
    print("stretch = " + str(defGrad(point)[0]))

    C_s = cauchy(mu1, mu2, mu4, g3, beta3, beta4, a1,
                 a2, I + grad(u_sol), p_sol, d_sol)
    C_stress.assign(project(C_s, W_DFnStress))
    cauchy_stress.append(C_stress(point)[0])
    print("stress = " + str(C_stress(point)[0]))

    # time increment
    t += float(dt)

# get analytical solution
stretch_vec = np.array(stretch_vec)
# stress_vec = np.array(stress_vec)
cauchy_stress = np.array(cauchy_stress)

# plot results
f = plt.figure(figsize=(12, 6))
plt.plot(stretch_vec, cauchy_stress, 'r-')
plt.xlabel("stretch")
plt.ylabel("cauchy stress")
plt.savefig('threeField.png')

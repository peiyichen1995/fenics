# system imports
from dolfin import *
from mshr import *
import math
import time

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross, matrix_cofactor
from utils import build_nullspace
from utils import XDMF2PVD

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

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


def pk1Stress(u, pressure, E, nu, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2):
    G = E / (2 * (1 + nu))
    c1 = G / 2.0

    I = Identity(V.mesh().geometry().dim())  # Identity tensor
    F = I + grad(u)          # Deformation gradient
    # C = F.T * F                # Right Cauchy-Green tensor
    C = variable(F.T * F)
    Ic = tr(C)               # Invariants of deformation tensors
    J = det(F)
    pk2 = 2 * c1 * I - pressure * inv(C)  # second PK stress
    NH = NeoHookean_imcompressible(mu1, mu2, mu3, mu4, beta3, beta4, a1, a2, F)
    PK2 = 2.0 * diff(NH, C)
    return pk2, (J - 1)


# directories
output_dir = "./output/"
mesh_dir = "./mesh/"

# mesh
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")
n = FacetNormal(mesh)

# mark boundaries
ds = Measure('ds', domain=mesh, subdomain_data=mf)

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
element_v = VectorElement("P", mesh.ufl_cell(), 1)
element_s = FiniteElement("P", mesh.ufl_cell(), 1)
mixed_element = MixedElement([element_v, element_s])
V = FunctionSpace(mesh, mixed_element)


# Define test and trial functions
dup = TrialFunction(V)
_u, _p = TestFunctions(V)

_u_p = Function(V)
u, p = split(_u_p)

# Create tensor function spaces for stress and stretch output
W_DFnStress = TensorFunctionSpace(mesh, "DG", degree=0)
defGrad = Function(W_DFnStress, name='F')
PK1_stress = Function(W_DFnStress, name='PK1')


# Displacement from previous iteration
P = Expression("1*t", t=0.0, degree=0)

# material parameters
E, nu = 1e3, 0.5
mu1 = 4.1543
mu2 = 2.5084
mu3 = 9.7227
mu4 = 19.285
beta3 = 3.6537
beta4 = 500.02
g3 = 9.7227

# # Total potential energy
pkstrs, hydpress = pk1Stress(
    u, p, E, nu, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
I = Identity(V.mesh().geometry().dim())
dgF = I + grad(u)
F1 = inner(dot(dgF, pkstrs), grad(_u)) * dx - dot(-P * n, _u) * ds(1)
F2 = hydpress * _p * dx
F = F1 + F2
J = derivative(F, _u_p, dup)

# Create nonlinear variational problem and solve
problem = NonlinearVariationalProblem(F, _u_p, bcs=[], J=J)
solver = NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'

# Time stepping parameters
dt = 0.1
t, T = 0.0, 10 * dt

# Save solution in VTK format
file_results = XDMFFile("./Results/TestUniaxialLoading/Uniaxial.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

solver.solve()


while t <= T:
    print('time: ', t)

    # Increase traction
    P.t = t

    # solve and save disp
    solver.solve()

    # Extract solution components
    u_plot, p_plot = _u_p.split()
    u_plot.rename("u", "displacement")
    p_plot.rename("p", "pressure")

    # get stretch at a point for plotting
    point = (0.5, 0.5, 0)
    DF = I + grad(u_plot)
    defGrad.assign(project(DF, W_DFnStress))

    # get stress at a point for plotting
    PK1_s, thydpress = pk1Stress(
        u_plot, p_plot, E, nu, mu1, mu2, mu3, mu4, beta3, beta4, a1, a2)
    PK1_stress.assign(project(PK1_s, W_DFnStress))

    # time increment
    t += float(dt)


file = File(output_dir + "displacements.pvd")
file << u_plot

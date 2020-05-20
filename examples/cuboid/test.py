from dolfin import *
import math
import ufl
import numpy

ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# directories
output_dir = "./output/"
mesh_dir = "./mesh/"

# Create mesh and define function space
mesh = Mesh(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# mark boundaries
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary_inner = CompiledSubDomain(
    "near(x[2], side) && on_boundary", side=1.0)
boundary_inner.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Mark boundary subdomians
bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
top = CompiledSubDomain("near(x[2], side) && on_boundary", side=1.0)

# Define Dirichlet boundary (x = 0 or x = 1)
t = Expression(('0', '0', '0.01'), element=V.ufl_element())
b = Expression(('0', '0', '0'), element=V.ufl_element())

bc_t = DirichletBC(V, t, top)
bc_b = DirichletBC(V, b, bottom)
bcs = [bc_t, bc_b]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v = TestFunction(V)             # Test function
u = Function(V)                 # Displacement from previous iteration

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T * F                   # Right Cauchy-Green tensor
A_1 = as_vector([cos(pi / 12), -sin(pi / 12), 0])
#A_2 = as_vector([cos(pi/12),sin(pi/12),0])
#M_1 = numpy.outer(A_1, A_1)
#M_2 = numpy.outer(A_2, A_2)
#J4_1 = numpy.trace(C*M_1)
#J4_2 = numpy.trace(C*M_2)
M_1 = outer(A_1, A_1)
#M_2 = outer(A_2, A_2)
J4_1 = tr(C * M_1)
#J4_2 = tr(C*M_2)

# Invariants of deformation tensors
# Ic = tr(C)
# J  = det(F)
I1 = tr(C)
I2 = 1 / 2 * (tr(C) * tr(C) - tr(C * C))
I3 = det(C)
#J4_1 = tr(C*M_1)
#J4_2 = tr(C*M_2)

# Elasticity parameters
# E, nu = 10.0, 0.3
# mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
eta1 = 141
eta2 = 160
eta3 = 3100
delta = 2 * eta1 + 4 * eta2 + 2 * eta3

e1 = 0.1
e2 = 1

k1 = 6.85
k2 = 754.01

# Stored strain energy density (compressible neo-Hookean model)
# psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
# compressible Mooney-Rivlin model
psi_MR = eta1 * I1 + eta2 * I2 + eta3 * I3 - delta * ln(sqrt(I3))
psi_P = e1 * (pow(I3, e2) + pow(I3, -e2) - 2)
#psi_ti_1 = k1*(exp(k2*conditional(gt(J4_1,1),J4_1-1,0))-1)/k2/2
psi_ti_1 = k1 / 2 / k2 * (exp(pow(conditional(gt(J4_1, 1), conditional(
    gt(J4_1, 2), J4_1 - 1, 2 * pow(J4_1 - 1, 2) - pow(J4_1 - 1, 3)), 0), 2) * k2) - 1)
psi_ti_2 = k1 * \
    (exp(k2 * conditional(gt(J4_1, 1), pow((J4_1 - 1), 2), 0)) - 1) / k2 / 2

psi = psi_MR + psi_P + psi_ti_1

# pressure
P = Constant(0.1)

# Total potential energy
Pi = psi * dx - dot(-P * n, u) * ds(1)

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
# solve(F == 0, u, bcs, J=J,
#      form_compiler_parameters=ffc_options)


# solve(F == 0, u, bcs,
#      solver_parameters={'linear_solver': 'gmres',
#                         'preconditioner': 'ilu'})

problem = NonlinearVariationalProblem(F, u, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 1000
prm['newton_solver']['relaxation_parameter'] = 1.0
solver.solve()


# solve(F == 0, u, bcs=bcs, J=J,
#      form_compiler_parameters={"optimize": True})

# Save solution in VTK format
file = File("displacement.pvd")
file << u

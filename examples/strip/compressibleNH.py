from dolfin import *
import math
import ufl
import numpy

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(10, 10), 10, 10)
V = VectorFunctionSpace(mesh, 'CG', 2)


# Mark boundary subdomians
bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side=0.0)
top = CompiledSubDomain("near(x[1], side) && on_boundary", side=10.0)

tDirBC = Expression(('-3*time_'), time_=1, degree=0)

bc_t = DirichletBC(V.sub(1), tDirBC, top)
bc_b = DirichletBC(V, Constant((0.0, 0.0)), bottom)
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
J = det(F)

# B_T1_12 = Expression(('0', '0'), element = V.ufl_element())
B = Expression(('0', '0'), element=V.ufl_element())
T = Constant((0.0, 0.0))  # Traction force on the boundary

Lambda = 10
mu = 6.3
c = Lambda / 4
d = Lambda / 4
Bulk = Lambda + 2 / 3 * mu

I1 = tr(C)

psi = c * (J * J - 1) - d * ln(J) - mu * ln(J) + 1 / 2 * mu * (I1 - 3)
# Total potential energy
Pi = psi * dx - dot(B, u) * dx - dot(T, u) * ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J_form = derivative(F, u, du)

# Solve variational problem
# solve(F == 0, u, bcs, J=J,
#      form_compiler_parameters=ffc_options)


# solve(F == 0, u, bcs,
#      solver_parameters={'linear_solver': 'gmres',
#                         'preconditioner': 'ilu'})

problem = NonlinearVariationalProblem(F, u, bcs, J_form)
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

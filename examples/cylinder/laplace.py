# system imports
from dolfin import *
from mshr import *

# my imports
from problems import CustomProblem
from solvers import CustomSolver

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# directories
output_dir = "./output/"
mesh_dir = "./mesh/"

# mesh
cylinder_o = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 2, 2)
cylinder_i = Cylinder(Point(0, 0, 0), Point(0, 0, 3), 1, 1)
geometry = cylinder_o - cylinder_i
mesh = generate_mesh(geometry, 10)

# function space
V = FunctionSpace(mesh, 'CG', 2)

# functions
v = TrialFunction(V)
w = TestFunction(V)
phi = Function(V)

# mark boundary subdomians
bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side=0.0)
top = CompiledSubDomain("near(x[2], side) && on_boundary", side=3.0)
inner = CompiledSubDomain(
    "near(sqrt(x[0]*x[0]+x[1]*x[1]), side, 0.1) && on_boundary", side=1.0)
outer = CompiledSubDomain(
    "near(sqrt(x[0]*x[0]+x[1]*x[1]), side, 0.1) && on_boundary", side=2.0)

# write mesh
mesh_file = File(mesh_dir + "mesh.xml")
mesh_file << mesh

# boundary conditions
bc_top = DirichletBC(V, Constant(1.0), top)
bc_bottom = DirichletBC(V, Constant(0.0), bottom)
bc_inner = DirichletBC(V, Constant(0.0), inner)
bc_outer = DirichletBC(V, Constant(1.0), outer)
bcs_1 = [bc_top, bc_bottom]
bcs_2 = [bc_inner, bc_outer]

# variational problem
Pi = 0.5 * dot(grad(phi), grad(phi)) * dx
dPi = derivative(Pi, phi, w)
J = derivative(dPi, phi, v)

# define variational problem for phi_1 and phi_2
problem_1 = CustomProblem(J, dPi, bcs_1)
problem_2 = CustomProblem(J, dPi, bcs_2)
solver = CustomSolver()

# create files for visualization and storage
phi1_pvd = File(output_dir + "phi1.pvd")
phi2_pvd = File(output_dir + "phi2.pvd")
phi1_h5 = HDF5File(MPI.comm_world, output_dir + "phi1.h5", "w")
phi2_h5 = HDF5File(MPI.comm_world, output_dir + "phi2.h5", "w")

# solve and write phi_1
solver.solve(problem_1, phi.vector())
phi1_pvd << phi
phi1_h5.write(phi, "phi1")

# solve and write phi_2
solver.solve(problem_2, phi.vector())
phi2_pvd << phi
phi2_h5.write(phi, "phi2")

# close files
phi1_h5.close()
phi1_h5.close()

# system imports
from dolfin import *
from mshr import *

# my imports
from problems import CustomProblem
from solvers import CustomSolver
from utils import MSH2XDMF, XDMF2PVD

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
MSH2XDMF(mesh_dir + "media_flatboundaries.msh",
         mesh_dir + "mesh.xdmf", mesh_dir + "mf.xdmf")
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

# function space
V = FunctionSpace(mesh, 'CG', 2)

# functions
v = TrialFunction(V)
w = TestFunction(V)
phi = Function(V)

# boundary conditions
bc_top = DirichletBC(V, Constant(1.0), mf, 7)
bc_bottom = DirichletBC(V, Constant(0.0), mf, 8)
bc_inner = DirichletBC(V, Constant(0.0), mf, 1)
bc_outer = DirichletBC(V, Constant(1.0), mf, 5)
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

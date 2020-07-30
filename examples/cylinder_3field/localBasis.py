# system imports
from dolfin import *
from mshr import *
import math

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross
from utils import build_nullspace

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
mesh = Mesh(mesh_dir + "mesh.xml")
n = FacetNormal(mesh)

# mark boundaries
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary_inner = CompiledSubDomain(
    "near(sqrt(x[0]*x[0]+x[1]*x[1]), side, 0.1) && on_boundary", side=1.0)
boundary_inner.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# function space
V = FunctionSpace(mesh, 'CG', 2)
diffV = VectorFunctionSpace(mesh, 'DG', 1)

# functions
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

e1Project = project(e1, diffV, solver_type="mumps")
file = XDMFFile(output_dir + "e1.xdmf")
file.write(e1Project, 0)

e2Project = project(e2, diffV, solver_type="mumps")
file = XDMFFile(output_dir + "e2.xdmf")
file.write(e2Project, 0)

e3Project = project(e3, diffV, solver_type="mumps")
file = XDMFFile(output_dir + "e3.xdmf")
file.write(e3Project, 0)

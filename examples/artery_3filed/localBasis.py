# system imports
from dolfin import *
from mshr import *
import math
import time

# my imports
from problems import ProblemWithNullSpace
from solvers import SolverWithNullSpace
from utils import my_cross
from utils import build_nullspace
from utils import XDMF2PVD

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
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

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

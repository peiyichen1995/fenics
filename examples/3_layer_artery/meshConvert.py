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

MSH2XDMF(mesh_dir + "media_flatboundaries.msh",
         mesh_dir + "mesh.xdmf", mesh_dir + "mf.xdmf")

mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

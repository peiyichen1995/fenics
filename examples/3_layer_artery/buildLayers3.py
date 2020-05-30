# system imports
from utils import build_nullspace
from utils import my_cross
from solvers import SolverWithNullSpace
from problems import ProblemWithNullSpace
import ufl
import math
from dolfin import *
from mshr import *
import numpy as np

# my imports
from problems import CustomProblem
from solvers import CustomSolver
from utils import MSH2XDMF, XDMF2PVD  # , shortest_dis

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
centerline_dir = "./centerline/"
surface_dir = "./surface/"

# mesh
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")


# function space
V = FunctionSpace(mesh, 'CG', 2)

phi2 = Function(V)


# read in laplace solutions
phi2_h5 = HDF5File(MPI.comm_world, output_dir + "phi2.h5", "r")
phi2_h5.read(phi2, "phi2")
phi2_h5.close()


def shortest_dis(phi, point, threshold1, threshold2):
    value = phi(point)

    return threshold1 < value < threshold2


# Define domain of three different layers
eps = DOLFIN_EPS
eps = 0.04
domain0 = AutoSubDomain(lambda x: shortest_dis(
    phi2, Point(x[0], x[1], x[2]), 0.0 - eps, 1 / 4 + eps))
domain1 = AutoSubDomain(lambda x: shortest_dis(
    phi2, Point(x[0], x[1], x[2]), 1 / 4 - eps, 2 / 3 + eps))
domain2 = AutoSubDomain(lambda x: shortest_dis(
    phi2, Point(x[0], x[1], x[2]), 2 / 3 - eps, 2 + eps))

# Have one function with tags of domains
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(2)

domain0.mark(domains, 1)
domain1.mark(domains, 2)
domain2.mark(domains, 3)
# mark domains
# dx = Measure('dx', domain=mesh, subdomain_data=domains)

# Save sub domains to VTK files
file = File(mesh_dir + "subdomains.pvd")
file << domains

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
from utils import MSH2XDMF, XDMF2PVD, shortest_dis

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

mesh
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

data = np.loadtxt(centerline_dir + "vascular_centerline.csv")
x, y, z = data[:, 4], data[:, 5], data[:, 6]

p0 = Point(x[0], y[0], z[0])
p1 = Point(x[1], y[1], z[1])

points = []

for i in range(len(x)):
    p = Point(x[i], y[i], z[i])
    points.append(p)

print(len(points))
print(p0.distance(points[0]))
print(shortest_dis(points, p1))


ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True


# function space
V = FunctionSpace(mesh, 'CG', 2)
VV = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)


# Define domain of three different layers
eps = DOLFIN_EPS
domain0 = AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2])) < 0.2 + eps)
domain1 = AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2])) > 0.2 - eps and shortest_dis(
        points, Point(x[0], x[1], x[2])) < 0.7 + eps)
domain2 = AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2])) > 0.7 - eps)

# Have one function with tags of domains
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)
domain0.mark(domains, 0)
domain1.mark(domains, 1)
domain2.mark(domains, 2)

# mark domains
dx = Measure('dx', domain=mesh, subdomain_data=domains)

# Save sub domains to VTK files
file = File(mesh_dir + "subdomains.pvd")
file << domains

# system imports
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

# mesh
# mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
#                     "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

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

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


ds = Measure('ds', domain=mesh, subdomain_data=mf)

# read center line
data = np.loadtxt(centerline_dir + "center.csv")
r, x, y, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

p0 = Point(x[0], y[0], z[0])
p1 = Point(x[1], y[1], z[1])

points = []
rs = []

for i in range(len(x)):
    p = Point(x[i], y[i], z[i])
    points.append(p)
    rs.append(r[i])

# read inner surface
data = np.loadtxt(surface_dir + "a0.csv")
x, y, z = data[:, 1], data[:, 2], data[:, 3]


surface = []


for i in range(len(x)):
    p = Point(x[i], y[i], z[i])
    surface.append(p)

# function space
V = FunctionSpace(mesh, 'CG', 2)
# VV = VectorFunctionSpace(mesh, 'CG', 2)
# VVV = TensorFunctionSpace(mesh, 'DG', 1)

phi2 = Function(V)


# read in laplace solutions
phi2_h5 = HDF5File(MPI.comm_world, output_dir + "phi2.h5", "r")
phi2_h5.read(phi2, "phi2")
phi2_h5.close()

print(type(phi2))


def shortest_dis(points, point, threshold1, threshold2):
    distance = point.distance(points[0])
    for i in range(len(points)):
        temp = point.distance(points[i])
        if(temp < distance):
            distance = temp

    # _, thick1 = tree_inner.compute_closest_entity(point)

    # _, thick2 = tree_outter.compute_closest_entity(point)

    return threshold1 < distance < threshold2 or distance == threshold1 or distance == threshold2


bmesh = BoundaryMesh(mesh, "exterior")


cell_f = MeshFunction('size_t', bmesh, bmesh.topology().dim() - 1)
cell_f.set_all(0)

AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2]), 0.0, 0.1)).mark(cell_f, 1)


bmesh_sub = SubMesh(bmesh, mf, 1)
# bmesh_sub = SubMesh(bmesh, cell_f, 1)
tree_inner = bmesh_sub.bounding_box_tree()


file = File(mesh_dir + "test.pvd")
file << bmesh_sub


# bmesh_sub = SubMesh(bmesh, mf, 7)
# bmesh_sub = SubMesh(bmesh, mf, 8)
# tree_outter = bmesh_sub.bounding_box_tree()


# def shortest_dis(point, threshold1, threshold2, tree_inner):
#
#     _, thick1 = tree_inner.compute_closest_entity(point)
#
#     # _, thick2 = tree_outter.compute_closest_entity(point)
#
#     return threshold1 < thick1 < threshold2


# Define domain of three different layers
eps = DOLFIN_EPS
eps = 0.05
domain0 = AutoSubDomain(lambda x: shortest_dis(
    surface, Point(x[0], x[1], x[2]), 0.0 - eps, 0.15 + eps))
domain1 = AutoSubDomain(lambda x: shortest_dis(
    surface, Point(x[0], x[1], x[2]), 0.15 - eps, 0.35 + eps))
domain2 = AutoSubDomain(lambda x: shortest_dis(
    surface, Point(x[0], x[1], x[2]), 0.35 - eps, 100))

# Have one function with tags of domains
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(3)

domain0.mark(domains, 1)
domain1.mark(domains, 2)
domain2.mark(domains, 3)
# mark domains
# dx = Measure('dx', domain=mesh, subdomain_data=domains)

# Save sub domains to VTK files
file = File(mesh_dir + "subdomains.pvd")
file << domains

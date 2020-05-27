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

# mesh
mesh, mf = XDMF2PVD(mesh_dir + "mesh.xdmf", mesh_dir +
                    "mf.xdmf", mesh_dir + "mesh.pvd", mesh_dir + "mf.pvd")

ds = Measure('ds', domain=mesh, subdomain_data=mf)

data = np.loadtxt(centerline_dir + "centerline.csv")
r, x, y, z = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

p0 = Point(x[0], y[0], z[0])
p1 = Point(x[1], y[1], z[1])

points = []
rs = []

for i in range(len(x)):
    p = Point(x[i], y[i], z[i])
    points.append(p)
    rs.append(r[i])

print(len(points))
print(p0.distance(points[0]))
# print(shortest_dis(rs, points, p0, 0, 0))


ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True


# function space
V = FunctionSpace(mesh, 'CG', 2)
VV = VectorFunctionSpace(mesh, 'CG', 2)
VVV = TensorFunctionSpace(mesh, 'DG', 1)

bmesh = BoundaryMesh(mesh, 'exterior')
# Suppose now bottom boundary is not of interest
# cell_f = MeshFunction('size_t', bmesh, bmesh.topology().dim() - 1)
# CompiledSubDomain('near(x[1], 0)').mark(cell_f, 1)
# print(type(cell_f))
# print(type(mf))
bmesh_sub = SubMesh(bmesh, mf, 1)
tree = bmesh_sub.bounding_box_tree()
_, d = tree.compute_closest_entity(p0)
print("choupanghu")
print(d)


def shortest_dis(points, point, threshold1, threshold2, mesh, mf):
    bmesh = BoundaryMesh(mesh, 'exterior')
    cell_f = MeshFunction('size_t', bmesh, bmesh.topology().dim() - 1)
    bmesh_sub = SubMesh(bmesh, mf, 1)
    tree = bmesh_sub.bounding_box_tree()

    distance = point.distance(points[0])
    id = 0
    _, thick = tree.compute_closest_entity(point)

    for i in range(len(points)):
        temp = point.distance(points[i])
        if(temp < distance):
            distance = temp

    return threshold1 < thick / distance < threshold2


# Define domain of three different layers
eps = DOLFIN_EPS
eps = 0.05
domain0 = AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2]), 0.0 - eps, 0.2 + eps, mesh, mf))
domain1 = AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2]), 0.2 - eps, 0.4 + eps, mesh, mf))
domain2 = AutoSubDomain(lambda x: shortest_dis(
    points, Point(x[0], x[1], x[2]), 0.4 - eps, 100, mesh, mf))

# Have one function with tags of domains
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)
domain0.mark(domains, 1)
domain1.mark(domains, 2)
domain2.mark(domains, 3)

# mark domains
dx = Measure('dx', domain=mesh, subdomain_data=domains)

# Save sub domains to VTK files
file = File(mesh_dir + "subdomains.pvd")
file << domains

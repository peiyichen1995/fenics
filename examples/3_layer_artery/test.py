from dolfin import *

mesh = UnitCubeMesh(4, 4, 4)
bmesh = BoundaryMesh(mesh, "exterior")

bbtree = BoundingBoxTree()
bbtree.build(bmesh)

vertex_distance_to_boundary = MeshFunction("double", mesh, 0)

for v_idx in range(mesh.num_vertices()):
    v = Vertex(mesh, v_idx)
    _, distance = bbtree.compute_closest_entity(v.point())
    print(distance)
    vertex_distance_to_boundary[v_idx] = distance

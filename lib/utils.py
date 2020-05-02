from dolfin import *
import meshio


def build_nullspace(V):
    x = Function(V).vector()
    nullspace_basis = [x.copy() for i in range(6)]
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0)
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0)
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2)
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0)
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2)
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1)
    for x in nullspace_basis:
        x.apply("insert")
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()
    return basis


def my_cross(a, b):
    return as_vector((a[1] * b[2] - a[2] * b[1],  a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def MSH2XDMF(MSH_name, XDMF_mesh_name, XDMF_mesh_face_name):
    msh = meshio.read(MSH_name)
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "tetra":
            tetra_cells = cell.data

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "tetra":
            tetra_data = msh.cell_data_dict["gmsh:physical"][key]

    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
    triangle_mesh = meshio.Mesh(points=msh.points,
                                cells=[("triangle", triangle_cells)],
                                cell_data={"face_id": [triangle_data]})

    meshio.write(XDMF_mesh_name, tetra_mesh)
    meshio.write(XDMF_mesh_face_name, triangle_mesh)


def XDMF2PVD(XDMF_mesh_name, XDMF_mesh_face_name, PVD_mesh_name, PVD_mesh_face_name):
    mesh = Mesh()
    with XDMFFile(XDMF_mesh_name) as infile:
        infile.read(mesh)
    File(PVD_mesh_name).write(mesh)

    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(XDMF_mesh_face_name) as infile:
        infile.read(mvc, "face_id")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    File(PVD_mesh_face_name).write(mf)

    return mesh, mf

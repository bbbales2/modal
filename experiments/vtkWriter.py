#%%

outfile = open("output.vtk", "w")

def toString(array):
    if len(array.shape) != 3:
        raise Exception("array must be 3d to use this function")

    nx = array.shape[0]
    ny = array.shape[1]
    nz = array.shape[2]

    string = """# vtk DataFile Version 3.0
        vtk output
        ASCII
        DATASET STRUCTURED_GRID
        DIMENSIONS {0} {1} {2}
        POINTS {3} float
        """.format(nx, ny, nz, nx * ny * nz)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                string += "{0} {1} {2}\n".format(i / float(nx), j / float(ny), k / float(nz))

    string += """POINT_DATA {0}
        SCALARS mydataset float
        LOOKUP_TABLE default
        """.format(nx * ny * nz)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                string += "{0}\n".format(array[i, j, k])

    return string

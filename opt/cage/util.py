import numpy as np
from scipy.sparse import coo_matrix


def laplacian_calculation(mesh, equal_weight=True):
    """
    Calculate a sparse matrix for laplacian operations.
    Parameters
    -------------
    mesh : trimesh.Trimesh
      Input geometry
    equal_weight : bool
      If True, all neighbors will be considered equally
      If False, all neightbors will be weighted by inverse distance
    Returns
    ----------
    laplacian : scipy.sparse.coo.coo_matrix
      Laplacian operator
    """
    # get the vertex neighbors from the cache
    neighbors = mesh.vertex_neighbors
    # avoid hitting crc checks in loops
    vertices = mesh.vertices.view(np.ndarray)

    # stack neighbors to 1D arrays
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])

    if equal_weight:
        # equal weights for each neighbor
        data = np.concatenate([[1.0 / len(n)] * len(n)
                               for n in neighbors])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = np.ones(3)
        # the distance from verticesex to neighbors
        norms = [1.0 / np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                 for i, n in enumerate(neighbors)]
        # normalize group and stack into single array
        data = np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)

    return matrix


def smoothing(mesh, num_iter=10, lbd=0.5, use_improved=True):

    # get the vertices
    vertices = mesh.vertices.copy()
    
    # iterative smoothing
    for _ in range(num_iter):

        laplacian_operator = laplacian_calculation(mesh, equal_weight=False)
        dot = laplacian_operator.dot(vertices) - vertices  # -H\mathbf{n}

        if use_improved:
            # implementation of improved smoothing algorithm from paper http://www.cad.zju.edu.cn/home/hwlin/pdf_files/Automatic-generation-of-coarse-bounding-cages-from-dense-meshes.pdf
            delta_d = mesh.vertex_normals
            inner_product = np.sum(dot*delta_d, axis=1, keepdims=True)
            is_outer = (inner_product >= 0).astype(np.float32)
            H_abs = np.linalg.norm(dot, axis=1, keepdims=True)
            outer_vec = is_outer * H_abs * delta_d
            inner_vec = (1-is_outer) * (dot - inner_product*delta_d)

            vertices += lbd * (outer_vec + inner_vec)

        else:
            # normal laplacian smoothing
            vertices += lbd * dot

        # update mesh
        mesh.vertices = vertices
    
    return mesh
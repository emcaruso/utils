import torch
import trimesh
import numpy as np

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)


    # print(len(mesh_.visual.uv))
    # print(len(mesh_.vertices))
    # print(len(mesh_.faces))
    # exit(1)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    uv = np.array(mesh_.visual.uv, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)

    return Mesh(vertices=vertices, indices=indices, uv=uv, device=device)

def write_mesh(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None
    mesh_ = trimesh.Trimesh(vertices=vertices, faces=indices, process=False)
    mesh_.export(path)

def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face 
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges

def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    assert counts.max() == 2

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])               
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0) # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei] 
        face_correspondences_indices[ei_unique] += 1

    return face_correspondences[counts.cpu() == 2].to(device=indices.device)
    # return face_correspondences[counts == 2].to(device=indices.device)

def compute_laplacian_uniform(mesh):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    verts_packed = mesh.vertices # (sum(V_n), 3)
    edges_packed = mesh.edges    # (sum(E_n), 2)
    V = mesh.vertices.shape[0]

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=mesh.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

class Mesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, uv=None, device='cpu', units='meters'):
        self.device = device
        self.units = units

        self.vertices = vertices.to(device, dtype=torch.float32) if torch.is_tensor(vertices) else torch.tensor(vertices, dtype=torch.float32, device=device)
        self.indices = indices.to(device, dtype=torch.int64) if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.int64, device=device) if indices is not None else None
        self.uv = uv.to(device, dtype=torch.float32) if torch.is_tensor(uv) else torch.tensor(uv, dtype=torch.float32, device=device) if uv is not None else None

        if self.indices is not None:
            self.compute_normals()

        self._edges = None
        self._connected_faces = None
        self._laplacian = None

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.indices = self.indices.to(device)

        if self.indices is not None:
            self.compute_normals()

        self._edges = None
        self._connected_faces = None
        self._laplacian = None

        self.compute_connectivity()

        self.device = device
        
        return self

    def detach(self):
        mesh = Mesh(self.vertices.detach(), self.indices.detach(), device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """

        assert len(vertices) == len(self.vertices)

        mesh_new = Mesh(vertices, self.indices, self.uv, self.device)
        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        return mesh_new

    @property
    def edges(self):
        if self._edges is None:
            self._edges = find_edges(self.indices)
        return self._edges

    @property
    def connected_faces(self):
        if self._connected_faces is None:
            self._connected_faces = find_connected_faces(self.indices)
        return self._connected_faces

    @property
    def laplacian(self):
        if self._laplacian is None:
            self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    def compute_connectivity(self):
        self._edges = self.edges
        self._connected_faces = self.connected_faces
        self._laplacian = self.laplacian

    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices][:, 0, :]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals)
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 

    def get_transformed_vertices(self, pose):
        return torch.matmul(self.vertices, pose.rotation().t())+pose.location()

    def transform_vertices(self, pose):
        self.vertices = self.get_transformed_vertices(pose)

    def uniform_scale(self, s, units:str = "scaled"):
        self.vertices *= s
        self.units = units


class AABB:
    def __init__(self, points):
        """ Construct the axis-aligned bounding box from a set of points.

        Args:
            points (tensor): Set of points (N x 3).
        """
        self.min_p, self.max_p = np.amin(points, axis=0), np.amax(points, axis=0)

    @classmethod
    def load(cls, path):
        points = np.loadtxt(path)
        return cls(points.astype(np.float32))

    def save(self, path):
        np.savetxt(path, np.array(self.minmax))

    @property
    def minmax(self):
        return [self.min_p, self.max_p]

    @property
    def center(self):
        return 0.5 * (self.max_p + self.min_p)

    @property
    def longest_extent(self):
        return np.amax(self.max_p - self.min_p)

    @property
    def corners(self):
        return np.array([
            [self.min_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.min_p[1], self.min_p[2]],
            [self.max_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.max_p[1], self.min_p[2]],
            [self.min_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.min_p[1], self.max_p[2]],
            [self.max_p[0], self.max_p[1], self.max_p[2]],
            [self.min_p[0], self.max_p[1], self.max_p[2]]
        ])

import copy as cp
import os

try:
    from .mesh import Mesh, AABB, read_mesh
except:
    from mesh import Mesh, AABB, read_mesh


class Object:
    def __init__(
        self, mesh=None, pose=None, material=None, textures=None, device="cpu"
    ):
        self.device = device

        if isinstance(mesh, str) and os.path.exists(mesh):
            self.mesh = read_mesh(mesh, device=device)

        elif isinstance(mesh, Mesh):
            self.mesh = mesh

        elif mesh is not None:
            raise ValueError("mesh is not path neither Mesh object")

        self.pose = pose
        self.textures = textures
        self.material = material
        self.to(device)

    def type(self, dtype):
        self.mesh.type(dtype)
        self.pose.type(dtype)
        self.material.type(dtype)
        return self

    def clone(self, same_pose=False):

        if same_pose:
            new_pose = self.pose
        else:
            new_pose = cp.deepcopy(self.pose)

        new_obj = Object(
            mesh=self.mesh,
            pose=new_pose,
            material=self.material,
            textures=self.textures,
            device=self.device,
        )
        return new_obj

    def to(self, device):

        self.device = device
        if self.mesh is not None and self.mesh.device != device:
            self.mesh = self.mesh.to(device)
        if self.pose is not None and self.pose.device != device:
            self.pose = self.pose.to(device)
        if self.material is not None and self.material.device != device:
            self.material = self.material.to(device)
        return self

    def set(self, key, value):
        assert hasattr(self, key)
        setattr(self, key, value)

    def get_vertices_from_pose(self):

        l = self.pose.location().to(self.device)
        R = self.pose.rotation().to(self.device)
        v_mesh = self.mesh.vertices.to(self.device)
        v = (v_mesh @ R.t()) + l
        return v

        # l = self.pose.location().to(self.device)
        # R = self.pose.rotation().to(self.device)
        # s = self.pose.scale.to(self.device)
        # v = ( ( (self.mesh.vertices * s)@R.t() ) + l)
        # return v

    def get_aabb(self, consider_pose=True):

        if not consider_pose:
            v = self.vertices.cpu().numpy()
        else:
            v = get_vertices_from_pose().cpu().numpy()

        aabb = AABB(v)
        return aabb

    def get_texture_mask(self, res):
        mesh = self.mesh

        # get mask texture of the uv map from mesh
        texture = mesh.get_texture_mask(res)
        return texture


try:
    from .mesh import Mesh, AABB
except:
    from mesh import Mesh, AABB

class Object():
    def __init__(self, mesh=None, pose=None, material=None, textures=None, device='cpu'):
        self.device = device
        self.mesh = mesh
        self.pose = pose
        self.textures = textures
        self.material = material
        self.to(device)

    def to(self, device):
        if self.mesh is not None and self.mesh.device!=device:
            self.mesh = self.mesh.to(device)
        if self.pose is not None and self.pose.device!=device:
            self.pose = self.pose.to(device)
        if self.material is not None and self.material.device!=device:
            self.material = self.material.to(device)
        return self

    def set(self, key, value):
        assert(hasattr(self,key))
        setattr(self,key,value)


    def get_vertices_from_pose(self):

        l = self.pose.location().to(self.device)
        R = self.pose.rotation().to(self.device)
        v_mesh = self.mesh.vertices.to(self.device)
        v = (v_mesh@R.t()) + l
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

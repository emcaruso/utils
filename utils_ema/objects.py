
try:
    from .mesh import *
except:
    from mesh import *

class Object():
    def __init__(self, mesh=None, pose=None, material=None, device='cpu'):
        self.device = device
        self.mesh = mesh
        self.pose = pose
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
        setattr(self,key,value.to(self.device))

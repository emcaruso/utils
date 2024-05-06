import torch
import copy as cp

class PointLight():
    def __init__(self, name="unk_point_light", position=torch.tensor([0,0,0]), intensity=torch.tensor([0,0,0]), device='cpu', dtype=torch.float32):
        self.name = name
        self.position = position
        self.intensity = intensity
        self.device = device
        self.dtype = dtype

    def clone(self, same_position = False, same_intensity = False, name = None ):

        if same_position: new_position = self.position
        else: new_position = cp.deepcopy(self.position) 

        if same_intensity: new_intensity = self.intensity
        else: new_intensity = cp.deepcopy(self.intensity) 

        if name is None: name = self.name+"_copy"

        new_cam = PointLight(name=name, position=new_position, intensity=new_intensity, device = self.device, dtype=self.typ)
        return new_cam

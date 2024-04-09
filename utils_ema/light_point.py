import torch

class PointLight():
    def __init__(self, name="unk_point_light", position=torch.tensor([0,0,0]), intensity=torch.tensor([0,0,0])):
        self.name = name
        self.position = position
        self.intensity = intensity


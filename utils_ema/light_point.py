import torch
import copy as cp


class PointLight:
    def __init__(
        self,
        name="unk_point_light",
        id=None,
        position=torch.tensor([0, 0, 0]),
        color=torch.tensor([0, 0, 0]),
        intensity=torch.tensor([0, 0, 0]),
        channel=None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.name = name
        self.id = id
        self.position = position
        self.channel = channel
        self.intensity = intensity
        self.color = color
        self.device = device
        self.dtype = dtype

    def clone(
        self,
        same_position=False,
        same_intensity=False,
        same_color=False,
        id=None,
        name=None,
    ):

        if same_position:
            new_position = self.position
        else:
            new_position = cp.deepcopy(self.position)

        if same_intensity:
            new_intensity = self.intensity
        else:
            new_intensity = cp.deepcopy(self.intensity)

        if same_color:
            new_color = self.color
        else:
            new_color = cp.deepcopy(self.color)

        if name is None:
            name = self.name + "_copy"

        if id is None:
            id = self.id

        new_light = PointLight(
            name=name,
            position=new_position,
            intensity=new_intensity,
            color=new_color,
            id=self.id,
            channel=self.channel,
            device=self.device,
            dtype=self.dtype,
        )
        return new_light

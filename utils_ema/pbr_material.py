import torch


class PBR_Material():
    def __init__(self, roughness, metallic=None, diffuse_albedo=None, specular_albedo=None, normal=None, dtype = torch.float32, requires_grad = False, device='cpu'):
        self.type = dtype
        self.set(roughness, metallic, diffuse_albedo, specular_albedo, normal, requires_grad, device=device)

    def to(self, device):
        if self.roughness is not None and self.roughness.device!=device:
            self.roughness = self.roughness.to(device)
        if self.metallic is not None and self.metallic.device!=device:
            self.metallic = self.metallic.to(device)
        if self.diffuse_albedo is not None and self.diffuse_albedo.device!=device:
            self.diffuse_albedo = self.diffuse_albedo.to(device)
        if self.specular_albedo is not None and self.specular_albedo.device!=device:
            self.specular_albedo = self.specular_albedo.to(device)
        if self.normal is not None and self.normal.device!=device:
            self.normal = self.normal.to(device)
        return self

    def set(self, roughness, metallic, diffuse_albedo, specular_albedo, normal, requires_grad=False, device='cpu'):
        self.roughness = roughness
        self.metallic = metallic
        self.diffuse_albedo = diffuse_albedo
        self.specular_albedo = specular_albedo
        self.normal = normal
        if isinstance(roughness,float): self.roughness=torch.tensor([roughness], dtype=self.type)
        if isinstance(metallic,float): self.metallic=torch.tensor([metallic], dtype=self.type)
        if isinstance(diffuse_albedo,float): self.diffuse_albedo=torch.tensor([diffuse_albedo], dtype=self.type)
        if isinstance(specular_albedo,float): self.specular_albedo=torch.tensor([specular_albedo], dtype=self.type)
        if isinstance(normal,float): self.normal=torch.tensor([normal], dtype=self.type)
        if self.roughness is not None: 
            self.roughness = self.roughness.to(device)
            self.roughness.requires_grad = requires_grad
        if self.metallic is not None: 
            self.metallic = self.metallic.to(device)
            self.metallic.requires_grad = requires_grad
        if self.diffuse_albedo is not None: 
            self.diffuse_albedo = self.diffuse_albedo.to(device)
            self.diffuse_albedo.requires_grad = requires_grad
        if self.specular_albedo is not None: 
            self.specular_albedo = self.specular_albedo.to(device)
            self.specular_albedo.requires_grad = requires_grad
        if self.normal is not None: 
            self.normal = self.normal.to(device)
            self.normal.requires_grad = requires_grad
        assert(self.roughness is None or torch.is_tensor(self.roughness))
        assert(self.metallic is None or torch.is_tensor(self.metallic))
        assert(self.diffuse_albedo is None or torch.is_tensor(self.diffuse_albedo))
        assert(self.specular_albedo is None or torch.is_tensor(self.specular_albedo))
        assert(self.normal is None or torch.is_tensor(self.normal))









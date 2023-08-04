import torch

class sphere():
    def __init__(self, frame, radius):
        self.radius = radius
        self.frame = frame

    def center(self):
        return self.frame.location()

    def point_on_sphere(self, point):
        r = self.radius.repeat # radius with same shape as point
        return torch.norm(point-sphere.center(), dim=-1) == r

    def ray2alpha_eps(self, pt, dir):
        assert(self.point_on_spere(pt)'''all true''')
        v =  (pt - self.center())
        azimuth_alpha = torch.atan2(v[...,0], v[...,1])
        elevation_alpha = torch.asin(v[...,2] / sphere_radius)
        alpha = torch.cat( (azimuth_alpha.unsqueeze(-1),elevation_alpha.unsqueeze(-1)), -1)
    
        dim_repeat = list(intersection_point.shape)[:-1]+[1]
        azimuth_eps = azimuth_alpha - torch.atan2(-dir[..., 0], -dir[..., 1])
        azimuth_eps = torch.atan2( torch.sin(azimuth_eps), torch.cos(azimuth_eps) )
        xy_projection = torch.sqrt(dir[...,0]**2 + dir[...,1]**2)
        elevation_eps = elevation_alpha + torch.asin(dir[...,2]/xy_projection)
        elevation_eps = torch.atan2( torch.sin(elevation_eps), torch.cos(elevation_eps) ) 
        eps = torch.cat( (azimuth_eps.unsqueeze(-1),elevation_eps.unsqueeze(-1)), -1)
        return alpha, eps

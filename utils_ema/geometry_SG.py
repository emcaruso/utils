import torch
import cv2
import math
import numpy as np
from utils_ema.plot import plotter
from utils_ema.image import Image
from utils_ema.torch_utils import set_seed
from tqdm import tqdm

class SphericalGaussians():
    def __init__(self, sgs_dict=None, n_sgs=None, gen_axis='uniform', lobe_ampl = 30., lobe_sharp = 100, device='cpu'):
        self.device = device
        assert( (sgs_dict is None) ^ (n_sgs is None) )
        if sgs_dict is not None:
            self.lobe_axis = sgs_dict["lobe_axis"]
            self.lobe_sharp = sgs_dict["lobe_sharp"]
            self.lobe_ampl = sgs_dict["lobe_ampl"]
            self.n_sgs = self.lobe_axis.shape[-2]
            assert(len(self.lobe_axis.shape)>=2) # shape n_gauss(batch), R3
        else:
            assert(n_sgs is not None)
            constr_axes = [ (-math.pi, math.pi), (-math.pi*0.4, math.pi*0.4) ] # azimuth, elevation 
            # generate random lobe axes
            if gen_axis == 'rand':
                self.n_sgs = n_sgs
                az = (torch.rand(n_sgs)*2-1)*math.pi
                el = (torch.rand(n_sgs)*2-1)*math.pi*0.4
                self.lobe_axis = torch.stack([torch.cos(az) * torch.cos(el), torch.sin(az) * torch.cos(el), torch.sin(el) ],dim=-1) 

            if gen_axis == 'uniform':
                self.n_sgs = int(n_sgs**2*0.5)
                az = torch.linspace(-math.pi, math.pi, int(n_sgs)+1)[:-1]
                el = torch.linspace(-math.pi*0.4, math.pi*0.4, int(n_sgs*0.5))
                AZ, EL = torch.meshgrid(az, el, indexing="ij")
                self.lobe_axis = torch.stack([torch.cos(AZ) * torch.cos(EL), torch.sin(AZ) * torch.cos(EL), torch.sin(EL) ],dim=-1) 
                self.lobe_axis = self.lobe_axis.view(-1,3)
                # self.lobe_axis = self.lobe_axis[..., :3] / (torch.norm(self.lobe_axis[..., :3], dim=-1, keepdim=True))

            # # lobe axis
            # self.lobe_axis = 
            # fixed ampl and sharp
            self.lobe_axis = - self.lobe_axis
            self.lobe_ampl = torch.ones((self.n_sgs, 3)) * lobe_ampl
            self.lobe_sharp = torch.ones((self.n_sgs, 1)) * lobe_sharp

        self.lobe_axis = self.lobe_axis.to(device)
        self.lobe_ampl = self.lobe_ampl.to(device)
        self.lobe_sharp = self.lobe_sharp.to(device)

    def __mul__(self, other):

        lin_comb = self.lobe_sharp * self.lobe_axis + other.lobe_sharp * other.lobe_axis

        new_sharp = torch.norm(lin_comb, dim=-1).unsqueeze(-1)
        new_axis = lin_comb/new_sharp
        new_ampl = self.lobe_ampl * other.lobe_ampl * torch.exp(new_sharp - self.lobe_sharp - other.lobe_sharp)

        # print(torch.max(self.lobe_sharp), "\n",torch.max(other.lobe_sharp),"\n",torch.max(new_sharp),"\nnew_sharp")
        # print(torch.min(self.lobe_sharp), "\n",torch.min(other.lobe_sharp),"\n",torch.min(new_sharp),"\nnew_sharp")

        sgs_dict = { "lobe_axis":new_axis, "lobe_sharp":new_sharp, "lobe_ampl":new_ampl }
        return SphericalGaussians(sgs_dict=sgs_dict, device=self.device)

    def hemisphere_int(self, n):

        cos_beta = torch.sum( self.lobe_axis*n, dim=-1, keepdim=True)
        lambda_val = self.lobe_sharp + 0.00001
        # orig impl; might be numerically unstable
        # t = torch.sqrt(lambda_val) * (1.6988 * lambda_val * lambda_val + 10.8438 * lambda_val) / (lambda_val * lambda_val + 6.2201 * lambda_val + 10.2415)

        inv_lambda_val = 1. / lambda_val
        t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                    1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

        # orig impl; might be numerically unstable
        # a = torch.exp(t)
        # b = torch.exp(t * cos_beta)
        # s = (a * b - 1.) / ((a - 1.) * (b + 1.))

        ### note: for numeric stability
        inv_a = torch.exp(-t)
        mask = (cos_beta >= 0).float()
        inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
        s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
        b = torch.exp(t * torch.clamp(cos_beta, max=0.))
        s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
        s = mask * s1 + (1. - mask) * s2

        A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
        A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

        return self.lobe_ampl * (A_b * (1. - s) + A_u * s)

    # def __mul__(self, other):
    #     # assume self.lobe_sharp << other.lobe_sharp
    #     ratio = self.lobe_sharp / other.lobe_sharp

    #     dot = torch.sum(self.lobe_axis * other.lobe_axis, dim=-1, keepdim=True)
    #     tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    #     tmp = torch.min(tmp, ratio + 1.)

    #     lambda3 = other.lobe_sharp * tmp
    #     lambda1_over_lambda3 = ratio / tmp
    #     lambda2_over_lambda3 = 1. / tmp
    #     diff = other.lobe_sharp * (tmp - ratio - 1.)

    #     new_axis = lambda1_over_lambda3 * self.lobe_axis + lambda2_over_lambda3 * other.lobe_axis
    #     new_sharp = lambda3
    #     new_ampl = self.lobe_ampl * other.lobe_ampl * torch.exp(diff)

    #     sgs_dict = { "lobe_axis":new_axis, "lobe_sharp":new_sharp, "lobe_ampl":new_ampl }
    #     return SphericalGaussians(sgs_dict=sgs_dict, device=self.device)


    def eval( self, x):
        assert(x.shape[-1]==1 or x.shape[-1]==3)
        assert(x.shape[-2]==1 or x.shape[-2]==self.n_sgs)

        x_norm = torch.nn.functional.normalize(x, dim=-1)
        lobe_axis_norm = torch.nn.functional.normalize(self.lobe_axis, dim=-1)

        dot = torch.clamp(torch.sum(x_norm * lobe_axis_norm, dim=-1), min=0).unsqueeze(-1)
        exp = torch.exp(self.lobe_sharp*(dot-1))
        res = self.lobe_ampl*exp

        return res

    def eval_and_integrate( self, x):
        res = self.eval( x)
        rgb =  torch.sum(res, dim=-2)
        rgb = rgb.permute(1,0,2)
        return rgb



    def show_envmap( self, resolution=256 ):

        # Generate the grid points
        az = torch.linspace(-math.pi, math.pi, resolution*2)
        el = torch.linspace(-math.pi/2, math.pi/2, resolution)
        AZ, EL = torch.meshgrid(az, el, indexing="ij")
        x = torch.stack([torch.cos(AZ) * torch.cos(EL), torch.sin(AZ) * torch.cos(EL), torch.sin(EL) ],dim=-1) 
        x=x.unsqueeze(-2)
        x = x.to(self.device)

        # plotter.plot_ray(torch.zeros_like(x),x)
        # plotter.show()

        # x = x.view(-1,3)

        # grid = torch.cat( (AZ.unsqueeze(-1), EL.unsqueeze(-1)), dim=-1 )
        rgb = self.eval_and_integrate(x)
        # print(rgb.shape)
        # print(rgb.dtype)
        img = Image(rgb)
        img.resize([800,400])
        img.show(wk=0)

        
    def get_gauss_mask_hemisphere( self, dirs):
        dirs_resh = dirs.view(dirs.shape[0], 1, 3)
        loax_resh = self.lobe_axis.view(1, self.lobe_axis.shape[0], 3)
        dot = torch.sum(dirs_resh * loax_resh, dim=2)
        mask = dot>0
        return mask

    def get_list_of_gauss_for_each_view(self , dirs):
        mask = self.get_gauss_mask_hemisphere(dirs)
        n = mask.shape[0]

        gauss_idxs = []
        for i in tqdm(range(n)):
            row_indices = (mask[i, :] == True).nonzero().squeeze()
            gauss_idxs.append(row_indices)
        return gauss_idxs

if __name__ == "__main__":
    set_seed(666)
    # sgs = SphericalGaussians(n_sgs=32)
    sgs = SphericalGaussians(n_sgs=32, gen_axis='rand')
    # sgs.eval(torch.Tensor([[0,0],[0,0],[0,0],[0,0]]))
    sgs.show_envmap()


    n = np.random.choice(range(len(sgs.lobe_axis)),16, replace=False)
    dirs = sgs.lobe_axis[n]
    # sgs.get_gauss_in_hemisphere(dirs)
    sgs.get_list_of_gauss_for_each_view(dirs)



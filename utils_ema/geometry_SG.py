import torch
import cv2
import math
import numpy as np
from utils_ema.plot import plotter
from utils_ema.image import Image
from utils_ema.torch_utils import set_seed
from tqdm import tqdm

class SphericalGaussians():
    def __init__(self, sgs_dict=None, n_sgs=None, gen_axis='uniform', lobe_ampl = 1, lobe_sharp = 1000, device='cpu'):
        self.device = device
        assert( (sgs_dict is None) ^ (n_sgs is None) )
        if sgs_dict is not None:
            self.lobe_axis = sgs_dict["lobe_axis"]
            self.lobe_ampl = sgs_dict["lobe_ampl"]
            self.lobe_sharp = sgs_dict["lobe_sharp"]
            self.n_sgs = self.lobe_axis.shape[0]
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
                self.lobe_axis = -self.lobe_axis.view(-1,3)
                # self.lobe_axis = self.lobe_axis[..., :3] / (torch.norm(self.lobe_axis[..., :3], dim=-1, keepdim=True))

            # # lobe axis
            # self.lobe_axis = 
            # fixed ampl and sharp
            self.lobe_ampl = torch.ones((self.n_sgs, 3)) * lobe_ampl
            self.lobe_sharp = torch.ones((self.n_sgs, 1)) * lobe_sharp

        self.lobe_axis = self.lobe_axis.to(device)
        self.lobe_ampl = self.lobe_ampl.to(device)
        self.lobe_sharp = self.lobe_sharp.to(device)

    def eval_batch(self, x, mask):
        assert(x.shape[-1]==3)
        assert(len(x.shape) == 3)
        assert(x.shape[0] == self.lobe_axis.shape[0])

        shape_ori = x.shape

        x_norm = torch.nn.functional.normalize(x, dim=-1)
        lobe_axis_norm = torch.nn.functional.normalize(self.lobe_axis, dim=-1)

        # print(x_norm.shape)
        # print(lobe_axis_norm.shape)
        # print(self.lobe_ampl.shape)

        dot = torch.sum(x_norm * lobe_axis_norm.unsqueeze(1), dim=-1, keepdim=True).squeeze(-1)    
        exp = torch.exp(self.lobe_sharp.squeeze()*(dot-1)).unsqueeze(-1)
        res = self.lobe_ampl.unsqueeze(0).unsqueeze(0)*exp

        # res_masked = res[mask>0]
        res_masked = res*mask.unsqueeze(-1)

        # res = torch.sum(res_masked, dim=1)

        return res


    def eval( self, x):
        '''
        x = [ batch (n points), 2 ]
        output = [ batch (n points), num channels ]

        '''
        assert(x.shape[-1]==3)
        if len(x.shape) == 3:

            shape_ori = x.shape
            x_flat = x.view(-1,3)
            n_x = x_flat.shape[0]

            x_norm = torch.nn.functional.normalize(x_flat, dim=-1)
            lobe_axis_norm = torch.nn.functional.normalize(self.lobe_axis, dim=-1)

            # print(x_norm.shape)
            # print(lobe_axis_norm.shape)
            # exit(1)

            dot = torch.sum(x_norm.unsqueeze(1) * lobe_axis_norm.unsqueeze(0), dim=-1, keepdim=True).squeeze(-1)    
            exp = torch.exp(self.lobe_sharp.squeeze()*(dot-1)).unsqueeze(-1)
            res = torch.sum(self.lobe_ampl.unsqueeze(0)*exp, dim=1)

            res = res.reshape(shape_ori)
            res = res.permute(1,0,2)

            return res

        elif len(x.shape) == 2:
            
            n_x = x.shape[0]

            x_norm = torch.nn.functional.normalize(x, dim=-1)
            lobe_axis_norm = torch.nn.functional.normalize(self.lobe_axis, dim=-1)

            print(x_norm.shape)
            print(lobe_axis_norm.shape)
            exit(1)


            dot = torch.sum(x_norm.unsqueeze(1) * lobe_axis_norm.unsqueeze(0), dim=-1, keepdim=True).squeeze(-1)
            exp = torch.exp(self.lobe_sharp.squeeze()*(dot-1)).unsqueeze(-1)
            res = torch.sum(self.lobe_ampl.unsqueeze(0)*exp, dim=1)

            return res



    def show_envmap( self, resolution=256 ):

        # Generate the grid points
        az = torch.linspace(-math.pi, math.pi, resolution*2)
        el = torch.linspace(-math.pi/2, math.pi/2, resolution)
        AZ, EL = torch.meshgrid(az, el, indexing="ij")
        x = torch.stack([torch.cos(AZ) * torch.cos(EL), torch.sin(AZ) * torch.cos(EL), torch.sin(EL) ],dim=-1) 
        # plotter.plot_ray(torch.zeros_like(x),x)
        # plotter.show()

        # x = x.view(-1,3)

        # grid = torch.cat( (AZ.unsqueeze(-1), EL.unsqueeze(-1)), dim=-1 )
        res = self.eval(x)
        # print(res.shape)
        # print(res.dtype)
        img = Image(res)
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



import cv2
import torch
import sys, os

from geometry_pose import *
from plot import *
from torch_utils import *


class Camera_cv():

    def __init__(self, K=torch.FloatTensor( [[30,0,18],[0,30,18],[0,0,1]]), frame=frame(), resolution=torch.LongTensor([700,700]), images=None, name="Unk Cam" ):
        self.name = name
        self.frame = frame
        self.images = images
        self.resolution = resolution
        self.K = K # in millimeters
        self.meter_pixel_ratio = self.size()*0.001/self.resolution
        self.millimeters_pixel_ratio = self.size()/self.resolution
    
    
    def cx(self): return self.K[...,0,2]
    def cy(self): return self.K[...,1,2]
    def fx(self): return self.K[...,0,0]
    def fy(self): return self.K[...,1,1]
    def lens(self): return torch.cat( (self.fx().unsqueeze(-1), self.fy().unsqueeze(-1)) , dim=-1)
    def size(self): return torch.cat( ( (self.cx()*2).unsqueeze(-1), (self.cy()*2).unsqueeze(-1) ) , dim=-1)
    def lens_squeezed(self): return (self.fx()+self.fy())/2
    def show_image(self,img_name="rgb", wk=0):
        cv2.imshow(img_name, self.images[img_name])
        cv2.waitKey(wk)
        # plt.imshow(self.images[img_name])
        # plt.show()

    def get_pixel_grid(self, n = None):
        if n is None:
            n=self.resolution
        offs = (self.resolution/n)/2
        x_range = torch.linspace(offs[0], self.resolution[0]-offs[0], n[0])  # 5 points from -1 to 1
        y_range = torch.linspace(offs[1], self.resolution[1]-offs[1], n[1])  # 5 points from -1 to 1
        X, Y = torch.meshgrid(x_range, y_range)
        grid = torch.cat( (X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1 )
        return grid

    def get_all_rays(self):
        grid = c.get_pixel_grid( )
        origin, dir = c.pix2ray( grid )
        return origin, dir

    def pix2dir( self, pix ):
        pix = pix*self.millimeters_pixel_ratio # pixels to millimeters
        ndc = (pix - self.size()/2) / self.lens()
        dir = torch.cat( (ndc, torch.ones( list(ndc.shape[:-1])+[1] ) ), -1 )
        dir_norm = torch.nn.functional.normalize( dir, dim=-1 )
        return torch.matmul(dir_norm, self.frame.rotation().T)

    def pix2ray( self, pix ):
        dir = self.pix2dir( pix )
        origin = self.frame.location()
        origin = repeat_tensor_to_match_shape( origin, dir.shape )
        return origin, dir

    


class Camera_on_sphere(Camera_cv):
    
    def __init__(self, az_el, az_el_idx, K, frame, resolution, images=None, name="Unk Cam on sphere" ):
        super().__init__(K=K, frame=frame, resolution=resolution, images=images, name=name)


if __name__=="__main__":
    # c = Camera_on_sphere()
    c = Camera_cv()
    c.frame.rotate_euler(eul(torch.FloatTensor([math.pi/4,math.pi/3,0])))
    c.frame.set_location(torch.FloatTensor([0.5,0.2,-0.1]))
    grid = c.get_pixel_grid( torch.LongTensor([10,10]) )
    origin, dir = c.pix2ray( grid )
    p = plotter()
    p.init_figure()
    p.plot_ray(origin, dir)
    p.plot_cam(c, 1)
    p.plot_frame(c.frame)
    p.show()

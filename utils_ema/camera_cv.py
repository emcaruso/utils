import cv2
import torch
import sys, os

try:
    from .geometry_pose import *
    from .plot import *
    from .torch_utils import *
    from .general import *
    from .images import *
except:
    from geometry_pose import *
    from plot import *
    from torch_utils import *
    from general import *
    from images import *

class Camera_opencv:
    """ Camera in OpenCV format.
        
    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """

    def __init__(self, K, R, t, device='cpu'):
        self.K = K.to(device) if torch.is_tensor(K) else torch.FloatTensor(K).to(device)
        self.R = R.to(device) if torch.is_tensor(R) else torch.FloatTensor(R).to(device)
        self.t = t.to(device) if torch.is_tensor(t) else torch.FloatTensor(t).to(device)
        self.device = device

    def to(self, device="cpu"):
        self.K = self.K.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.device = device
        return self

    @property
    def center(self):
        return -self.R.t() @ self.t

    @property
    def P(self):
        return self.K @ torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)


class Intrinsics():
    def __init__(self, K=torch.FloatTensor([[0.030,0,0.018],[0,0.030,0.018],[0,0,1]]), resolution=torch.LongTensor([700,700]), units:str='meters'):
        self.K = K
        self.units = units
        self.resolution = resolution
        self.pixel_unit_ratio = resolution[0]/self.size()[0]
        self.unit_pixel_ratio = self.size()[0]/resolution[0]

    def cx(self): return self.K[...,0,2]
    def cy(self): return self.K[...,1,2]
    def fx(self): return self.K[...,0,0]
    def fy(self): return self.K[...,1,1]
    def lens(self): return torch.cat( (self.fx().unsqueeze(-1), self.fy().unsqueeze(-1)) , dim=-1)
    def size(self): return torch.cat( ( (self.cx()*2).unsqueeze(-1), (self.cy()*2).unsqueeze(-1) ) , dim=-1)
    def lens_squeezed(self): return (self.fx()+self.fy())/2

class Camera_cv():

    def __init__(self, intrinsics:Intrinsics = Intrinsics(), frame:Frame = Frame(), image_paths=None, name="Unk Cam", load_images=False, device='cpu'):
        self.device=device
        self.name = name
        self.frame = frame
        self.images = {}
        self.image_paths = image_paths
        self.intr = intrinsics
        if self.intr.units != self.frame.units: raise ValueError("frame units ("+self.frame.units+") and intrinsics units ("+self.intr.units+") must be the same")
        if load_images: self.load_images()

    def get_camera_opencv(device=None):
        if device is None:
            device = self.device
        return Camera_opencv(self.K, self.frame.rotation, self.frame.location, device)

    def load_images(self):
        if self.images == {}:
            for image_name,image_path in self.image_paths.items():
                img = cv2.imread(image_path)
                if is_grayscale(img):
                    img=img[...,:1]
                img = torch.FloatTensor(img)
                self.images[image_name] = img / 255.0

    def free_images(self):
        del self.images
        self.images = {}
    
    def show_image(self,img_name="rgb", wk=0):
        image = self.images[img_name]
        if torch.is_tensor(image):
            image = image.numpy()
        cv2.imshow(img_name, image)
        cv2.waitKey(wk)

    def get_image(self, img_name="rgb"):
        image = self.images[img_name]
        if torch.is_tensor(image):
            image = image.numpy()
        return image

    def show_images(self, wk=0):
        for name in self.images.keys():
            self.show_image(name, wk)

    def get_pixel_grid(self, n = None, longtens=False):
        if n is None:
            n=self.intr.resolution
        offs = (self.intr.resolution/n)/2
        x_range = torch.linspace(offs[0], self.intr.resolution[0]-offs[0], n[0])  # 5 points from -1 to 1
        y_range = torch.linspace(offs[1], self.intr.resolution[1]-offs[1], n[1])  # 5 points from -1 to 1
        X, Y = torch.meshgrid(x_range, y_range, indexing="ij")
        grid = torch.cat( (X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1 )
        if longtens:
            return torch.trunc(grid).type(torch.LongTensor)
        return grid

    def sample_pixels( self, num_pixels, longtens=False ):
        pixels_idxs = torch.reshape(self.get_pixel_grid( ), (-1,len(self.intr.resolution)))
        perm = torch.randperm(pixels_idxs.shape[0])
        n = min(pixels_idxs.shape[0],num_pixels )
        sampl_image_idxs = pixels_idxs[perm][:n]
        return sampl_image_idxs

    def get_all_rays(self):
        grid = c.get_pixel_grid( )
        origin, dir = c.pix2ray( grid )
        return origin, dir

    def pix2dir( self, pix ):
        pix = pix*self.intr.unit_pixel_ratio # pixels to units
        ndc = (pix - self.intr.size()/2) / self.intr.lens()
        dir = torch.cat( (ndc, torch.ones( list(ndc.shape[:-1])+[1] ) ), -1 )
        dir_norm = torch.nn.functional.normalize( dir, dim=-1 )
        return torch.matmul(dir_norm, self.frame.rotation().T)

    def pix2ray( self, pix ):
        dir = self.pix2dir( pix )
        origin = self.frame.location()
        origin = repeat_tensor_to_match_shape( origin, dir.shape )
        return origin, dir

    def collect_pixs_from_img( self, image, pix ):
        assert(pix.dtype==torch.int32)
        return image[pix[:,0], pix[:,1],...]

class Camera_on_sphere(Camera_cv):
    
    def __init__(self, az_el, az_el_idx, K=torch.FloatTensor( [[30,0,18],[0,30,18],[0,0,1]]), frame=None, resolution=torch.LongTensor([700,700]), images=None, name="Unk Cam on sphere" ):
        super().__init__(K=K, frame=frame, resolution=resolution, images=images, name=name)
        self.alpha = az_el

    def pix2eps( self, pix ):
        assert( pix.dtype==torch.float32)
        eps = -torch.arctan2(((pix-(self.intr.resolution/2))*self.millimeters_pixel_ratio), self.lens())
        return eps

    def get_sample_from_pixs( self, pixs ):
        eps = self.pix2eps( pixs )
        alpha = repeat_tensor_to_match_shape(self.alpha, eps.shape)
        sample = { 'eps':eps, 'alpha':alpha }
        return sample

    def sample_pixels_from_err_img( self, num_pixels, show=True ):
        pixs = sample_from_image_pdf( self.images["err"], num_pixels )
        
        # show sampled pixels
        if show:
            show_pixs(pixs, self.images["err"].shape,wk=1)

        pixs = torch.FloatTensor( pixs+0.5 )
        return pixs



    def render_ddf( self, ddf, device, wk=0, update_err=False, path_err="" , prt=False):

        # for k,v in self.images.items():
        #     if k=="err":
        #         continue
        #     show_image( k+":_gt", v.numpy(), wk )

        with torch.no_grad():
            grid = self.get_pixel_grid()
            sample = self.get_sample_from_pixs( grid )
            sample = dict_to_device(sample, device)
            output = ddf.forward( sample ).detach().cpu()
            count = 0
            errs = []
            for k,v in self.images.items():
                if k=="err":
                    continue
                o = output[...,count:count+v.shape[-1]]
                count += v.shape[-1]
                errs.append(torch.abs(o-v))
                show_image( k, o.numpy(), wk )

            if update_err:
                err = torch.cat( errs, dim=-1 )
                err = torch.norm(err, dim=-1)
                if prt:
                    print("err: "+str(torch.sum(err).item()))
                cv2.imwrite(path_err+"/"+self.name+".png", err.numpy()*255)
                show_image("err", err, wk)

if __name__=="__main__":
    # c = Camera_on_sphere()
    c = Camera_cv()
    c.sample_pixels( 10 )
    c.frame.rotate_euler(eul(torch.FloatTensor([math.pi/4,math.pi/3,0])))
    c.frame.set_location(torch.FloatTensor([0.5,0.2,-0.1]))
    pixs = c.get_pixel_grid( torch.LongTensor([10,10]) )
    # pixs = c.sample_pixels( 10 )
    origin, dir = c.pix2ray( pixs )
    p = plotter
    p.plot_ray(origin, dir)
    p.plot_cam(c, 1)
    p.plot_frame(c.frame)
    p.show()








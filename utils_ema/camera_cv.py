import cv2
import torch
import sys, os
import copy as cp

try:
    from .geometry_pose import *
    from .plot import *
    from .torch_utils import *
    from .general import *
    from .image import *
    try: from .diff_renderer import *
    except: pass
except:
    from geometry_pose import *
    from plot import *
    from torch_utils import *
    from general import *
    from image import *
    try: from diff_renderer import *
    except: pass

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
    def __init__(self, K=torch.FloatTensor([[0.030,0,0.018],[0,0.030,0.018],[0,0,1]]), D=None, 
                    resolution=torch.LongTensor([700,700]), sensor_size=torch.FloatTensor([0.0036,0.0036]), 
                    units:str='meters', typ=torch.float32):
        self.units = units
        self.typ = typ
        self.sensor_size = sensor_size
        self.resolution = resolution
        self.D = D
        self.K = K
        self.K_pix = self.get_K_in_pixels()
        self.K_und, self.K_pix_und, self.roi_und = self.get_K_und()
        self.dtype(typ)

    def cx(self): return self.K[...,0,2]
    def cy(self): return self.K[...,1,2]
    def fx(self): return self.K[...,0,0]
    def fy(self): return self.K[...,1,1]
    def pixel_unit_ratio(self): return self.resolution[0]/self.sensor_size[0]
    def unit_pixel_ratio(self): return self.sensor_size[0]/self.resolution[0]
    def lens(self): return torch.cat( (self.fx().unsqueeze(-1), self.fy().unsqueeze(-1)) , dim=-1)
    def lens_squeezed(self): return (self.fx()+self.fy())/2
    def to(self, device):
        self.K = self.K.to(device)
        self.resolution = self.resolution.to(device)
        return self
    def dtype(self, dtype):
        self.K = self.K.to(dtype)
        return self
    def uniform_scale(self, s:float, units:str="scaled" ):
        self.K[...,0,0]*=s
        self.K[...,1,1]*=s
        self.K[...,:2,-1]*=s
        self.units = units
    def get_K_in_pixels(self):
        K = self.K.clone()
        r = self.pixel_unit_ratio()
        K[...,0,0]*=r
        K[...,1,1]*=r
        K[...,:2,-1]*=r
        return K
    def get_K_und(self, alpha=0, central_pp=True):
        K_pix_und = None
        K_und = None
        roi_und = None
        if self.D is not None:
            w = int(self.resolution[0])
            h = int(self.resolution[1])
            K_pix_und, roi_und = cv2.getOptimalNewCameraMatrix(self.K_pix.numpy(),self.D.numpy(),(w,h),alpha,(w,h), central_pp)
            K_pix_und = torch.from_numpy(K_pix_und) 
            K_und = K_pix_und * self.unit_pixel_ratio()
        return K_und, K_pix_und, roi_und
    def undistort_image(self, img):
        map1, map2 = cv2.initUndistortRectifyMap(self.K_pix.numpy(), self.D.numpy(), None, self.K_pix_und.numpy(), (int(self.resolution[0]), int(self.resolution[1])), cv2.CV_32FC1)
        undistorted = cv2.remap(img.numpy(), map1, map2, cv2.INTER_LINEAR)
        # x,y,w,h = self.roi_und
        # undistorted = undistorted[y:y+h, x:x+w]
        return torch.from_numpy(undistorted)




class Camera_cv():

    def __init__(self, intrinsics = Intrinsics(), pose = Pose(), image_paths=None, frame=None, name="Unk Cam", load_images=None, device='cpu', dtype=torch.float32):
        self.device = device
        self.name = name
        self.frame = frame
        self.pose = pose.to(device).dtype(dtype)
        self.intr = intrinsics.to(device).dtype(dtype)
        self.images = {}
        self.image_paths = image_paths
        self.typ = dtype
        if self.intr.units != self.pose.units: raise ValueError("frame units ("+self.pose.units+") and intrinsics units ("+self.intr.units+") must be the same")
        if load_images is not None: self.load_images( load_images, device)

    def clone(self, same_intr = False, same_pose = False, image_paths = None, name = None ):

        if same_intr: new_intr = self.intr
        else: new_intr = cp.deepcopy(self.intr) 

        if same_pose: new_pose = self.pose
        else: new_pose = cp.deepcopy(self.pose) 

        if image_paths is None: image_paths = self.image_paths
        if name is None: name = self.name+"_copy"

        new_cam = Camera_cv(new_intr, new_pose, image_paths, self.frame, name, device = self.device, dtype=self.typ)
        return new_cam


    def to(self, device):
        self.pose = self.pose.to(device)
        self.intr = self.intr.to(device)

    def get_camera_opencv(self, device=None):
        if device is None:
            device = self.device
        K = self.intr.get_K_in_pixels()
        self.pose.invert()
        R = self.pose.rotation()
        t = self.pose.location()
        self.pose.invert()
        return Camera_opencv( K, R, t, device)

    def load_images(self, images=None, device='cpu'):

        if images is None:
            images = self.image_paths.keys()

        for image_name in images:
            image_path = self.image_paths[image_name]
            img = cv2.imread(image_path)
            if is_grayscale(img):
                img=img[...,:1]
            img = torch.FloatTensor(img)
            img = img / 255.0
            img = img.to(device)
            self.images[image_name] = img

    def free_images(self):
        del self.images
        self.images = {}
    
    def show_image(self,img_name="rgb", wk=0):
        image = self.get_image(img_name)
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        cv2.imshow(img_name, image)
        cv2.waitKey(wk)

    def get_image(self, img_name="rgb"):
        image = self.images[img_name]
        # if torch.is_tensor(image):
        #     image = image.numpy()
        return image


    def show_images(self, wk=0):
        for name in self.images.keys():
            self.show_image(name, wk)

    def get_pixel_grid(self, n = None, longtens=False, device='cpu'):
        if n is None:
            n=self.intr.resolution
        offs = (self.intr.resolution/n)/2
        x_range = torch.linspace(offs[0], self.intr.resolution[0]-offs[0], n[0])  # 5 points from -1 to 1
        y_range = torch.linspace(offs[1], self.intr.resolution[1]-offs[1], n[1])  # 5 points from -1 to 1
        X, Y = torch.meshgrid(x_range, y_range, indexing="ij")
        grid = torch.cat( (X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1 )
        grid = grid.to(device)
        if longtens:
            grid = torch.trunc(grid).to(torch.int32)
        return grid

    def sample_rand_pixs( self, num_pixels, longtens=False ):
        pixels_idxs = torch.reshape(self.get_pixel_grid( ), (-1,len(self.intr.resolution)))
        perm = torch.randperm(pixels_idxs.shape[0])
        n = min(pixels_idxs.shape[0],num_pixels )
        sampl_image_idxs = pixels_idxs[perm][:n]
        if longtens:
            sampl_image_idxs = torch.trunc(sampl_image_idxs).to(torch.int32)
        return sampl_image_idxs

    def sample_rand_pixs_in_mask( self, percentage, mask, longtens=False ):
        grid = self.get_pixel_grid( device=mask.device, longtens=longtens)
        pixels_idxs = grid[ mask> 0]
        if not pixels_idxs.numel(): return None
        pixels_idxs = torch.reshape(pixels_idxs, (-1,len(self.intr.resolution)))
        perm = torch.randperm(pixels_idxs.shape[0])
        sampl_image_idxs = pixels_idxs[perm][:(int(len(perm)*percentage))]
        if longtens:
            sampl_image_idxs = torch.trunc(sampl_image_idxs).to(torch.int32)
        return sampl_image_idxs

    def get_all_rays(self):
        grid = c.get_pixel_grid( )
        origin, dir = c.pix2ray( grid )
        return origin, dir

    def pix2dir( self, pix ):
        pix = pix*self.intr.unit_pixel_ratio() # pixels to units
        ndc = (pix - self.intr.sensor_size/2) / self.intr.lens()
        dir = torch.cat( (ndc, torch.ones( list(ndc.shape[:-1])+[1] ).to(ndc.device) ), -1 )
        dir_norm = torch.nn.functional.normalize( dir, dim=-1 )
        return torch.matmul(dir_norm, self.pose.rotation().T.to(ndc.device))

    def pix2ray( self, pix ):
        dir = self.pix2dir( pix )
        origin = self.pose.location()
        origin = repeat_tensor_to_match_shape( origin, dir.shape )
        return origin, dir

    def collect_pixs_from_img( self, image, pix ):
        assert(pix.dtype==torch.int32)
        return image[pix[:,0], pix[:,1],...]

    def get_overlayed_image( self, obj, image_name='rgb' ):
        # image = self.get_image(image_name)
        image = self.get_image(image_name)
        gbuffer = Renderer.render(self, obj, ["mask"], with_antialiasing=True)
        overlayed = (gbuffer["mask"].to(image.device) + 1.0) * image
        overlayed = overlayed.clamp_(min=0.0, max=1.0).cpu()
        return overlayed

    # projection
    def get_points_wrt_cam( self, points):
        assert(torch.is_tensor(points))
        assert(points.shape[-1]==3)
        assert(len(points.shape)==2)
        points = points.to(self.typ)
        T = self.pose.get_inverse()
        # points_wrt_cam = torch.matmul( points, T[...,:3,:3].transpose(-2,-1) ) + T[...,:3,-1] 
        points_wrt_cam = torch.matmul( points, T[...,:3,:3].transpose(-2,-1) ) + T[...,:3,-1] 
        return points_wrt_cam

    def project_points_in_cam( self, points_wrt_cam, longtens=True):
        assert(torch.is_tensor(points_wrt_cam))
        assert(points_wrt_cam.shape[-1]==3)
        assert(len(points_wrt_cam.shape)==2)
        points_wrt_cam *= self.intr.pixel_unit_ratio()
        uv = points_wrt_cam @ torch.transpose(self.intr.K_pix_und, -2,-1)
        pixels = uv[..., :2] / uv[..., 2:]
        # pixels = torch.index_select(pixels, 1, torch.LongTensor([1,0]))
        if longtens:
            pixels = torch.trunc(pixels).to(torch.int32)
        return pixels

    def project_points_opencv( self, points ):
        #points from units to pixels
        points_wrt_cam = self.get_points_wrt_cam(points)
        points_pix = (points_wrt_cam * self.intr.pixel_unit_ratio()).numpy()
        # rvec, _ = cv2.Rodrigues(self.pose.rotation().numpy())
        rvec= np.array([[0,0,0]], dtype='float32')
        # tvec = (self.pose.location() * self.intr.pixel_unit_ratio()).numpy()
        tvec= np.array([[0,0,0]], dtype='float32')
        K = self.intr.K_pix.numpy()
        D = self.intr.D.numpy()
        # D = np.array([[0,0,0,0]], dtype='float32')
        proj_points, _ = cv2.projectPoints( points_pix, rvec, tvec, K, D )
        proj_points = np.squeeze(proj_points)
        proj_points = np.trunc(proj_points)
        proj_points = proj_points.astype('int32')
        return proj_points

    def project_points( self, points, longtens=True ):
        assert(torch.is_tensor(points))
        assert(points.shape[-1]==3)
        assert(len(points.shape)==2)
        points_wrt_cam = self.get_points_wrt_cam(points)
        pixels = self.project_points_in_cam( points_wrt_cam, longtens)
        return pixels




    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t).
        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """
        # 
        points_c = points @ torch.transpose(self.pose.rotation(), 0, 1) + self.pose.location()
        pixels = points_c @ torch.transpose(self.intr.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)
        return torch.cat([pixels, depths], dim=-1)




# class Camera_on_sphere(Camera_cv):
    
#     def __init__(self, az_el, az_el_idx, K=torch.FloatTensor( [[30,0,18],[0,30,18],[0,0,1]]), pose=None, resolution=torch.LongTensor([700,700]), images=None, name="Unk Cam on sphere" ):
#         super().__init__(K=K, pose=pose, resolution=resolution, images=images, name=name)
#         self.alpha = az_el

#     def pix2eps( self, pix ):
#         assert( pix.dtype==torch.float32)
#         eps = -torch.arctan2(((pix-(self.intr.resolution/2))*self.millimeters_pixel_ratio), self.lens())
#         return eps

#     def get_sample_from_pixs( self, pixs ):
#         eps = self.pix2eps( pixs )
#         alpha = repeat_tensor_to_match_shape(self.alpha, eps.shape)
#         sample = { 'eps':eps, 'alpha':alpha }
#         return sample

#     def sample_pixels_from_err_img( self, num_pixels, show=True ):
#         pixs = sample_from_image_pdf( self.images["err"], num_pixels )
        
#         # show sampled pixels
#         if show:
#             show_pixs(pixs, self.images["err"].shape,wk=1)

#         pixs = torch.FloatTensor( pixs+0.5 )
#         return pixs



#     def render_ddf( self, ddf, device, wk=0, update_err=False, path_err="" , prt=False):

#         # for k,v in self.images.items():
#         #     if k=="err":
#         #         continue
#         #     show_image( k+":_gt", v.numpy(), wk )

#         with torch.no_grad():
#             grid = self.get_pixel_grid()
#             sample = self.get_sample_from_pixs( grid )
#             sample = dict_to_device(sample, device)
#             output = ddf.forward( sample ).detach().cpu()
#             count = 0
#             errs = []
#             for k,v in self.images.items():
#                 if k=="err":
#                     continue
#                 o = output[...,count:count+v.shape[-1]]
#                 count += v.shape[-1]
#                 errs.append(torch.abs(o-v))
#                 show_image( k, o.numpy(), wk )

#             if update_err:
#                 err = torch.cat( errs, dim=-1 )
#                 err = torch.norm(err, dim=-1)
#                 if prt:
#                     print("err: "+str(torch.sum(err).item()))
#                 cv2.imwrite(path_err+"/"+self.name+".png", err.numpy()*255)
#                 show_image("err", err, wk)

if __name__=="__main__":
    # c = Camera_on_sphere()
    c = Camera_cv()
    c.sample_pixels( 10 )
    c.pose.rotate_euler(eul(torch.FloatTensor([math.pi/4,math.pi/3,0])))
    c.pose.set_location(torch.FloatTensor([0.5,0.2,-0.1]))
    pixs = c.get_pixel_grid( torch.LongTensor([10,10]) )
    # pixs = c.sample_pixels( 10 )
    origin, dir = c.pix2ray( pixs )
    p = plotter
    p.plot_ray(origin, dir)
    p.plot_cam(c, 1)
    p.plot_frame(c.pose)
    p.show()








import cv2
import torch
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as T
from skimage import feature, filters

class Image():
    def __init__(self, img=None, path=None, gray=False, resolution_drop=1., device='cpu'):
        assert((img is None) ^ (path is None))
        self.device = device
        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            self.img = img.to(device)
        if path is not None:
            self.img = torch.from_numpy(cv2.imread(path)).to(device)
            self.img = self.img[:,:,[2,1,0]]
            # self.img = self.swapped()

        if resolution_drop!=1.:
            self.resize(resolution_drop=resolution_drop)

        if gray:
        # if self.is_grayscale(self.img):
            self.img = self.img[...,0]
            # self.img = self.img[...,0:1]
            # self.img = torch.mean( self.img, dim=-1, dtype=torch.uint8)

        self.shape = self.img.shape

    def swapped(self):
        return torch.swapaxes(self.img,0,1)

    def float(self):
        if self.img.dtype == torch.float32:
            return self.img
        img = self.img.type(torch.float32)/255
        return img

    def is_grayscale( self, image ):
        if image.shape[-1]==3:
            b1 = torch.max(torch.abs(image[...,0]-image[...,1]))==0
            b2 = torch.max(torch.abs(image[...,0]-image[...,2]))==0
            return b1 and b2
        return False

    def get_gray_cmap( self , cmap='viridis'):
        dtype = self.img.dtype
        c = plt.get_cmap(cmap)
        colormap_tensor = c(self.img.cpu().view(-1).numpy())
        s = self.img.squeeze().shape
        rgb_tensor = torch.flip(torch.from_numpy(colormap_tensor[:, :3]), dims=[-1]).view( tuple(list(s)+[3]) ).to(self.device)
        rgb_tensor = rgb_tensor.type(dtype)
        return rgb_tensor


    def shape(self):
        return self.img.shape

    def resize(self, resolution=None, resolution_drop=None):
        assert( (resolution is None) ^ (resolution_drop is None) )
        if(resolution is not None):
            self.img = torch.from_numpy(self.resized(resolution)).to(self.device)
        elif(resolution_drop is not None):
            r = self.resolution()*resolution_drop
            self.img = torch.from_numpy(self.resized(r.type(torch.LongTensor))).to(self.device)

    def resolution(self):
        return torch.LongTensor([self.img.shape[0],self.img.shape[1]])

    def resized(self, resolution):
        resized = cv2.resize(self.numpy(), (int(resolution[1]), int(resolution[0])), interpolation= cv2.INTER_LINEAR)
        return resized

    def clone(self):
        return self.img.detach().clone()

    def gray(self):
        if len(self.shape)>2:
            gray = self.float()
            gray = gray.mean(dim=-1)
            # return gray.to
            return gray
        else: return self.img

    def numpy(self):
        return self.img.detach().cpu().numpy()

    def show(self, img_name="Unk", wk=0, resolution_drop = 1):
        resized = cv2.resize(self.numpy(), (int(self.img.shape[1]/resolution_drop), int(self.img.shape[0]/resolution_drop)), interpolation= cv2.INTER_LINEAR)
        cv2.imshow(img_name, resized)
        cv2.waitKey(wk)

    def save(self, img_path):
        cv2.imwrite(img_path, self.numpy())

    def get_indices_with_val(self, val):
        indices = torch.nonzero(self.img == val)
        indices = indices.to(torch.int32)
        return indices

    def sobel(self):
        # trans_lookup = T.Compose([
        #     T.Grayscale(),
        #     T.ToPILImage(),
        # ])
        # img_new = self.clone()
        # img_new = trans_lookup(torch.swapaxes(self.img,0,-1))
        # # img_new = img_new.filter(ImageFilter.FIND_EDGES)
        # img_new = feature.canny(img_new)
        # img_new = T.ToTensor()(img_new)  # Convert the image to a PyTorch tensor
        # img_new = torch.swapaxes(img_new,0,-1)

        # print(self.gray().numpy().shape)
        # img_new = torch.from_numpy(feature.canny(self.gray().numpy()))
        img_new = torch.from_numpy(filters.sobel(self.gray().numpy()))
        return Image(img_new)

    def max_pooling(self, kernel_size=5):
        image_curr = F.max_pool2d(self.img.type(torch.float32).unsqueeze(-1), kernel_size=kernel_size, stride=(1), padding=int(kernel_size/2)).type(torch.uint8)
        image_curr = image_curr.squeeze(-1)
        # cv2.imshow("",image_curr.numpy())
        # cv2.waitKey(0)
        return image_curr

    import torch.nn.functional as F

    def eval_bilinear(self, pixels, top_left):

        top_right       = top_left+torch.tensor([1,0], device=pixels.device)
        bottom_left     = top_left+torch.tensor([0,1], device=pixels.device)
        bottom_right    = top_left+torch.tensor([1,1], device=pixels.device)

        frac_vertical = ( (pixels[:,1] - top_left[:,1])*0.5 ).unsqueeze(-1)
        frac_horizont = ( (pixels[:,0] - top_left[:,0])*0.5 ).unsqueeze(-1)

        img = self.float().to(pixels.device)
        top_left_rgb     = img[top_left[:,1].int()     , top_left[:,0].int()]
        top_right_rgb    = img[top_right[:,1].int()    , top_right[:,0].int()]
        bottom_left_rgb  = img[bottom_left[:,1].int()  , bottom_left[:,0].int()]
        bottom_right_rgb = img[bottom_right[:,1].int() , bottom_right[:,0].int()]

        top_interpolation = top_left_rgb * (1 - frac_horizont) + top_right_rgb * frac_horizont
        bottom_interpolation = bottom_left_rgb * (1 - frac_horizont) + bottom_right_rgb * frac_horizont
        interpolated_rgb = top_interpolation * (1 - frac_vertical) + bottom_interpolation * frac_vertical

        return interpolated_rgb


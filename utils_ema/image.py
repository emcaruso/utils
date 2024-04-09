import cv2
import torch
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as T
from skimage import feature, filters
try:
    from .general import get_monitor
except:
    from general import get_monitor

m = get_monitor()

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

    def to(self, device):
        self.device=device
        self.img = self.img.to(device)
        return self

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

    def get_gray_cmap( self , cmap='nipy_spectral'):
        dtype = self.img.dtype
        c = plt.get_cmap(cmap)
        gray = self.gray()
        colormap_tensor = c(gray.cpu().view(-1).numpy())
        s = gray.squeeze().shape
        rgb_tensor = torch.flip(torch.from_numpy(colormap_tensor[:, :3]), dims=[-1]).view( tuple(list(s)+[3]) ).to(self.device)
        rgb_tensor = rgb_tensor.type(dtype)
        return Image(rgb_tensor)

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

    def gray(self, keepdim=False):
        if len(self.shape)>2:
            gray = self.float()
            gray = gray.mean(dim=-1, keepdim=keepdim)
            print(keepdim)
            print(gray.shape)
            # return gray.to
            return gray
        else: return self.img

    def numpy(self):
        return self.img.detach().cpu().numpy()

    def show(self, img_name="Unk", wk=0, resolution_drop = 1):
        resized = cv2.resize(self.numpy(), (int(self.img.shape[1]/resolution_drop), int(self.img.shape[0]/resolution_drop)), interpolation= cv2.INTER_LINEAR)
        cv2.imshow(img_name, resized)
        key = cv2.waitKey(wk)
        return key


    def save(self, img_path, verbose=False):
        cv2.imwrite(img_path, self.numpy())
        print("saved image in: ", img_path)

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

    def get_pix_max_intensity(self, dtype=torch.float32):
        img = torch.mean(self.img, dim=-1, dtype=dtype)
        m = img.max()
        pix_max = torch.nonzero(img == m)
        pix_max = pix_max.flip(dims=[-1])
        return pix_max, m

    def get_pix_min_intensity(self, dtype=torch.float32):
        img = torch.mean(self.img, dim=-1, dtype=dtype)
        m = img.min()
        pix_min = torch.nonzero(img == m)
        pix_min = pix_min.flip(dims=[-1])
        return pix_min, m

    def get_intensity_mean(self, dtype=torch.float32):
        m = torch.mean(self.img, dtype=dtype)
        m = m.flip(dims=[-1])
        return m

        # print(pix_max.shape)
        # self.show()
        # flat_index = img.argmax()
        # coords = torch.unravel_index(flat_index, img.shape)
        # index = img.argmax()
        # print(index)

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

    def sample_pixels(self, num_samples):
        """
        Samples pixels from a one-channel image (torch tensor) based on the pixel values as probabilities.

        Args:
        image (torch.Tensor): A one-channel image tensor.
        num_samples (int): Number of pixels to sample.

        Returns:
        torch.Tensor: Indices of sampled pixels.
        """

        if len(self.img.shape)>=2:
            new_img = self.img.mean(dim=-1)
        else:
            new_img = self.img

        new_img = torch.pow(new_img, 2)
            
        # Flatten the image and normalize the pixel values to get probabilities
        flat_image = new_img.flatten()
        probabilities = flat_image / flat_image.sum()

        # Sample pixel indices based on the computed probabilities
        sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True)

        # Convert flat indices to 2D indices
        rows = sampled_indices // new_img.shape[1]
        cols = sampled_indices % new_img.shape[1]

        return torch.stack((cols, rows), dim=1)

    def draw_circles(self, centers, radius=3, color=(255,0,255), thickness=2):
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers.astype(np.int32))

        # centers = centers.flip(dims=[-1])
        centers = torch.flip(centers.type(torch.int32), dims = [-1])

        img = self.numpy()
        for center in centers:
            cv2.circle(img, center.numpy(), radius, color, thickness)
            # cv2.circle(img, (100,100), radius, color, thickness)
        return Image(img)

    def show_points(self, coords=[], method="cv2", wk=1, name="unk"):
        if method=="plt":
            plt.imshow(self.numpy().astype(np.uint8))  # Cast to uint8 for image display
            for y, x in coords:
                plt.plot(x, y, 'ro')  # 'ro' for red circle; adjust color and marker as needed
            plt.show()
        elif method == "cv2":
            img = Image(self.numpy())
            coords = torch.flip(coords, dims=[-1])
            for coord in coords:
                img.draw_circles(coord, radius=3, color=(0, 0, 255), thickness=-1)
            key = img.show(img_name=name, wk=wk)
            return key


    @staticmethod
    def show_multiple_images( images, wk=0, name="image", undistort=None, cams=None ):
        for i, img in enumerate(images):

            if undistort is not None:
                assert cams is not None
                cam = cams[i]
                img = cam.intr.undistort_image( img )

            img = img.numpy()
            resized = cv2.resize(img, (int(m.width/2), int(m.height/2)), interpolation= cv2.INTER_LINEAR)
            winname=name+"_"+str(i).zfill(3)
            cv2.namedWindow(winname)        # Create a named window
            cv2.moveWindow(winname,  int(((i%2)==1)*(m.width/2)),int((i>1)*(m.height/2)) )
            cv2.imshow(winname, resized)
        key = cv2.waitKey(wk)
        return key



import cv2
import torch
import os, sys
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as T

class Image():
    def __init__(self, img=None, path=None, gray=False, device='cpu'):
        assert((img is None) ^ (path is None))
        self.device = device
        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            self.img = img.to(device)
            return
        if path is not None:
            self.img = torch.from_numpy(cv2.imread(path)).to(device)

        if self.is_grayscale(self.img):
            self.img = self.img[...,0]
            # self.img = self.img[...,0:1]
            # self.img = torch.mean( self.img, dim=-1, dtype=torch.uint8)

    def float(self):
        return self.img.to(torch.float32)/255

    def is_grayscale( self, image ):
        assert(image.shape[-1]==3)
        b1 = torch.max(torch.abs(image[...,0]-image[...,1]))==0
        b2 = torch.max(torch.abs(image[...,0]-image[...,2]))==0
        return b1 and b2

    def shape(self):
        return self.img.shape

    def clone(self):
        return self.img.detach().clone()

    def gray(self):
        if len(self.shape)>2:
            gray = self.img.to(torch.float32)
            gray = gray.mean(dim=-1)
            # return gray.to
            return gray
        else: return self.img

    def numpy(self):
        return self.img.cpu().numpy()

    def show(self, img_name="Unk", wk=0, drop_rate = 1):
        resized = cv2.resize(self.numpy(), (int(self.img.shape[1]/drop_rate), int(self.img.shape[0]/drop_rate)), interpolation= cv2.INTER_LINEAR)
        cv2.imshow(img_name, self.numpy())
        cv2.waitKey(wk)

    def get_indices_with_val(self, val):
        indices = torch.nonzero(self.img == val)
        indices = indices.to(torch.int32)
        return indices

    def sobel(self):
        trans_lookup = T.Compose([
            T.Grayscale(),
            T.ToPILImage(),
        ])
        img_new = self.clone()
        img_new = trans_lookup(torch.swapaxes(self.img,0,-1))
        img_new = img_new.filter(ImageFilter.FIND_EDGES)
        img_new = T.ToTensor()(img_new)  # Convert the image to a PyTorch tensor
        img_new = torch.swapaxes(img_new,0,-1)

        return Image(img_new)

    def max_pooling(self, kernel_size=5):
        image_curr = F.max_pool2d(self.img.type(torch.float32).unsqueeze(-1), kernel_size=kernel_size, stride=(1), padding=int(kernel_size/2)).type(torch.uint8)
        image_curr = image_curr.squeeze(-1)
        # cv2.imshow("",image_curr.numpy())
        # cv2.waitKey(0)
        return image_curr


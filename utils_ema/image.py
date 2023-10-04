import cv2
import torch
import os, sys

class Image():
    def __init__(self, img=None, path=None, gray=False, device='cpu'):
        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            self.img = img.to(device)
            return
        if path is not None:
            self.img = torch.from_numpy(cv2.imread(path)).to(device)
        if gray:
            self.img = self.img[...,0]
            # self.img = torch.mean( self.img, dim=-1, dtype=torch.uint8)


    def get_indices_with_val(self, val):
        # print(self.img.shape)
        
        indices = torch.nonzero(self.img == val)
        indices = indices.to(torch.int32)
        # indices = torch.nonzero(self.img > 0.5)
        return indices

    def sobel(self):
        pass

    def max_pooling(self):
        pass

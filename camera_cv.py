import cv2
import torch
import sys, os

from geometry_pose import *


class Camera_cv():

    def __init__(self, K=torch.eye(3, dtype=torch.float32), w_T_cam=torch.eye(4, dtype=torch.float32) ):
        self.K = K
        self.frame = frame(w_T_cam)

    def cx(self): return self.K[0,2]
    def cy(self): return self.K[1,2]
    def fx(self): return self.K[0,0]
    def fy(self): return self.K[1,1]



class Camera_on_sphere(Camera_cv):
    
    def __init__(self, K=torch.eye(3, dtype=torch.float32), w_T_cam=torch.eye(4, dtype=torch.float32) ):
        super().__init__(K, w_T_cam)


if __name__=="__main__":
    c = Camera_on_sphere()

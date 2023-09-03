import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys, os

try:
    from .geometry_euler import *
    from .plot import *
except:
    from geometry_euler import *
    from plot import *


class Pose():

    def __init__(self, T=torch.eye(4, dtype=torch.float32), units='meters'):
        assert(T.shape[-2:]==(4,4))
        self.T = T
        self.units = units

    def location(self): return self.T[...,:3,-1]
    def rotation(self): return self.T[...,:3,:3]
    def set_pose(self, T): self.T = T
    def set_location(self, new_loc): self.T[...,:3,-1] = new_loc
    def set_rotation(self, new_rot): self.T[...,:3,:3] = new_rot
    def set_euler(self, e): self.set_rotation(e.eul2rot())
    # def rotate(self, rot): self.set_rotation(torch.matmul(rot, self.rotation()))
    def rotate(self, rot): self.set_rotation(torch.matmul(self.rotation(), rot))
    def rotate_euler(self, e): self.rotate(e.eul2rot())
    def move_location(self, v): self.set_location(self.location()+v)
    def transform(self, T_tr):
        self.rotate(T_tr[...,:3,:3])
        self.set_location( T_tr[...,:3,:3]@self.location()+T_tr[...,:3,-1])
    def to(self, device):
        self.T = self.T.to(device)
        return self

    def __eq__(self, other):
        return torch.equal(self.T,other.T)

if __name__ == "__main__":
    p = Pose()
    pl = plotter()

    # for i in np.arange(0,math.pi/4, 0.1):
    #     p.set_euler(eul(torch.FloatTensor([i,0,0])))

    for i in range(10):
        pl.init_figure()
        p.rotate_euler(eul(torch.FloatTensor([0.1,0,0])))
        pl.plot_frame(p)
        pl.show()

#     for i in range(10):
#         p.move_location(torch.FloatTensor([0.1,0,0]))

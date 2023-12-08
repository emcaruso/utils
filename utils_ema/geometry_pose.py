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

    def __init__(self, euler:eul = eul(torch.zeros([3], dtype=torch.float32)), position = torch.zeros([3], dtype=torch.float32), T=None , units='meters', device='cpu'):
        assert(isinstance(euler,eul))
        assert(torch.is_tensor(position))

        if T is not None:
            self.euler = euler
            self.euler = self.euler.rot2eul( R= T[...,:3,:3] )
            self.position = T[...,:3,-1]
        else:
            self.euler = euler
            self.position = position

        self.units = units
        self.device = device

    def location(self):
        return self.position
    def rotation(self):
        return self.euler.eul2rot()
    def set_pose_from_T(self, T):
        assert(torch.is_tensor(T))
        R = T[...,:3,:3]
        t = T[...,:3,-1]
        self.euler = self.euler.rot2eul(R)
        self.position = t
    def set_pose(self, euler, position):
        self.euler = euler
        self.position = position
    def set_location(self, new_loc):
        self.position = new_loc
    def set_rotation(self, R):
        assert(torch.is_tensor(R))
        self.euler = self.euler.rot2eul(R)
    def set_euler(self, e):
        assert(isinstance(e, eul))
        self.euler = e
    def rotate_by_R(self, R):
        self.euler.rotate_by_R(R)
    def rotate_by_euler(self, e): 
        self.euler.rotate_by_euler(e)
    def move_location(self, v): self.set_location(self.location()+v)
    def move_location_local(self, v): self.set_location( self.location() + (self.rotation() @ v) )
    def uniform_scale(self, s:float, units:str="scaled"):
        self.set_location(self.location()*s)
        self.units = units
    def scale(self, s:torch.FloatTensor): self.set_location(self.location()*s)

    def transform(self, T_tr):
        R = T_tr[...,:3,:3]
        t = T_tr[...,:3,-1]
        self.euler.rotate_by_R(R)
        new_loc = R@self.location()+t
        self.set_location( new_loc )

    def to(self, device):
        self.euler = self.euler.to(device)
        self.position = self.position.to(device)
        self.device = device
        return self
    def dtype(self, dtype):
        self.euler = self.euler.to(dtype)
        self.position = self.position.to(dtype)
        return self
    def invert(self):
        R = self.euler.eul2rot()
        R_t = R.transpose(-2,-1)
        self.euler = self.euler.rot2eul(R_t)
        self.position = -R_t @ self.location()

    def get_R_inv(self):
        return self.rotation().transpose(-2,-1)

    def get_t_inv(self):
        R_inv = self.get_R_inv()
        return - R_inv @ self.location()

    def get_T(self):
        R = self.rotation()
        t = self.location()
        new_shape = list(R.shape)
        new_shape[-1]=4
        new_shape[-2]=4
        T = torch.zeros( *new_shape , dtype= t.dtype)
        T[...,:3,:3] = R
        T[...,:3,-1] = t
        T[...,3,3] = 1
        return T

    def get_T_inverse(self):
        R_inv = self.rotation().transpose(-2,-1)
        t_inv = - R_inv @ self.location()
        new_shape = list(R_inv.shape)
        new_shape[-1]=4
        new_shape[-2]=4
        T_inv = torch.zeros( *new_shape , dtype= self.T.dtype)
        T_inv[...,:3,:3] = R_inv
        T_inv[...,:3,-1] = t_inv
        T_inv[...,3,3] = 1
        return T_inv

    # def to(self, device):
    #     self.T = self.T.to(device)
    #     self.device = device
    #     return self
    # def dtype(self, dtype):
    #     self.T = self.T.to(dtype)
    #     return self
    # def invert(self):
    #     self.T = self.get_inverse()
    # def get_inverse(self):
    #     R_inv = self.rotation().transpose(-2,-1)
    #     t_inv = - R_inv @ self.location()
    #     new_shape = list(R_inv.shape)
    #     new_shape[-1]=4
    #     new_shape[-2]=4
    #     T_inv = torch.zeros( *new_shape , dtype= self.T.dtype)
    #     T_inv[...,:3,:3] = R_inv
    #     T_inv[...,:3,-1] = t_inv
    #     T_inv[...,3,3] = 1
    #     return T_inv

    def __eq__(self, other):
        b1 = torch.equal(self.position,other.position)
        b2 = torch.equal(self.euler.e,other.euler.e)
        return (b1 and b2)

# clpositionss Pose():

#     def __init__(self, T=torch.eye(4, dtype=torch.float32), units='meters', device='cpu'):
#         assert(T.shape[-2:]==(4,4))
#         self.T = T
#         self.units = units
#         self.device = device

#     def location(self): return self.T[...,:3,-1]
#     def rotation(self): return self.T[...,:3,:3]
#     def set_pose(self, T): self.T = T
#     def set_location(self, new_loc): self.T[...,:3,-1] = new_loc
#     def set_rotation(self, new_rot): self.T[...,:3,:3] = new_rot
#     def set_euler(self, e): self.set_rotation(e.eul2rot())
#     # def rotate(self, rot): self.set_rotation(torch.matmul(rot, self.rotation()))
#     def rotate(self, rot): self.set_rotation(torch.matmul(self.rotation(), rot))
#     def rotate_euler(self, e): self.rotate(e.eul2rot())
#     def move_location(self, v): self.set_location(self.location()+v)
#     def move_location_local(self, v): self.set_location( self.location() + (self.rotation() @ v) )
#     def uniform_scale(self, s:float, units:str="scaled"):
#         self.set_location(self.location()*s)
#         self.units = units
#     def scale(self, s:torch.FloatTensor): self.set_location(self.location()*s)
#     def transform(self, T_tr):
#         self.rotate(T_tr[...,:3,:3])
#         self.set_location( T_tr[...,:3,:3]@self.location()+T_tr[...,:3,-1])
#     def to(self, device):
#         self.T = self.T.to(device)
#         self.device = device
#         return self
#     def dtype(self, dtype):
#         self.T = self.T.to(dtype)
#         return self
#     def invert(self):
#         self.T = self.get_inverse()
#     def get_inverse(self):
#         R_inv = self.rotation().transpose(-2,-1)
#         t_inv = - R_inv @ self.location()
#         new_shape = list(R_inv.shape)
#         new_shape[-1]=4
#         new_shape[-2]=4
#         T_inv = torch.zeros( *new_shape , dtype= self.T.dtype)
#         T_inv[...,:3,:3] = R_inv
#         T_inv[...,:3,-1] = t_inv
#         T_inv[...,3,3] = 1
#         return T_inv

#     def __eq__(self, other):
#         return torch.equal(self.T,other.T)

if __name__ == "__main__":
    p = Pose()

    # for i in np.arange(0,math.pi/4, 0.1):
    #     p.set_euler(eul(torch.FloatTensor([i,0,0])))

    for i in range(10):
        eul_rot = eul(torch.FloatTensor([0.1,0,0]))

        p.rotate_by_euler(eul_rot)
        plotter.plot_frame(p)
    plotter.show()

#     for i in range(10):

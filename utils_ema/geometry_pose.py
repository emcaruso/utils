import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys, os
import copy as cp
from itertools import permutations


try:
    from .geometry_euler import eul
    from .plot import *
except:
    from geometry_euler import eul
    from plot import *


class Pose:

    def __init__(
        self,
        euler: eul = eul(torch.zeros([3], dtype=torch.float32)),
        position=torch.zeros([3], dtype=torch.float32),
        T=None,
        # scale=torch.ones([3], dtype=torch.float32),
        scale=torch.ones([1], dtype=torch.float32),
        units="meters",
        device="cpu",
    ):
        assert isinstance(euler, eul)
        assert torch.is_tensor(position)

        if T is not None:
            self.euler = euler
            R = R = T[..., :3, :3]
            if not eul.is_rotation_matrix(R):
                raise ValueError("T has not a valid rotation matrix")
            self.euler = self.euler.rot2eul(R=R)
            self.position = T[..., :3, -1]
        else:
            self.euler = euler
            self.position = position
        self.scale = scale.to(self.position.dtype)

        self.units = units
        self.device = device

    @staticmethod
    def cvvecs2pose(rvec, tvec):
        rot, _ = cv2.Rodrigues(rvec)

        # rot = rot.transpose()

        # axis_permutations = list(permutations([0, 1, 2]))
        # for perm in axis_permutations:
        #     prot = rot[np.ix_(perm, perm)]
        #     print(perm)
        #     # rot = rot[[2, 0, 1], :]
        #     # rot = rot.transpose()
        #     euler = eul(torch.zeros([3], dtype=torch.float32))
        #     euler = euler.rot2eul( torch.from_numpy(prot) )
        #     # euler.e = euler.e[[1,0,2]]
        #     position = torch.from_numpy(tvec).squeeze(-1)
        #     # pose = Pose(euler = euler, position = position.unsqueeze(0))
        #     pose = Pose(euler = euler, position = position)
        #     plotter.plot_pose(pose)
        #     plotter.show()
        #     plotter.reset()
        # #     break

        euler = eul(torch.zeros([3], dtype=torch.float32))
        euler = euler.rot2eul(torch.from_numpy(rot))
        # euler.e = euler.e[[1,0,2]]
        position = torch.from_numpy(tvec).squeeze(-1)
        # pose = Pose(euler = euler, position = position.unsqueeze(0))
        pose = Pose(euler=euler, position=position)
        return pose

    @staticmethod
    def pose2cvecs():
        pass

    def location(self):
        return self.position

    def rotation(self):
        return self.euler.eul2rot()

    def set_pose_from_T(self, T):
        assert torch.is_tensor(T)
        R = T[..., :3, :3]
        t = T[..., :3, -1]
        self.euler = self.euler.rot2eul(R)
        self.position = t

    def set_pose(self, euler, position):
        self.euler = euler
        self.position = position

    def set_location(self, new_loc):
        self.position = new_loc

    def set_rotation(self, R):
        assert torch.is_tensor(R)
        self.euler = self.euler.rot2eul(R)

    def set_euler(self, e):
        assert isinstance(e, eul)
        self.euler = e

    def rotate_by_R(self, R):
        self.euler.rotate_by_R(R)

    def rotate_by_euler(self, e):
        self.euler.rotate_by_euler(e)

    def move_location(self, v):
        self.set_location(self.location() + v)

    def move_location_local(self, v):
        self.set_location(self.location() + (self.rotation() @ v))

    def uniform_scale(self, s: float, units: str = "scaled"):
        self.set_location(self.location() * s)
        self.units = units

    # def scale(self, s:torch.FloatTensor): self.set_location(self.location()*s)

    def transform(self, T_tr):
        R = T_tr[..., :3, :3]
        t = T_tr[..., :3, -1]
        self.euler.rotate_by_R(R)
        new_loc = R @ self.location() + t
        self.set_location(new_loc)

    def to(self, device):
        self.euler = self.euler.to(device)
        self.position = self.position.to(device)
        self.scale = self.scale.to(device)
        self.device = device
        return self

    def dtype(self, dtype):
        self.euler = self.euler.to(dtype)
        self.position = self.position.to(dtype)
        self.scale = self.scale.to(dtype)
        return self

    def invert(self):
        R = self.euler.eul2rot()
        R_t = R.transpose(-2, -1)
        self.euler = self.euler.rot2eul(R_t)
        self.position = -R_t @ self.location()
        return self

    def get_R_inv(self):
        return self.rotation().transpose(-2, -1)

    def get_t_inv(self):
        R_inv = self.get_R_inv()
        return -R_inv @ self.location()

    def get_T(self):
        R = self.rotation()
        t = self.location()
        new_shape = list(R.shape)
        new_shape[-1] = 4
        new_shape[-2] = 4
        T = torch.zeros(*new_shape, dtype=t.dtype)
        T[..., :3, :3] = R
        T[..., :3, -1] = t
        T[..., 3, 3] = 1
        return T

    def get_T_inverse(self):
        R_inv = self.rotation().transpose(-2, -1)
        t_inv = -R_inv @ self.location()
        new_shape = list(R_inv.shape)
        new_shape[-1] = 4
        new_shape[-2] = 4
        T_inv = torch.zeros(*new_shape, dtype=self.get_T().dtype)
        T_inv[..., :3, :3] = R_inv
        T_inv[..., :3, -1] = t_inv
        T_inv[..., 3, 3] = 1
        return T_inv

    def clone(self):

        euler = cp.deepcopy(self.euler)
        position = cp.deepcopy(self.position)
        scale = cp.deepcopy(self.scale)

        pose = Pose(
            euler=euler,
            position=position,
            scale=scale,
            units=self.units,
            device=self.device,
        )
        return pose

        # if same_pose: new_pose = self.pose
        # else: new_pose = cp.deepcopy(self.pose)

        # if image_paths is None: image_paths = self.image_paths
        # if name is None: name = self.name+"_copy"

        # new_cam = Camera_cv(new_intr, new_pose, image_paths, self.frame, name, device = self.device, dtype=self.typ)
        # return new_cam

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
        b1 = torch.equal(self.position, other.position)
        b2 = torch.equal(self.euler.e, other.euler.e)
        return b1 and b2

    def __mul__(self, other):

        T_self = self.get_T()
        # R_self = self.rotation()
        # t_self = self.location()
        # s_self = self.scale
        T_other = other.get_T()
        # R_other = other.rotation()
        # t_other = other.location()
        # s_other = other.scale

        T = T_self @ T_other
        pose = Pose(T=T)
        # R = pose.rotation()
        # t = pose.location()

        # R_ = R_self@R_other

        # print(T_self)
        # print(R_self)
        # print(t_self)
        # print(s_self)
        # print(T_other)
        # print(R_other)
        # print(t_other)
        # print(s_other)
        # print(R)
        # print(t)

        return pose

    @classmethod
    def average_poses(cls, pose_array):
        assert hasattr(pose_array, "__iter__")

        Rs = []
        ts = []
        for pose in pose_array:
            if pose is not None:
                assert type(pose) == cls
                Rs.append(pose.rotation().unsqueeze(0))
                ts.append(pose.location().unsqueeze(0))

        R = torch.cat(Rs, dim=0)
        t = torch.cat(ts, dim=0)
        R_avg = R.mean(dim=0)
        U, _, Vt = torch.svd(R_avg)
        R_mean = torch.matmul(U, Vt.t())
        t_mean = t.mean(dim=0)
        e = eul.rot2eul_YXZ(R=R_mean)
        pose_mean = Pose(euler=e, position=t_mean)
        return pose_mean

    def dist(self, other):
        t_i = self.location()
        t_o = other.location()
        assert t_i.shape[0] == t_o.shape[0]
        t_norm = torch.norm(t_i - t_o)

        e_i = self.euler
        e_o = other.euler
        angle = e_i.dist(e_o)
        return t_norm, angle

        # assert(e_i.shape[0]==e_o.shape[0])


if __name__ == "__main__":
    p = Pose()

    # for i in np.arange(0,math.pi/4, 0.1):
    #     p.set_euler(eul(torch.FloatTensor([i,0,0])))

    for i in range(10):
        eul_rot = eul(torch.FloatTensor([0.1, 0, 0]))

        p.rotate_by_euler(eul_rot)
        plotter.plot_frame(p)
    plotter.show()

#     for i in range(10):

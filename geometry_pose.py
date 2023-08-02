import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys, os
from geometry_euler import *


class frame():

    def __init__(self, w_T_cam=torch.eye(4, dtype=torch.float32)):
        self.w_T_cam = w_T_cam

    def location(self): return self.w_T_cam[...,-1,:3]
    def rotation(self): return self.w_T_cam[...,:3,:3]
    def set_pose(self, w_T_cam): self.w_T_cam = w_T_cam
    def set_location(self, new_loc): self.w_T_cam[...,-1,:3] = new_loc
    def set_rotation(self, new_rot): self.w_T_cam[...,:3,:3] = new_rot
    def set_euler(self, e): self.set_rotation(e.eul2rot())
    def rotate(self, rot): self.set_rotation(torch.matmul(rot, self.rotation()))
    def rotate_euler(self, r): self.rotate(r.eul2rot())
    def move_location(self, v): self.set_location(self.location()+v)

    def show_pose(self, bound=2):
        x_axis_end = torch.FloatTensor((1, 0, 0))
        y_axis_end = torch.FloatTensor((0, 1, 0))
        z_axis_end = torch.FloatTensor((0, 0, 1))
        x_axis_end_rot = torch.matmul(self.rotation(), x_axis_end)
        y_axis_end_rot = torch.matmul(self.rotation(), y_axis_end)
        z_axis_end_rot = torch.matmul(self.rotation(), z_axis_end)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        a_x = self.location()+x_axis_end_rot
        a_y = self.location()+y_axis_end_rot
        a_z = self.location()+z_axis_end_rot
        ax.plot([self.location()[0], a_x[0]], [self.location()[1], a_x[1]], [self.location()[2], a_x[2]], 'r-', label='Y-axis')
        ax.plot([self.location()[0], a_y[0]], [self.location()[1], a_y[1]], [self.location()[2], a_y[2]], 'g-', label='Y-axis')
        ax.plot([self.location()[0], a_z[0]], [self.location()[1], a_z[1]], [self.location()[2], a_z[2]], 'b-', label='Z-axis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-bound,bound)
        ax.set_ylim(-bound,bound)
        ax.set_zlim(-bound,bound)
        ax.legend()
        plt.show()

if __name__ == "__main__":
    p = frame()

    for i in np.arange(0,math.pi/4, 0.1):
        p.set_euler(eul(torch.FloatTensor([i,0,0])))
        p.show_pose()

    # for i in range(10):
    #     p.rotate_euler(eul(torch.FloatTensor([0.1,0,0])))
    #     p.show_pose()

#     for i in range(10):
#         p.move_location(torch.FloatTensor([0.1,0,0]))
#         p.show_pose()

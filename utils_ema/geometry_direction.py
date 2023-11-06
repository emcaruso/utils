import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys, os

try:
    from .geometry_pose import *
    from .geometry_euler import *
    from .plot import *
except:
    from geometry_pose import *
    from geometry_euler import *
    from plot import *


class AzimuthElevation():

    def __init__(self, azimuth=torch.FloatTensor([0]), elevation=torch.FloatTensor([0]), units='radians', device='cpu'):
        self.azimuth = azimuth
        self.elevation = elevation
        self.units = units
        self.device = device

    def add_and_clamp(self, azimuth_inc, elevation_inc):

        # azimuth manifold
        self.azimuth += azimuth_inc
        n = int(self.azimuth/(math.pi*2))
        self.azimuth -= n*math.pi*2

        #elevaton, clamp between -pi/2 and pi/2
        self.elevation = torch.clamp( self.elevation+elevation_inc, min=-math.pi/2+0.00001, max=math.pi/2-0.00001)

    # direction from center to point on the sphere described by azimuth and elevation
    def to_direction(self):
        direction = torch.stack([
            torch.cos(self.azimuth) * torch.cos(self.elevation),
            torch.sin(self.azimuth) * torch.cos(self.elevation),
            torch.sin(self.elevation) ],dim=-1) 
        return direction

    def to_pose_on_sphere(self, distance = 8):
        e = eul(torch.FloatTensor([self.azimuth, self.elevation, 0]))
        # print(euler_flat)
        # 
        pose = Pose(T=torch.eye(4, dtype=torch.float32), device=self.device)
        # pose.set_euler(e)
        # pose.rotate_euler(eul(torch.FloatTensor([0,-math.pi*0.5,math.pi])))
        pose.rotate_euler(eul(torch.FloatTensor([0,-math.pi*0.5,0])))
        pose.rotate_euler(eul(torch.FloatTensor([0,0,math.pi])))
        pose.rotate_euler(eul(torch.FloatTensor([e.e[0],0,0])))
        pose.rotate_euler(eul(torch.FloatTensor([0,e.e[1],0])))
        # pose.rotate_euler(eul(torch.FloatTensor([0,-math.pi*0.5,0])))
        pose.move_location_local(torch.FloatTensor([0,0,-distance]))
        return pose
        



if __name__ == "__main__":
    n_i = 10
    n_j = 10
    for i in range(n_i):
        for j in range(n_j):
            azel = AzimuthElevation(azimuth = (i/n_i)*math.pi, elevation = (j/n_j)*math.pi*0.5)
            pose = azel.to_pose_on_sphere()
            plotter.plot_frame(pose,size=0.1)
    plotter.show()

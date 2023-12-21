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


class Direction():

    def __init__(self, input=torch.FloatTensor([0,0]), units='radians', requires_grad=None):
        if input.shape[-1]==2:
            self.azel = input
        elif input.shape[-1]==3:
            self.azel = self.azel_from_vec3D(input) 
        else:
            raise Exception
        if requires_grad is not None:
            self.azel.requires_grad = requires_grad
        self.units = units
        self.device = input.device

    def __add__(self, other):
        if len(self.azel.shape)==2 and len(other.azel.shape)==2:
            azel = torch.cat( (self.azel , other.azel), dim=0 )
            return Direction( input=azel, units=self.units )
        else:
            assert(False)


    def normalize(self):
        # azimuth manifold
        n = (self.azel[...,0]/(math.pi*2)).type(torch.int32)
        self.azel[...,0] = self.azel[...,0] - n*math.pi*2
        self.azel[...,1] = torch.clamp( self.azel[...,1], min=-math.pi/2+0.00001, max=math.pi/2-0.00001)

    def add_and_clamp(self, azimuth_inc, elevation_inc):

        # azimuth manifold
        self.azel[...,0] += azimuth_inc
        self.azel[...,1] += elevation_inc
        self.normalize()

        #n = int(self.azel[...,0]/(math.pi*2))
        #self.azel[...,0] -= n*math.pi*2

        ##elevaton, clamp between -pi/2 and pi/2
        #self.azel[...,1] = torch.clamp( self.azel[...,1]+elevation_inc, min=-math.pi/2+0.00001, max=math.pi/2-0.00001)

    # direction from center to point on the sphere described by azimuth and elevation
    def vec3D(self):
        direction = torch.stack([
            torch.cos(self.azel[...,0]) * torch.cos(self.azel[...,1]),
            torch.sin(self.azel[...,0]) * torch.cos(self.azel[...,1]),
            torch.sin(self.azel[...,1]) ],dim=-1) 
        return direction

    ##3d vector to azimuth and elevation
    #def azel_from_vec3D(self, input):
    #    assert(input.shape[-1]==3)
    #    # x = input[...,0].unsqueeze(-1)
    #    # y = input[...,1].unsqueeze(-1)
    #    # z = input[...,2].unsqueeze(-1)
    #    x = input[...,0]
    #    y = input[...,1]
    #    z = input[...,2]
    #    return torch.stack( (torch.atan2(y, x+1e-8), torch.atan2(z, torch.sqrt( (torch.pow(x,2) + torch.pow(y,2))+1e-8))), dim=-1)
    def azel_from_vec3D(self, input):
        assert(input.shape[-1]==3)
        # x = input[...,0].unsqueeze(-1)
        # y = input[...,1].unsqueeze(-1)
        # z = input[...,2].unsqueeze(-1)
        x = input[...,0]
        y = input[...,1]
        z = input[...,2]
        # xy_norm = torch.sqrt((torch.pow(x,2) + torch.pow(y,2)))
        xy_norm = torch.norm( torch.cat( (x.unsqueeze(-1), y.unsqueeze(-1) ), dim=-1), dim=-1 )
        return torch.stack( (torch.atan2(y, x+1e-6), torch.atan2(z, xy_norm+1e-6)), dim=-1)

    # direction to pose on a sphere
    def to_pose_on_sphere(self, distance = 8):
        e = eul(torch.FloatTensor([self.azel[...,0], self.azel[...,1], 0]))
        pose = Pose()
        pose.set_pose_from_T(torch.eye(4, dtype=torch.float32))
        pose.to(self.device)
        # pose.set_euler(e)
        # pose.rotate_euler(eul(torch.FloatTensor([0,-math.pi*0.5,math.pi])))
        pose.rotate_by_euler(eul(torch.FloatTensor([0,-math.pi*0.5,0])))
        # pose.rotate_by_euler(eul(torch.FloatTensor([0,0,math.pi])))
        pose.rotate_by_euler(eul(torch.FloatTensor([e.e[0],0,0])))
        pose.rotate_by_euler(eul(torch.FloatTensor([0,e.e[1],0])))
        # pose.rotate_euler(eul(torch.FloatTensor([0,-math.pi*0.5,0])))
        pose.move_location_local(torch.FloatTensor([0,0,-distance]))
        return pose
        



if __name__ == "__main__":
    n_i = 10
    n_j = 10
    for i in range(n_i):
        for j in range(n_j):
            azel = Direction(input = [(i/n_i)*math.pi, (j/n_j)*math.pi*0.5])
            pose = azel.to_pose_on_sphere()
            plotter.plot_frame(pose,size=0.1)
    plotter.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, os
import copy as cp
from itertools import permutations
from utils_ema.geometry_euler import eul
from utils_ema.geometry_pose import Pose
from utils_ema.plot import *

class Plane():

    def __init__(self, pose=Pose(), offset_z=0.005):
        self.pose = pose
        self.units = self.pose.units
        self.device = self.pose.device

    def normal(self):
        normal = self.pose.rotation() @ torch.tensor([0,0,1], dtype=self.pose.rotation().dtype)
        return normal

    # def find_intersection(self, ray):
    def find_ray_plane_intersection(self, ray_origin, ray_direction):
        """
        Finds the intersection of a ray and a plane.

        Parameters:
        - ray_origin: Origin of the ray (3D point).
        - ray_direction: Direction vector of the ray (must be normalized).
        - plane_normal: Normal vector of the plane (must be normalized).
        - plane_point: A point that lies on the plane (3D point).

        Returns:
        - The 3D point of intersection, or None if there is no intersection.
        """

        plane_point = self.pose.location()
        plane_normal = self.normal()

        # Compute the denominator of the t equation
        denom = torch.dot(plane_normal, ray_direction)

        # If denom is close to 0, the ray is parallel to the plane (no intersection)
        if torch.isclose(denom, torch.tensor(0.0, dtype=self.pose.rotation().dtype), atol=1e-6):
            return None

        # Compute the numerator of the t equation
        num = torch.dot(plane_normal, (plane_point - ray_origin))

        # Solve for t
        t = num / denom

        # Calculate the intersection point
        intersection_point = ray_origin + t * ray_direction
        print(intersection_point)

        return intersection_point

    def reflect_rays(self, ray_origin, ray_direction):
        """
        Compute the direction of the reflected ray.

        Parameters:
        - ray_direction: The normalized direction vector of the incident ray.
        - normal: The normalized normal vector of the surface at the point of incidence.

        Returns:
        - The normalized direction vector of the reflected ray.
        """

        # Make sure the inputs are floats for dot product computation
        normal = self.normal().type(ray_direction.dtype)
        ray_direction = ray_direction

        # Compute the reflection direction
        reflected_dir = ray_direction - 2 * torch.dot(ray_direction, normal) * normal

        # Optionally, normalize the reflected ray direction
        reflected_dir = reflected_dir / torch.norm(reflected_dir)

        intersection_point = self.find_ray_plane_intersection(ray_origin, ray_direction)

        return intersection_point, reflected_dir

import cv2
import torch
import sys, os
import copy as cp

try:
    from .camera_cv import *
    from .geometry_direction import Direction
    except: pass
except:
    from camera_cv import *
    from geometry_direction import Direction
    except: pass

class CamerasOnSphere():
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.cameras = []
        self.device = device

    def generate_cameras_on_sphere():
        radius = self.cfg.radius
        directions = Direction.sample_directions(self.cfg.n, device=self.device)
        poses = directions.to_pose_on_sphere(distance=radius)
        



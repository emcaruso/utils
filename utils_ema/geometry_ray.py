import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys, os
from utils_ema.geometry_pose import *
from utils_ema.geometry_direction import Direction
from utils_ema.plot import *


class Ray():

    def __init__(self, origin=torch.FloatTensor([0,0,0]), direction=Direction(), units="meters", requires_grad=None):
        self.units = units
        self.device = input.device

import numpy as np
import sys, os
import torch
from .user_interface import *


class mover():  

    loc_speed = 0.1
    rot_speed = 0.001

    @staticmethod
    def wasd2v(keys):
        loc_v = torch.zeros([3])
        w = 'w' in keys
        s = 's' in keys
        a = 'a' in keys
        d = 'd' in keys
        if (w ^ s):
            if w: loc_v[2] = mover.loc_speed
            if s: loc_v[2] = -mover.loc_speed
        if (a ^ d):
            if a: loc_v[0] = -mover.loc_speed
            if d: loc_v[0] = mover.loc_speed
        return loc_v



if __name__=="__main__":
    user.detect_mouse_and_key()
    while True:
        v = mover.wasd2v(user.keys)
        print(v)
        time.sleep(0.1)



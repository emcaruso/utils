import numpy as np
import sys, os
import torch
import time

try:
    from .user_interface import User
    from .geometry_direction import Direction
except:
    from user_interface import User
    from geometry_direction import Direction

class MoverOrbital():

    def __init__(self, orbital_speed=0.001):
        self.direction = Direction(torch.FloatTensor([0,0]))
        self.orbital_speed = orbital_speed
        self.user = User
        self.user.detect_for_orbital()
        self.inc = [0,0]
        self.last_time = 0

    def get_pose(self, distance=8):
        azimuth_inc =  -0.01*self.user.pos_delta[0]
        elevation_inc =  0.01*self.user.pos_delta[1]
        dt = time.time()-self.last_time
        self.inc=[azimuth_inc/dt, elevation_inc/dt]
        self.direction.add_and_clamp(azimuth_inc, elevation_inc)
        # self.direction.add_and_clamp(azimuth_inc, 0)
        self.last_time = time.time()
        p = self.direction.to_pose_on_sphere(distance)
        return p


# class Mover():  

#     def __init__(self, loc_speed=0.1, rot_speed=0.001):
#         self.loc_speed = loc_speed
#         self.rot_speed = rot_speed
#         self.user = User
#         self.user.detect_for_orbital()
#         self.user.detect_key()

#     def wasd2v(self,keys):
#         loc_v = torch.zeros([3])
#         w = 'w' in keys
#         s = 's' in keys
#         a = 'a' in keys
#         d = 'd' in keys
#         if (w ^ s):
#             if w: loc_v[2] = self.loc_speed
#             if s: loc_v[2] = -self.loc_speed
#         if (a ^ d):
#             if a: loc_v[0] = -self.loc_speed
#             if d: loc_v[0] = self.loc_speed
#         return loc_v

#     def get_v(self):
#         return self.wasd2v(self.user.keys)

#     def get_w(self):
#         return self.user.pos_delta

#     @staticmethod
#     def get_moving_inputs():
#         self.user.detect_for_orbital()
#         self.user.detect_key()
#         while True:
#             v = Mover.wasd2v(self.user.keys)
#             w = self.user.pos_delta
#             print(v, w)
#             time.sleep(0.033)


if __name__=="__main__":

    mover = MoverOrbital()
    while True:
        mover.get_pose()
        print(mover.inc)
    # User.detect_for_orbital()
    # User.detect_key()
    # while True:
    #     v = User.get_v()
    #     w = User.get_w()
    #     print(v, w)
    #     time.sleep(0.1)



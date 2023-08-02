import numpy as np
import sys, os
import torch
sys.path.append(os.path.abspath('../user'))

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.abspath(current+"/..")
sys.path.insert(1, parent)
from user.user import *



# class cam():

#     def __init__(self):
#         self.loc = np.array([0,0,0])
#         self.eul = np.array([0,0,0])
#         self.rot = np.array([0,0,0])
#         self.loc_speed = 1
#         self.eul_speed = 1

#     def check_key_press():
#         while True:
#             try:
#                 # Check if any of the keys are pressed
#                 for key, name in key_mapping.items():
#                     if keyboard.is_pressed(key):
#                         print(f"The key {name} is pressed.")
#                         return  # Exit the function when a key is detected
#             except KeyboardInterrupt:
#                 # Exit the loop when Ctrl+C is pressed
#                 break



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
        # print(user.key)
        v = mover.wasd2v(user.keys)
        print(v)
        time.sleep(0.1)



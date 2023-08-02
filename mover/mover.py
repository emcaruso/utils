import numpy as np
from submodules.user.user import *


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
    def wasd2pose(self, key):
        loc_v = torch.zeros([3])
        if key=="w":
            loc_v[2] = loc_speed


if __name__=="__main__":
    while True:
        user.detect_key()
        print(user.key)



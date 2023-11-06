from pynput import keyboard
from pynput import mouse
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
from screeninfo import get_monitors
import numpy as np
import time


class User():
    keys = []
    res = [get_monitors()[0].width,get_monitors()[0].height]
    center = np.array([res[0]/2,res[1]/2])
    pos = center
    pos_last = center
    pos_delta =  [0,0]
    mouse_controller = MouseController()
    dt = 0.033
    # dt = 0

    @staticmethod
    def on_press(key):
        if key.char not in User.keys:
            User.keys.append(key.char)
        time.sleep(User.dt)

    @staticmethod
    def on_release(key):
        assert(key.char in User.keys)
        User.keys.remove(key.char)
        if key==keyboard.Key.esc:
            return False


    @staticmethod
    def movement_on_click(x,y,button,pressed):
        while True:
            px,py = User.mouse_controller.position

            User.pos_last = User.pos
            User.pos = [px,py]
            User.pos_delta = [ User.pos[0]-User.pos_last[0], User.pos[1]-User.pos_last[1] ]

            if User.pos[0]==0 or User.pos[1]==0 or User.pos[0]==User.res[0]-1 or User.pos[1]==User.res[1]-1:
                User.pos = User.center
                User.pos_last = User.center
                User.mouse_controller.position = tuple(User.center)

            # if User.pos_delta[0]==0 and User.pos_delta[1]==0:
            #     break

            time.sleep(User.dt)



    @staticmethod
    def detect_key():
        listener = keyboard.Listener(on_press=User.on_press, on_release=User.on_release)
        listener.start()

    @staticmethod
    def detect_for_orbital():
        listener = mouse.Listener(on_click=User.movement_on_click )
        listener.start()

    @staticmethod
    def detect_mouse_and_key():
        User.detect_key()
        User.detect_mouse_movement()

    @staticmethod
    def while_till_q( func ):
        User.detect_key()
        while True:
            if 'q' in User.keys:
                break
            else:
                func()


if __name__=="__main__":
    # User.detect_mouse_and_key()
    User.detect_for_orbital()
    while True:
        print(User.pos_delta)
        # print(User.pos_delta, User.keys)
        # print("pos_delta ",User.pos_delta)
        time.sleep(0.03)

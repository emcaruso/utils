from pynput import keyboard
from pynput import mouse
from pynput.mouse import Controller as MouseController
import numpy as np
import time


class user():
    # Mapping of keys to their corresponding names
    key_mapping = {
        keyboard.Key.up: "w",
        keyboard.Key.left: "a",
        keyboard.Key.down: "s",
        keyboard.Key.right: "d",
    }
    key = None
    center = np.array([400,300])
    pos = center
    pos_delta = None
    mouse_controller = MouseController()
    dt = 0.01

    @staticmethod
    def on_press(key):
        user.key = key
        time.sleep(user.dt)

    @staticmethod
    def on_release(key):
        user.key = None
        if key==keyboard.Key.esc:
            return False



    @staticmethod
    def on_click(x,y,button,pressed):
        return False

    @staticmethod
    def on_move(x,y):
        if not np.array_equal(user.pos, user.center):
            user.pos_delta = user.pos - user.center
            user.pos = user.center
            user.mouse_controller.position = tuple(user.center)
            time.sleep(user.dt)
        else:
            user.pos = np.array([x,y])
            user.pos_delta = None

    @staticmethod
    def detect_key():
        listener = keyboard.Listener(on_press=user.on_press, on_release=user.on_release)
        listener.start()

    @staticmethod
    def detect_mouse_movement():
        listener = mouse.Listener(on_move=user.on_move,on_click=user.on_click )
        listener.start()

if __name__=="__main__":
    user.detect_key()
    user.detect_mouse_movement()
    while True:
        print(user.pos_delta, user.key)
        time.sleep(0.01)
